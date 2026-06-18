# Yatsaury — Design Document

A CLI tool that turns source material (**PDF, web URL, or raw text**) into **training-ready
datasets**, primarily to **fine-tune an LLM with new knowledge**. The first domain is Islamic
content such as Sirah Nabawiyah. Because the content is religious, **faithfulness to the source
(grounding, no hallucination)** is the top priority.

## 1. Goals & Scope

- **Inputs**: PDF files, web URLs, raw text (pluggable source loaders).
- **Dataset types**: Q&A, instruction-tuning, RAG chunks, summary/paraphrase. Priority is fine-tuning.
- **Export formats**: JSONL and HuggingFace dataset.
- **Interface**: CLI.
- **LLM**: OpenAI-compatible client (configurable `base_url` + `api_key` + `model`); works with
  OpenAI, Ollama, vLLM, LM Studio, etc.
- **Dataset language**: follows the source language by default, overridable via `--lang`.
- **Non-goals (for now)**: web UI, model training/serving itself, multi-user/server deployment.

## 2. Pipeline Architecture

A linear pipeline of independent, swappable stages connected by typed Pydantic models:

```
[Source Loaders] -> [Extract & Clean] -> [Chunker] -> [Generators] -> [Grounding/Verify] -> [Dedup] -> [Exporters]
  PDF/URL/text        normalized text      Chunk[]      Sample[]          scored Sample[]      unique[]    JSONL / HF / CSV
```

Each stage consumes one typed model and produces the next, so any stage is unit-testable in
isolation and re-runnable from cache.

## 3. Core Data Models (`models.py`, Pydantic v2)

These are the contracts that decouple all stages.

- **`Document`** — `id`, `source_uri`, `source_type` (pdf/url/text), `raw_text`, `title`, `metadata`
  (page count, author, fetch date, …).
- **`Chunk`** — `id`, `doc_id`, `text`, `token_count`, `char_span` (start/end offsets for citation),
  `page` (for PDFs), `ordinal`.
- **`Sample`** — the unified generated record with a `dataset_type` discriminator
  (`qa`/`instruction`/`rag`/`summary`):
  `id`, `chunk_id`, `dataset_type`, `payload` (type-specific dict), `source_text`,
  `source_citation`, `supporting_quote`, `grounding_score`, `verified` (bool), `dedup_hash`, `lang`,
  optional `fact_id`.

A single `Sample` model (typed by `dataset_type`) lets grounding, dedup, and export stay generic.

## 4. Abstractions (Protocols/ABCs + string→impl registry)

Three plugin families; a registry resolves CLI string flags (e.g. `--source pdf`) to implementations.

- **`SourceLoader`** — `supports(uri) -> bool`, `load(uri) -> Document`.
  Implementations: `PdfLoader`, `UrlLoader`, `TextLoader`. A `resolve_loader(uri)` factory picks one
  by extension / URL scheme / `--source` override.
- **`Generator`** — `dataset_type: str`, `generate(chunk, n, llm) -> list[Sample]`.
  Implementations: `QaGenerator`, `InstructionGenerator`, `RagGenerator` (needs no LLM — packages
  chunk+metadata), `SummaryGenerator`.
- **`Exporter`** — `export(samples, out_path) -> None`.
  Implementations: `JsonlExporter`, `HfExporter`, `CsvReviewExporter`.

A thin **`Orchestrator`** (`pipeline.py`) wires the stages, owns config, isolates errors (one bad
chunk must not kill the run), drives progress, and uses a content-hash disk cache (`.cache/`) for
resumability (skip already-completed extraction/generation on re-runs).

## 5. Tech Stack

| Concern | Choice | Rationale |
|---|---|---|
| Python / tooling | 3.11+, **uv**, Hatchling backend | fast single-tool deps+venv; modern stdlib |
| CLI | **Typer** | type-hint subcommands, auto `--help` |
| Config | **pydantic-settings** | layered env / `.env` / TOML into a typed model |
| Models / validation | **Pydantic v2** | validation + JSON (de)serialization for cache/export |
| PDF | **PyMuPDF** (+ `pypdf` fallback) | best text fidelity + page numbers for citations |
| Web | **trafilatura** + **httpx** | main-content extraction, strips boilerplate |
| Chunking | **tiktoken** + custom token-aware recursive splitter | accurate token budgets, lean deps; Unicode-aware for Arabic/transliteration |
| LLM | official **`openai`** SDK with `base_url`, JSON mode | single client across all OpenAI-compatible backends; wrapped in our `LLMClient` |
| Retry/backoff | **tenacity** | robust around flaky LLM/HTTP calls |
| HF export | **`datasets`** | `Dataset.from_list().save_to_disk()` / `push_to_hub()` |
| UX | **rich** | progress bars, logging |
| Quality | **ruff**, **mypy**, **pytest** (mocked LLM/HTTP) | lint/format, types, deterministic free tests |

## 6. Project Structure

```
src/yatsaury/
  __main__.py            # python -m yatsaury
  cli.py                 # Typer app + subcommands         [critical]
  config.py              # pydantic-settings Settings (layered)
  models.py              # Document, Chunk, Sample          [critical]
  pipeline.py            # Orchestrator wiring all stages   [critical]
  cache.py               # content-hash disk cache
  llm/
    client.py            # LLMClient (openai SDK, retries, JSON mode)
    prompts.py           # grounding-first gen + judge prompts [critical]
  sources/
    base.py              # SourceLoader protocol + resolve_loader()
    pdf.py  url.py  text.py
  processing/
    clean.py             # whitespace, header/footer, dehyphenation
    chunk.py             # token-aware recursive splitter
  generators/
    base.py              # Generator protocol + registry
    qa.py  instruction.py  rag.py  summary.py
  quality/
    verify.py            # quote-check + LLM-judge grounding gate [critical]
    dedup.py             # exact + near-dup
  exporters/
    base.py              # Exporter protocol + registry
    jsonl.py  hf.py  review_csv.py
tests/
  conftest.py  fixtures/ (sample PDF, recorded LLM/HTML)  test_*.py
examples/sirah_sample.txt
pyproject.toml  README.md  .env.example  config.example.toml  .gitignore
```

`src/` layout so tests run against the installed package. Entry point: `yatsaury = "yatsaury.cli:app"`.

## 7. CLI Design

**Verbs**:
- `generate` — main end-to-end run (load → chunk → generate → verify → dedup → export).
- `inspect` — load + chunk only; print chunk stats. Cheap dry-run before spending tokens.
- `verify` — re-score an existing JSONL without regenerating.
- `export` — convert existing JSONL → HF/CSV, or build the final dataset from a reviewed file.
- `config show` — print resolved config with secrets masked.

**Key `generate` flags**: `-i/--input` (repeatable path/URL/`-`), `--source auto|pdf|url|text`,
`-t/--type qa|instruction|rag|summary|all` (repeatable), `-o/--output`, `-f/--format jsonl|hf|csv`
(repeatable), `--chunk-size`/`--chunk-overlap`, `-n/--per-chunk`, `--paraphrases`, `--difficulty`,
`--verify/--no-verify`, `--min-score`, `--model`/`--base-url`/`--api-key`, `--judge-model`,
`--dedup/--no-dedup`, `--limit-chunks`, `--lang auto|id|en|ar`, `--dry-run`.

**Examples**:
```bash
# Smoke test on local Ollama, cheap
yatsaury generate -i sirah.pdf -t qa -n 2 --limit-chunks 3 \
  --base-url http://localhost:11434/v1 --model llama3.1

# Full grounded build, multiple sources/types, HF + review CSV
yatsaury generate -i sirah_ibn_hisham.pdf -i https://example.org/seerah \
  -t qa -t instruction -t rag --paraphrases 3 --difficulty easy,medium,hard \
  --verify --min-score 0.8 -f hf -f csv -o ./datasets/seerah_v1

# Inspect chunking before spending tokens
yatsaury inspect -i sirah.pdf --chunk-size 600

# Build final HF dataset from human-approved rows
yatsaury export -i ./datasets/seerah_v1/reviewed.jsonl -f hf -o ./datasets/final
```

## 8. Grounding & Quality Strategy (religious-accuracy core)

Layered defense — no single mechanism is trusted alone:

1. **Generation-time grounding** — prompts force answers derived *only* from the provided chunk;
   each sample must include a literal `supporting_quote` copied from the chunk; if the chunk is
   insufficient the model emits `{"insufficient": true}` (dropped); the system prompt forbids
   outside knowledge (no invented dates/names/rulings). JSON mode + Pydantic validate-or-retry.
2. **Programmatic quote check** — verify `supporting_quote` is a real (whitespace-fuzzy) substring
   of the chunk. Deterministic anti-hallucination gate, no LLM needed.
3. **LLM-as-judge** (`quality/verify.py`) — a second call returns `grounding_score` (0–1),
   `is_supported`, and a rationale; use a stronger/different `--judge-model` when available. Drop
   samples below `--min-score`; record the score on every kept sample.
4. **Citation** — every sample carries `source_citation` (title + page + char span) and
   `source_text` for full traceability back to the book.
5. **Dedup** (`quality/dedup.py`) — exact (normalized hash) + near-dup (`rapidfuzz`, or embeddings
   via the same OpenAI-compatible `/embeddings` endpoint if available); keep highest-scoring of a cluster.
6. **Human review loop** — `review_csv.py` exports
   `question, answer, supporting_quote, source_citation, grounding_score, page, approved`; a
   reviewer/scholar marks `approved`; `export` then builds the final dataset only from approved
   rows. Recommended default before publishing religious datasets.

**Recommended default**: generation grounding + programmatic quote check + LLM judge with
`--min-score 0.7`, then human review CSV before publishing.

## 9. Knowledge-Injection Techniques

Fine-tuning teaches style/format more than facts; to make facts stick, maximize coverage and
redundancy of each fact across varied surface forms:

- **`--paraphrases N`** — multiple differently-worded Q&A per fact (highest-leverage for retention).
- **`--difficulty`** — recall / comprehension / multi-hop questions over the same passage.
- **Format diversity** — emit the same fact as Q&A, as an instruction sample, and within a summary.
- **Bidirectional/inverse questions** — ask both directions to avoid the reversal curse.
- **Atomic single-fact Q&A**; optional **fact-extraction step** (phase 5) to guarantee coverage;
  carry a `fact_id` for coverage reporting (facts with too few samples).
- README note: for genuinely new facts, **RAG often beats fine-tuning** — the `rag` type + citations
  also yields a retrieval corpus; recommend a hybrid (fine-tune for style + RAG for fact lookup).

## 10. Config & Secrets

A single pydantic-settings `Settings` (`config.py`), precedence high→low:

1. CLI flags (`--model`, `--base-url`, `--api-key`, …)
2. Env vars (`YATSAURY_*`; also accept `OPENAI_API_KEY` / `OPENAI_BASE_URL` as fallbacks)
3. `.env` file (git-ignored)
4. `config.toml` (non-secret defaults: chunk size, per-chunk, default types, min-score, judge model)
5. Built-in defaults

Ship `.env.example` and `config.example.toml`. Secrets only via env/`.env`, never committed;
`config show` masks the key. `.gitignore`: `.env`, `.cache/`, `datasets/`, `.venv/`.

## 11. Testing Strategy

- `pytest` with mocked HTTP (respx/responses) and recorded LLM fixtures → deterministic and free.
- Per-stage tests: sources, chunking, generators, verify (quote-check + judge), dedup, exporters.
- Local Ollama for free end-to-end manual runs.
- `ruff check` + `mypy` in CI/local pre-commit.
