# Yatsaury — Design Document

A CLI tool that turns source material (**PDF, web URL, or raw text**) into **training-ready
datasets**, primarily to **fine-tune an LLM with new knowledge**. The first domain is Islamic
content such as Sirah Nabawiyah. Because the content is religious, **faithfulness to the source
(grounding, no hallucination)** is the top priority.

## 1. Goals & Scope

- **Inputs**: PDF files, web URLs, raw text (pluggable source loaders).
- **Dataset types**: Q&A, instruction-tuning, RAG chunks, summary/paraphrase. Priority is fine-tuning.
- **Record schemas**: multiple trainer-ready layouts (Alpaca, ShareGPT, ChatML/OpenAI-messages,
  plain-QA, completion, RAG record, raw) — pluggable and extensible.
- **Serialization formats**: JSONL, HuggingFace dataset, and CSV (for human review).
- **Interface**: CLI.
- **LLM**: OpenAI-compatible client (configurable `base_url` + `api_key` + `model`); works with
  OpenAI, Ollama, vLLM, LM Studio, etc.
- **Dataset language**: follows the source language by default, overridable via `--lang`.
- **Non-goals (for now)**: web UI, model training/serving itself, multi-user/server deployment.

## 2. The Three Axes (key design idea)

Output is the product of **three independent, composable axes**. Keeping them separate is what makes
the tool flexible without code duplication — add one new schema and *every* dataset type and
serialization format can use it for free.

| Axis | Question it answers | Stage that owns it | CLI flag | Examples |
|---|---|---|---|---|
| **`dataset_type`** | *What* is the content? | Generators | `--type` | `qa`, `instruction`, `rag`, `summary` |
| **`record schema`** | *How* is each record shaped for the trainer? | Schema adapters | `--schema` | `chatml`, `sharegpt`, `alpaca`, `qa`, `completion`, `rag`, `raw` |
| **`serialization`** | *How* is it stored on disk? | Exporters | `--format` | `jsonl`, `hf`, `csv` |

Flow: **generate** a neutral `Sample` → **render** it to a chosen record schema (a plain dict) →
**serialize** that dict to a chosen file format.

```
[Generators] -> Sample (semantic, schema-neutral) -> [Schema Adapter] -> trainer record (dict) -> [Exporter] -> file
                  dataset_type=qa/instruction/...        --schema                                    --format
```

## 3. Pipeline Architecture

```
[Source Loaders] -> [Extract & Clean] -> [Chunker] -> [Generators] -> [Grounding/Verify] -> [Dedup] -> [Schema Adapter] -> [Exporters]
  PDF/URL/text        normalized text      Chunk[]      Sample[]          scored Sample[]      unique[]    record dicts        JSONL / HF / CSV
```

Each stage consumes one typed object and produces the next, so any stage is unit-testable in
isolation and re-runnable from cache.

## 4. Core Data Models (`models.py`, Pydantic v2)

The contracts that decouple all stages.

- **`Document`** — `id`, `source_uri`, `source_type` (pdf/url/text), `raw_text`, `title`, `metadata`.
- **`Chunk`** — `id`, `doc_id`, `text`, `token_count`, `char_span`, `page`, `ordinal`.
- **`Citation`** — `title`, `page`, `char_span`, `source_uri`.
- **`Sample`** — the unified, **schema-neutral** generated record. The same `Sample` can be rendered
  into any record schema later.

```jsonc
{
  "id": "smp_01H...",
  "chunk_id": "chk_0007",
  "dataset_type": "qa",            // qa | instruction | rag | summary
  "payload": { /* type-specific, see §5 */ },
  "source_text": "…the chunk the model was grounded on…",
  "supporting_quote": "an exact span copied from source_text",
  "source_citation": { "title": "Sirah Ibn Hisham", "page": 42,
                       "char_span": [1200, 1280], "source_uri": "sirah.pdf" },
  "grounding_score": 0.96,
  "verified": true,
  "dedup_hash": "sha256:…",
  "fact_id": "fact_0042",          // optional, groups paraphrases of one fact
  "lang": "id"
}
```

The `payload` is the only part that varies by `dataset_type`; everything else (grounding, citation,
dedup, language) is uniform so quality and export stages stay generic.

## 5. `payload` shapes per `dataset_type`

```jsonc
// dataset_type: "qa"
"payload": {
  "question": "Siapa istri pertama Nabi Muhammad ﷺ?",
  "answer": "Khadijah binti Khuwailid radhiyallahu 'anha.",
  "difficulty": "easy"            // easy | medium | hard (optional)
}

// dataset_type: "instruction"
"payload": {
  "instruction": "Sebutkan istri pertama Nabi Muhammad ﷺ dan jelaskan singkat.",
  "input": "",                    // optional extra context
  "output": "Istri pertama beliau adalah Khadijah binti Khuwailid …"
}

// dataset_type: "summary"
"payload": {
  "passage": "…original passage…",
  "summary": "Ringkasan padat dari passage.",
  "key_points": ["poin 1", "poin 2"]   // optional
}

// dataset_type: "rag"  (no LLM generation — packaged chunk + metadata)
"payload": {
  "text": "…the chunk text…",
  "title": "Sirah Ibn Hisham",
  "page": 42,
  "char_span": [1200, 1480]
}
```

## 6. Record schemas & concrete examples

A **record schema** renders a `Sample` into a flat dict matching a known training convention. Below,
the *same* Q&A `Sample` from §4 rendered into each schema. Not every schema fits every type — see the
compatibility matrix at the end.

**`qa`** — plain question/answer (simplest):
```json
{"question": "Siapa istri pertama Nabi Muhammad ﷺ?", "answer": "Khadijah binti Khuwailid radhiyallahu 'anha."}
```

**`alpaca`** — instruction/input/output (Stanford Alpaca):
```json
{"instruction": "Siapa istri pertama Nabi Muhammad ﷺ?", "input": "", "output": "Khadijah binti Khuwailid radhiyallahu 'anha."}
```

**`sharegpt`** — conversation turns (Axolotl/many SFT pipelines):
```json
{"conversations": [
  {"from": "human", "value": "Siapa istri pertama Nabi Muhammad ﷺ?"},
  {"from": "gpt", "value": "Khadijah binti Khuwailid radhiyallahu 'anha."}
]}
```

**`chatml`** — OpenAI/messages chat format (default; configurable system prompt):
```json
{"messages": [
  {"role": "system", "content": "You are a knowledgeable, precise assistant on Islamic history (Sirah)."},
  {"role": "user", "content": "Siapa istri pertama Nabi Muhammad ﷺ?"},
  {"role": "assistant", "content": "Khadijah binti Khuwailid radhiyallahu 'anha."}
]}
```

**`completion`** — prompt/completion (legacy text completion FT):
```json
{"prompt": "Siapa istri pertama Nabi Muhammad ﷺ?\n\n", "completion": " Khadijah binti Khuwailid radhiyallahu 'anha."}
```

**`rag`** — retrieval record (for the `rag` dataset_type), carries text + metadata for indexing:
```json
{"id": "chk_0007", "text": "…the chunk text…", "title": "Sirah Ibn Hisham", "page": 42, "source": "sirah.pdf", "char_span": [1200, 1480]}
```

**`raw`** — passthrough of the full `Sample` (with grounding metadata), for debugging/audit:
```json
{"id": "smp_01H...", "dataset_type": "qa", "payload": {"question": "…", "answer": "…"},
 "supporting_quote": "…", "source_citation": {"title": "Sirah Ibn Hisham", "page": 42},
 "grounding_score": 0.96, "lang": "id"}
```

> Optionally, `supporting_quote`/`source_citation` can be appended into the answer/output for the
> most important facts (`--cite-in-answer`) so the model learns precise wording — valuable for
> religious text. Off by default.

### Compatibility matrix (schema × dataset_type)

| record schema \ type | qa | instruction | summary | rag |
|---|:--:|:--:|:--:|:--:|
| `qa`        | ✓ | ✓ (instruction→Q, output→A) | – | – |
| `alpaca`    | ✓ | ✓ | ✓ (instruction="Summarize…", input=passage, output=summary) | – |
| `sharegpt`  | ✓ | ✓ | ✓ | – |
| `chatml`    | ✓ | ✓ | ✓ | – |
| `completion`| ✓ | ✓ | ✓ | – |
| `rag`       | – | – | – | ✓ |
| `raw`       | ✓ | ✓ | ✓ | ✓ |

A schema adapter declares which `dataset_type`s it supports; the pipeline skips (and logs)
incompatible combinations instead of emitting malformed records.

## 7. Abstractions (Protocols/ABCs + string→impl registry)

Four plugin families. Each has a registry so CLI string flags (e.g. `--schema sharegpt`) resolve to
implementations; adding a new variant = one new class + one registration line.

- **`SourceLoader`** — `supports(uri) -> bool`, `load(uri) -> Document`.
  Impls: `PdfLoader`, `UrlLoader`, `TextLoader`; `resolve_loader(uri)` factory.
- **`Generator`** — `dataset_type: str`, `generate(chunk, n, llm) -> list[Sample]`.
  Impls: `QaGenerator`, `InstructionGenerator`, `RagGenerator` (no LLM), `SummaryGenerator`.
- **`SchemaAdapter`** — `name: str`, `supports(dataset_type) -> bool`, `render(sample) -> dict`.
  Impls: `QaSchema`, `AlpacaSchema`, `ShareGptSchema`, `ChatmlSchema`, `CompletionSchema`,
  `RagSchema`, `RawSchema`. **This is the new axis** that cleanly separates content from layout.
- **`Exporter`** — `export(records: Iterable[dict], out_path) -> None` (operates on already-rendered
  dicts, so it is schema-agnostic). Impls: `JsonlExporter`, `HfExporter`, `CsvReviewExporter`.

A thin **`Orchestrator`** (`pipeline.py`) wires the stages, owns config, isolates errors (one bad
chunk must not kill the run), drives progress, and uses a content-hash disk cache (`.cache/`) for
resumability.

## 8. Tech Stack

| Concern | Choice | Rationale |
|---|---|---|
| Python / tooling | 3.11+, **uv**, Hatchling backend | fast single-tool deps+venv; modern stdlib |
| CLI | **Typer** | type-hint subcommands, auto `--help` |
| Config | **pydantic-settings** | layered env / `.env` / TOML into a typed model |
| Models / validation | **Pydantic v2** | validation + JSON (de)serialization for cache/export |
| PDF | **PyMuPDF** (+ `pypdf` fallback) | best text fidelity + page numbers for citations |
| Web | **trafilatura** + **httpx** | main-content extraction, strips boilerplate |
| Chunking | **tiktoken** + custom token-aware recursive splitter | accurate token budgets, lean deps; Unicode-aware for Arabic/transliteration |
| LLM | official **`openai`** SDK with `base_url`, JSON mode | single client across all OpenAI-compatible backends; wrapped in `LLMClient` |
| Retry/backoff | **tenacity** | robust around flaky LLM/HTTP calls |
| HF export | **`datasets`** | `Dataset.from_list().save_to_disk()` / `push_to_hub()` |
| UX | **rich** | progress bars, logging |
| Quality | **ruff**, **mypy**, **pytest** (mocked LLM/HTTP) | lint/format, types, deterministic free tests |

## 9. Project Structure

```
src/yatsaury/
  __main__.py            # python -m yatsaury
  cli.py                 # Typer app + subcommands         [critical]
  config.py              # pydantic-settings Settings (layered)
  models.py              # Document, Chunk, Citation, Sample [critical]
  pipeline.py            # Orchestrator wiring all stages   [critical]
  cache.py               # content-hash disk cache
  llm/
    client.py            # LLMClient (openai SDK, retries, JSON mode)
    prompts.py           # grounding-first gen + judge prompts [critical]
  sources/
    base.py  pdf.py  url.py  text.py
  processing/
    clean.py  chunk.py
  generators/
    base.py  qa.py  instruction.py  rag.py  summary.py
  schemas/               # NEW axis: record-schema adapters
    base.py              # SchemaAdapter protocol + registry [critical]
    qa.py  alpaca.py  sharegpt.py  chatml.py  completion.py  rag.py  raw.py
  quality/
    verify.py            # quote-check + LLM-judge grounding gate [critical]
    dedup.py
  exporters/
    base.py  jsonl.py  hf.py  review_csv.py
tests/
  conftest.py  fixtures/  test_*.py
examples/sirah_sample.txt
pyproject.toml  README.md  .env.example  config.example.toml  .gitignore
```

`src/` layout so tests run against the installed package. Entry point: `yatsaury = "yatsaury.cli:app"`.

## 10. CLI Design

**Verbs**: `generate` (end-to-end), `inspect` (load+chunk stats), `verify` (re-score JSONL),
`export` (re-render/re-serialize an existing dataset, or build final from a reviewed file),
`schemas` (list available record schemas + their compatible types), `config show`.

**Key `generate` flags**: `-i/--input` (repeatable), `--source auto|pdf|url|text`,
`-t/--type qa|instruction|rag|summary|all` (repeatable),
`-s/--schema chatml|sharegpt|alpaca|qa|completion|rag|raw` (repeatable),
`-f/--format jsonl|hf|csv` (repeatable), `-o/--output`, `--chunk-size`/`--chunk-overlap`,
`-n/--per-chunk`, `--paraphrases`, `--difficulty`, `--verify/--no-verify`, `--min-score`,
`--system-prompt` (for `chatml`), `--cite-in-answer`, `--model`/`--base-url`/`--api-key`,
`--judge-model`, `--dedup/--no-dedup`, `--limit-chunks`, `--lang auto|id|en|ar`, `--dry-run`.

**Defaults**: `--schema chatml` for qa/instruction/summary, `rag` schema for the rag type;
`--format jsonl`. Incompatible type×schema pairs are skipped with a warning.

**Examples**:
```bash
# Q&A as ChatML JSONL (modern chat fine-tuning), cheap smoke test on local Ollama
yatsaury generate -i sirah.pdf -t qa -s chatml -f jsonl -o ./out \
  --base-url http://localhost:11434/v1 --model llama3.1 --limit-chunks 3

# One generation run, emitted in MULTIPLE schemas AND formats at once
yatsaury generate -i sirah_ibn_hisham.pdf -t qa -t instruction \
  -s chatml -s sharegpt -s alpaca -f jsonl -f hf \
  --paraphrases 3 --difficulty easy,medium,hard --verify --min-score 0.8 -o ./datasets/seerah_v1

# RAG corpus from a URL
yatsaury generate -i https://example.org/seerah -t rag -s rag -f jsonl -o ./datasets/rag

# List schemas and which dataset types each supports
yatsaury schemas

# Re-render an existing dataset into a new schema without regenerating (no LLM cost)
yatsaury export -i ./datasets/seerah_v1/qa.jsonl -s sharegpt -f hf -o ./datasets/seerah_sharegpt

# Build final HF dataset from human-approved rows
yatsaury export -i ./datasets/seerah_v1/reviewed.csv -s chatml -f hf -o ./datasets/final
```

## 11. Grounding & Quality Strategy (religious-accuracy core)

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
4. **Citation** — every sample carries `source_citation` and `source_text` for full traceability.
5. **Dedup** (`quality/dedup.py`) — exact (normalized hash) + near-dup (`rapidfuzz`, or embeddings
   via the OpenAI-compatible `/embeddings` endpoint if available); keep highest-scoring of a cluster.
6. **Human review loop** — `CsvReviewExporter` writes
   `question, answer, supporting_quote, source_citation, grounding_score, page, approved`; a
   reviewer marks `approved`; `export` then builds the final dataset (in any schema/format) only
   from approved rows. Recommended default before publishing religious datasets.

**Recommended default**: generation grounding + programmatic quote check + LLM judge with
`--min-score 0.7`, then human review CSV before publishing.

## 12. Knowledge-Injection Techniques

Fine-tuning teaches style/format more than facts; to make facts stick, maximize coverage and
redundancy of each fact across varied surface forms:

- **`--paraphrases N`** — multiple differently-worded Q&A per fact (highest-leverage for retention);
  paraphrases of one fact share a `fact_id`.
- **`--difficulty`** — recall / comprehension / multi-hop over the same passage.
- **Format diversity** — emit the same fact as multiple `dataset_type`s and `schema`s.
- **Bidirectional/inverse questions** — both directions, to avoid the reversal curse.
- **Atomic single-fact Q&A**; optional fact-extraction step (phase 5) for coverage; `fact_id`-based
  coverage reporting (facts with too few samples).
- README note: for genuinely new facts, **RAG often beats fine-tuning** — the `rag` type + citations
  also yields a retrieval corpus; recommend a hybrid (fine-tune for style + RAG for fact lookup).

## 13. Config & Secrets

A single pydantic-settings `Settings` (`config.py`), precedence high→low:

1. CLI flags (`--model`, `--base-url`, `--api-key`, …)
2. Env vars (`YATSAURY_*`; also accept `OPENAI_API_KEY` / `OPENAI_BASE_URL` as fallbacks)
3. `.env` file (git-ignored)
4. `config.toml` (non-secret defaults: chunk size, per-chunk, default types/schemas, min-score, judge model)
5. Built-in defaults

Ship `.env.example` and `config.example.toml`. Secrets only via env/`.env`, never committed;
`config show` masks the key. `.gitignore`: `.env`, `.cache/`, `datasets/`, `.venv/`.

## 14. Testing Strategy

- `pytest` with mocked HTTP (respx/responses) and recorded LLM fixtures → deterministic and free.
- Per-stage tests: sources, chunking, generators, **schema adapters (render snapshots per schema)**,
  verify (quote-check + judge), dedup, exporters.
- Schema round-trip tests: a fixed `Sample` renders to the expected dict for every schema; the
  compatibility matrix is enforced (unsupported pairs are skipped, not malformed).
- Local Ollama for free end-to-end manual runs. `ruff check` + `mypy` in local/CI.
