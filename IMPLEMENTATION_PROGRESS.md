# Yatsaury — Implementation Progress

Tracks the MVP-first build order. See [DESIGN.md](./DESIGN.md) for full architecture.

**Status legend**: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Skeleton (foundation)

Goal: project boots, `yatsaury --help` works, empty test suite green.

- [ ] `pyproject.toml` via `uv init` (deps, ruff/mypy config, entry point `yatsaury = "yatsaury.cli:app"`)
- [ ] `src/yatsaury/` package layout (`__init__.py`, `__main__.py`)
- [ ] `models.py` — `Document`, `Chunk`, `Sample` (Pydantic v2)
- [ ] `config.py` — pydantic-settings `Settings` (layered loading)
- [ ] `cli.py` — empty Typer app with `--help`
- [ ] `.gitignore`, `.env.example`, `config.example.toml`, `README.md` stub
- [ ] **Verify**: `uv run yatsaury --help` lists commands; `uv run pytest` green

## Phase 1 — End-to-end vertical slice (MVP)

Goal: simplest full path produces valid JSONL.

- [ ] `sources/text.py` — `TextLoader` (raw text / `.txt` / `.md`) + `sources/base.py` protocol
- [ ] `processing/chunk.py` — token-aware recursive splitter (tiktoken)
- [ ] `llm/client.py` — `LLMClient` (openai SDK + `base_url`, JSON mode, tenacity retries)
- [ ] `llm/prompts.py` — grounding-first Q&A generation prompt
- [ ] `generators/qa.py` — `QaGenerator` + `generators/base.py` protocol & registry
- [ ] `exporters/jsonl.py` — `JsonlExporter` + `exporters/base.py` protocol & registry
- [ ] `pipeline.py` — minimal Orchestrator wiring the above
- [ ] `cli.py` — implement `generate` (text → qa → jsonl)
- [ ] `examples/sirah_sample.txt`
- [ ] **Verify**: `uv run yatsaury generate -i examples/sirah_sample.txt -t qa -f jsonl -o ./out --base-url http://localhost:11434/v1 --model llama3.1 --limit-chunks 2` → `out.jsonl` lines are valid JSON with `question`/`answer`/`supporting_quote`/`source_citation`

## Phase 2 — Real sources

Goal: usable on actual Sirah PDFs and web pages.

- [ ] `sources/pdf.py` — `PdfLoader` (PyMuPDF, page numbers) + `pypdf` fallback
- [ ] `sources/url.py` — `UrlLoader` (trafilatura + httpx)
- [ ] `sources/base.py` — `resolve_loader()` auto-detection
- [ ] `processing/clean.py` — normalization (whitespace, headers/footers, dehyphenation)
- [ ] `cli.py` — `inspect` command (load + chunk, print stats)
- [ ] **Verify**: `uv run yatsaury inspect -i <real sirah PDF>` prints sane chunk/token stats; generate from a URL succeeds

## Phase 3 — Grounding & quality (religious-accuracy core)

Goal: tool becomes trustworthy for religious content.

- [ ] Programmatic `supporting_quote` substring check (whitespace-fuzzy)
- [ ] `quality/verify.py` — LLM-judge grounding scorer + `--judge-model`
- [ ] `cli.py` — `--verify/--no-verify`, `--min-score` filtering; `verify` subcommand (re-score JSONL)
- [ ] `quality/dedup.py` — exact (normalized hash) + near-dup (`rapidfuzz`)
- [ ] `exporters/review_csv.py` — `CsvReviewExporter` (human-review CSV)
- [ ] `cli.py` — `export` builds final dataset from approved rows
- [ ] **Verify**: a deliberately unsupported sample is dropped by quote-check/judge; `--min-score` filters; review CSV opens in a spreadsheet; `export` keeps only `approved` rows

## Phase 4 — Remaining dataset types & formats

- [ ] `generators/instruction.py` — `InstructionGenerator`
- [ ] `generators/rag.py` — `RagGenerator` (no LLM; chunk + metadata)
- [ ] `generators/summary.py` — `SummaryGenerator`
- [ ] `exporters/hf.py` — `HfExporter` (`datasets` lib)
- [ ] `cli.py` — `--type all`, multi-format `-f` output
- [ ] **Verify**: `-t all -f hf` writes a dataset that round-trips via `datasets.load_from_disk()`

## Phase 5 — Knowledge-injection enhancements

- [ ] `--paraphrases` — multiple paraphrased Q&A per fact
- [ ] `--difficulty` — easy/medium/hard mix
- [ ] Optional fact-extraction step before generation (`fact_id` on samples)
- [ ] Bidirectional / inverse questions
- [ ] Coverage report (facts with too few samples)
- [ ] **Verify**: same fact appears as multiple varied samples; coverage report runs

## Phase 6 — Polish

- [ ] `cache.py` — content-hash disk cache / resumability
- [ ] `--dry-run` — plan + token/cost estimate, no LLM calls
- [ ] Embeddings-based near-dup (optional, via `/embeddings`)
- [ ] `rich` progress bars
- [ ] `config show` command
- [ ] README + examples, broaden test coverage
- [ ] **Verify**: re-run skips cached work; `--dry-run` prints estimate; `ruff check` + `mypy` clean

---

## Notes / Decisions Log

- 2026-06-18 — Project bootstrapped from planning. Confirmed: name `yatsaury`, tooling `uv`,
  OpenAI-compatible LLM client, JSONL + HF export, dataset language follows source (`--lang` override).
- _(add decisions/changes here as implementation proceeds)_
