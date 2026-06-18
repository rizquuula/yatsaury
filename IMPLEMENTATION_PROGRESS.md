# Yatsaury — Implementation Progress (TDD)

MVP-first build order, driven by **Test-Driven Development**. See [DESIGN.md](./DESIGN.md) for architecture.

**Status legend**: `[ ]` not started · `[~]` in progress · `[x]` done

**Item icons**: 🔴 write test first (must fail) · 🟢 implementation (only after its 🔴) · 🔵 refactor ·
plain box = scaffolding/config/docs, **no test by design**. Every code/logic task is a 🔴→🟢 pair.

## TDD workflow (rules)

Every unit of work follows **red → green → refactor**:

1. 🔴 **Red** — write the test first; run it; confirm it *fails* for the right reason.
2. 🟢 **Green** — write the *minimum* code to make it pass.
3. 🔵 **Refactor** — clean up with the test still green.

Rules of this file:
- **A 🟢 implementation box may not be checked until its paired 🔴 test box is checked** (test exists and was seen failing first).
- `uv run pytest` must be green at the end of every item before moving on.
- LLM and HTTP are **always mocked** in unit tests (recorded fixtures) → deterministic and free.
  End-to-end items may run against local Ollama and are marked `@pytest.mark.e2e` (opt-in, not in default run).
- Each phase ends with `uv run pytest && uv run ruff check && uv run mypy` clean.

---

## Phase 0 — Skeleton & test harness

Goal: project boots, `yatsaury --help` works, test infrastructure runs.

- [x] `pyproject.toml` via `uv init` (deps incl. pytest/respx, ruff/mypy config, entry point `yatsaury = "yatsaury.cli:app"`)
- [x] `src/yatsaury/` package layout (`__init__.py`, `__main__.py`)
- [x] Test harness: `tests/conftest.py`, `tests/fixtures/`, pytest config (markers incl. `e2e`), coverage
- [x] 🔴 `test_models.py` — `Document`/`Chunk`/`Citation`/`Sample` validate, (de)serialize, reject bad input
- [x] 🟢 `models.py` — implement the Pydantic v2 models
- [x] 🔴 `test_config.py` — precedence flag > env > `.env` > toml > default; secret masking
- [x] 🟢 `config.py` — pydantic-settings `Settings` (layered)
- [x] 🔴 `test_cli_smoke.py` — Typer `--help` exits 0 and lists verbs (CliRunner)
- [x] 🟢 `cli.py` — empty Typer app + stubbed subcommands
- [x] `.gitignore`, `.env.example`, `config.example.toml`, `README.md` stub
- [x] **Phase gate**: `uv run pytest` green; `uv run yatsaury --help` works

## Phase 1 — End-to-end vertical slice (MVP)

Goal: simplest full path produces valid records, proven by tests.

- [x] 🔴 `test_sources_text.py` — `TextLoader` loads str/`.txt`/`.md` → `Document`
- [x] 🟢 `sources/base.py` (protocol) + `sources/text.py`
- [x] 🔴 `test_chunk.py` — token-aware splitter respects size/overlap, sets `char_span`, never splits mid-token boundary wrongly
- [x] 🟢 `processing/chunk.py` (tiktoken)
- [x] 🔴 `test_llm_client.py` — `LLMClient` parses JSON-mode output, retries on failure (mocked openai), passes `base_url`
- [x] 🟢 `llm/client.py` (openai SDK + tenacity) + `llm/prompts.py` (grounding-first Q&A prompt)
- [x] 🔴 `test_generators_qa.py` — `QaGenerator` turns a chunk + mocked LLM response into valid `Sample`s; drops `{"insufficient": true}`
- [x] 🟢 `generators/base.py` (protocol + registry) + `generators/qa.py`
- [x] 🔴 `test_schemas.py` — fixed `Sample` renders to expected dict for `chatml` and `qa` (snapshot); `supports()` correct
- [x] 🟢 `schemas/base.py` (protocol + registry) + `schemas/chatml.py` + `schemas/qa.py`
- [x] 🔴 `test_exporters_jsonl.py` — `JsonlExporter` writes one valid JSON object per line from rendered dicts
- [x] 🟢 `exporters/base.py` (protocol + registry) + `exporters/jsonl.py`
- [x] 🔴 `test_pipeline.py` — Orchestrator wires generate → render → serialize (all mocked); one bad chunk doesn't abort the run
- [x] 🟢 `pipeline.py` (minimal Orchestrator)
- [x] 🔴 `test_cli_generate.py` — `generate` (CliRunner, mocked LLM) produces ChatML JSONL; `-s qa` produces Q&A from same run
- [x] 🟢 `cli.py` — implement `generate` with `-t/-s/-f` + `examples/sirah_sample.txt`
- [x] 🔴 `test_e2e_generate.py` (`@e2e`) — real local Ollama run yields valid records
- [x] **Phase gate**: unit suite green; opt-in e2e passes against Ollama

## Phase 2 — Real sources

- [x] 🔴 `test_sources_pdf.py` — `PdfLoader` extracts text + page numbers from a tiny fixture PDF (`pypdf` fallback path tested too)
- [x] 🟢 `sources/pdf.py` (PyMuPDF + `pypdf` fallback)
- [x] 🔴 `test_sources_url.py` — `UrlLoader` extracts main content from a recorded HTML fixture (mocked httpx), strips boilerplate
- [x] 🟢 `sources/url.py` (trafilatura + httpx)
- [x] 🔴 `test_resolve_loader.py` — picks loader by extension / URL scheme / `--source` override
- [x] 🟢 `sources/base.py` — `resolve_loader()`
- [x] 🔴 `test_clean.py` — normalizes whitespace, strips repeated headers/footers, de-hyphenates line breaks
- [x] 🟢 `processing/clean.py`
- [x] 🔴 `test_cli_inspect.py` — `inspect` prints chunk count/token stats (CliRunner)
- [x] 🟢 `cli.py` — `inspect` command
- [x] **Phase gate**: suite green on real PDF/URL fixtures

## Phase 3 — Grounding & quality (religious-accuracy core)

- [x] 🔴 `test_quote_check.py` — `supporting_quote` substring check passes on real spans, fails on invented text (whitespace-fuzzy)
- [x] 🟢 programmatic quote-check in `quality/verify.py`
- [x] 🔴 `test_verify_judge.py` — LLM-judge (mocked) returns `grounding_score`/`is_supported`; samples below `--min-score` dropped
- [x] 🟢 `quality/verify.py` — LLM-judge scorer + `--judge-model`
- [x] 🔴 `test_dedup.py` — exact dups collapse; near-dups (rapidfuzz) collapse keeping highest score
- [x] 🟢 `quality/dedup.py`
- [x] 🔴 `test_review_csv.py` — `CsvReviewExporter` writes expected columns incl. `approved`; round-trips back
- [x] 🟢 `exporters/review_csv.py`
- [x] 🔴 `test_cli_verify_export.py` — `verify` re-scores a JSONL; `export` keeps only `approved` rows
- [x] 🟢 `cli.py` — `--verify/--no-verify`, `--min-score`; `verify` + `export` subcommands
- [x] **Phase gate**: an unsupported sample is provably dropped end-to-end

## Phase 4 — Remaining dataset types, schemas & formats

- [x] 🔴 `test_generators_instruction.py` / `_summary.py` / `_rag.py` (rag = no LLM)
- [x] 🟢 `generators/instruction.py`, `generators/summary.py`, `generators/rag.py`
- [x] 🔴 `test_schemas.py` (extend) — render snapshots for `alpaca`/`sharegpt`/`completion`/`rag`/`raw`; **compatibility matrix enforced** (unsupported type×schema skipped + logged, never malformed)
- [x] 🟢 `schemas/alpaca.py`, `sharegpt.py`, `completion.py`, `rag.py`, `raw.py`
- [x] 🔴 `test_exporters_hf.py` — `HfExporter` output round-trips via `datasets.load_from_disk()`
- [x] 🟢 `exporters/hf.py`
- [x] 🔴 `test_cli_schemas_multi.py` — `schemas` lists adapters+types; `generate -t all -s a -s b -f jsonl -f hf` emits one dataset per schema; incompatible pairs skipped with warning
- [x] 🟢 `cli.py` — `schemas` command, `--system-prompt`, `--cite-in-answer`, multi `-t/-s/-f`
- [x] **Phase gate**: full matrix covered by tests

## Phase 5 — Knowledge-injection enhancements

- [x] 🔴 `test_paraphrases.py` — N paraphrases per fact share a `fact_id`
- [x] 🟢 `--paraphrases`
- [x] 🔴 `test_difficulty.py` — easy/medium/hard mix produced
- [x] 🟢 `--difficulty`
- [x] 🔴 `test_fact_extraction.py` — fact-extraction step yields atomic facts; Q&A generated per fact
- [x] 🟢 optional fact-extraction step (`fact_id` on samples)
- [x] 🔴 `test_bidirectional.py` — inverse question generated for a fact
- [x] 🟢 bidirectional / inverse questions
- [x] 🔴 `test_coverage_report.py` — flags facts with too few samples
- [x] 🟢 coverage report
- [x] **Phase gate**: suite green

## Phase 6 — Polish

- [ ] 🔴 `test_cache.py` — second run is a cache hit (no LLM/extraction re-call) for same input+config
- [ ] 🟢 `cache.py` — content-hash disk cache / resumability
- [ ] 🔴 `test_dry_run.py` — `--dry-run` prints plan + token/cost estimate, makes zero LLM calls
- [ ] 🟢 `--dry-run`
- [ ] 🔴 `test_embeddings_dedup.py` — embeddings near-dup (mocked `/embeddings`)
- [ ] 🟢 embeddings-based near-dup (optional)
- [ ] 🔴 `test_cli_config_show.py` — `config show` prints resolved config with the API key masked
- [ ] 🟢 `config show` command
- [ ] `rich` progress bars (cosmetic, no test); README + examples; raise coverage threshold
- [ ] **Phase gate**: `uv run pytest && uv run ruff check && uv run mypy` clean; coverage ≥ target

## Phase 7 — Web UI (`yatsaury web`, NiceGUI)

Thin front-end over the same `Orchestrator`. Can begin once Phase 1 exists; richer after Phase 3–4.
Storage: `./.yatsaury/`. Layout: single column (process on top, history below).

- [ ] 🔴 `test_session_store.py` — `SessionStore.create/list/get/update`; dir layout created; `list()` sorted newest-first; status transitions queued→running→done/error
- [ ] 🟢 `session/models.py` (`Session`, `SessionStatus`, `SessionInput`) + `session/store.py`
- [ ] 🔴 `test_session_persistence.py` — finished session keeps `samples.jsonl`; re-export to another schema/format works with no LLM call
- [ ] 🟢 wire `SessionStore` re-export path to §6 adapters + exporters
- [ ] 🔴 `test_web_jobs.py` — background job (mocked Orchestrator) updates `status`/`progress`; failure sets `error` without crashing
- [ ] 🟢 `web/jobs.py` — background job runner + progress callbacks
- [ ] 🔴 `test_web_app.py` (`nicegui.testing`) — page renders; upload+text accepted; clicking **Process** creates a session & starts a job; history shows it; download link present
- [ ] 🟢 `web/app.py` — single-column NiceGUI page (form + history + live `ui.timer` refresh)
- [ ] 🔴 `test_cli_web.py` — `web` command builds the app/server with host/port/workspace (no real serve)
- [ ] 🟢 `cli.py` — `web` command (`--host/--port/--workspace/--open`)
- [ ] 🔴 `test_cli_generate_session.py` — `generate --session` records the run into the store (shared history with web)
- [ ] 🟢 `cli.py` — `--session` flag on `generate`
- [ ] **Phase gate**: open `yatsaury web`, upload a file, Process, see it in history, reopen → history persists; unit suite green

---

## Notes / Decisions Log

- 2026-06-18 — Project bootstrapped from planning. Confirmed: name `yatsaury`, tooling `uv`,
  OpenAI-compatible LLM client, JSONL + HF export, dataset language follows source (`--lang` override).
- 2026-06-18 — Introduced the **three-axis** model (DESIGN.md §2): `dataset_type` (content) ×
  `record schema` (Alpaca/ShareGPT/ChatML/qa/completion/rag/raw) × `serialization` (jsonl/hf/csv).
  Added a new `schemas/` adapter family + `--schema` flag; exporters now serialize already-rendered
  dicts (schema-agnostic). Default schema `chatml`.
- 2026-06-18 — Adopted **TDD** (red→green→refactor) as the build process; each component pairs a
  test-first 🔴 item with its 🟢 implementation. LLM/HTTP mocked in unit tests; e2e against Ollama
  is opt-in (`@pytest.mark.e2e`).
- 2026-06-18 — Added a local **web UI** (`yatsaury web`) as a thin layer over the same pipeline.
  Stack **NiceGUI**; sessions persisted under **`./.yatsaury/`** (gitignored) so history survives
  restarts; layout single-column (process on top, history below); long runs as background jobs with
  live progress. New `session/` (store) and `web/` (app+jobs) modules; `--workspace` config added.
- _(add decisions/changes here as implementation proceeds)_
