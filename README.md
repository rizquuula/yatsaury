# Yatsaury

Turn source material — **PDF, web URL, or raw text** — into **training-ready datasets** for
fine-tuning LLMs with new knowledge.

Yatsaury loads your sources, cleans and chunks them, generates samples with an LLM, scores each
sample for **faithfulness to the source** (grounding, no hallucination), de-duplicates, and exports
the result in whatever record schema and file format your trainer expects. It runs as a **CLI** or a
local **web UI** over the same pipeline.

> The first target domain is Islamic content such as Sirah Nabawiyah. Because the content is
> religious, grounding in the source is the top priority — every sample is quote-checked and
> LLM-judged before it ships.

## The three axes

Output is the product of three independent, composable axes. Add one new schema and *every* dataset
type and file format can use it for free.

| Axis | Question | CLI flag | Options |
|---|---|---|---|
| **Dataset type** | *What* is the content? | `--type` | `qa`, `instruction`, `rag`, `summary` |
| **Record schema** | *How* is each record shaped for the trainer? | `--schema` | `chatml`, `sharegpt`, `alpaca`, `qa`, `completion`, `rag`, `raw` |
| **Serialization** | *How* is it stored on disk? | `--format` | `jsonl`, `hf`, `csv` |

## Pipeline

```
[Source loaders] -> [Extract & clean] -> [Chunker] -> [Generators] -> [Grounding/verify] -> [Dedup] -> [Schema adapter] -> [Exporters]
  PDF/URL/text        normalized text     Chunk[]      Sample[]         scored Sample[]       unique[]    record dicts        JSONL / HF / CSV
```

Each stage consumes one typed object and produces the next, so any stage is unit-testable in
isolation and re-runnable from cache.

## Requirements

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An OpenAI-compatible LLM endpoint — works with OpenAI, Ollama, vLLM, LM Studio, etc.

## Install

```bash
make install        # uv sync --all-extras
```

## Configure

Copy the example env file and fill in your LLM credentials:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `YATSAURY_API_KEY` | — | LLM API key (`OPENAI_API_KEY` accepted as fallback) |
| `YATSAURY_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (`OPENAI_BASE_URL` fallback) |
| `YATSAURY_MODEL` | `gpt-4o-mini` | Generation model |
| `YATSAURY_JUDGE_MODEL` | *(uses model)* | Separate model for the quality judge |
| `YATSAURY_JUDGE_BATCH_SIZE` | `1` | Samples per judge call |
| `YATSAURY_CHUNK_SIZE` | `512` | Chunk size in tokens |
| `YATSAURY_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `YATSAURY_PER_CHUNK` | `3` | Samples generated per chunk |
| `YATSAURY_MIN_SCORE` | `70` | Minimum quality score (0–100) to keep a sample |
| `YATSAURY_LANG` | `auto` | Output language (`auto` follows the source) |
| `YATSAURY_WORKSPACE` | `./.yatsaury` | Session/history storage |
| `YATSAURY_WEB_HOST` | `127.0.0.1` | Web UI host |
| `YATSAURY_WEB_PORT` | `8080` | Web UI port |

Non-secret defaults can alternatively live in `config.toml` (see `config.example.toml`); keep API
keys in `.env`. Verify the resolved configuration any time with:

```bash
uv run yatsaury config
```

## Usage

### Web UI

```bash
make web            # or: uv run yatsaury web
```

Opens the local web app (default `http://127.0.0.1:8080`). Every run is persisted to the workspace,
so reopening the web shows your history.

### CLI

**Generate** a dataset from one or more sources:

```bash
uv run yatsaury generate \
  -i examples/sirah_sample.txt \
  --type qa --schema chatml --format jsonl \
  -o ./output
```

All three axes are repeatable, so a single run can fan out across multiple types, schemas, and
formats. Other useful options: `--per-chunk/-n`, `--chunk-size`, `--chunk-overlap`,
`--limit-chunks`, `--paraphrases`, `--difficulty easy,medium,hard`, `--model`, `--base-url`,
`--dry-run` (print the plan without making any LLM calls), and `--session` (record the run in the
session store).

**Other commands:**

| Command | Purpose |
|---|---|
| `yatsaury generate` | Generate training samples from source documents |
| `yatsaury inspect -i <src>` | Load and chunk sources; print document/chunk statistics |
| `yatsaury verify -i <data.jsonl>` | Re-score an existing JSONL dataset for quality |
| `yatsaury export -i <data.jsonl>` | Re-render a dataset into a new schema/format (no LLM cost) |
| `yatsaury schemas` | List record schemas and their compatible dataset types |
| `yatsaury web` | Launch the web UI |
| `yatsaury config` | Show the resolved configuration (API key masked) |

`verify` runs both a deterministic quote check and an LLM judge (use `--no-judge` for the quote check
only); `export` lets you reshape a reviewed CSV or JSONL into any schema/format without re-calling
the model.

## Project layout

```
src/yatsaury/
├── cli.py            # Typer CLI entry point (yatsaury ...)
├── pipeline.py       # Orchestrator wiring all stages together
├── models.py         # Core Pydantic models: Document, Chunk, Sample, Citation
├── config.py         # Settings (env + TOML via pydantic-settings)
├── sources/          # Source loaders: pdf, url, text
├── processing/       # clean + chunk
├── generators/       # Dataset-type generators: qa, instruction, rag, summary, ...
├── quality/          # verify (grounding/judge), dedup, coverage
├── schemas/          # Record-schema adapters: chatml, sharegpt, alpaca, qa, ...
├── exporters/        # Serializers: jsonl, hf, review_csv
├── llm/              # OpenAI-compatible client + prompts
├── session/          # Session store for persisted runs/history
└── web/              # NiceGUI web app
```

See [`DESIGN.md`](DESIGN.md) for the full architecture and [`IMPLEMENTATION_PROGRESS.md`](IMPLEMENTATION_PROGRESS.md)
for status.

## Development

```bash
make test           # unit tests (excludes e2e)
make test-e2e       # end-to-end tests (requires a local Ollama)
make lint           # ruff
make fmt            # ruff format
make typecheck      # mypy (strict)
make check          # lint + typecheck + test — the full CI gate
make clean          # remove build artefacts and caches
```

Run `make help` to see all targets.
