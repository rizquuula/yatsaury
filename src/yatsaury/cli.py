"""Yatsaury CLI — Turn source material into LLM training datasets."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="yatsaury",
    help="Turn source material into LLM training datasets.",
    no_args_is_help=True,
)


@app.command()
def generate(
    input: list[str] = typer.Option(
        ..., "-i", "--input", help="Input file paths or URLs (repeatable)"
    ),
    type: list[str] = typer.Option(
        ["qa"], "-t", "--type", help="Dataset type(s): qa, instruction, rag, summary"
    ),
    schema: list[str] = typer.Option(
        ["chatml"],
        "-s",
        "--schema",
        help="Record schema(s): chatml, qa, sharegpt, alpaca, completion, rag, raw",
    ),
    format: list[str] = typer.Option(
        ["jsonl"], "-f", "--format", help="Output format(s): jsonl, hf, csv"
    ),
    output: Path = typer.Option(Path("./output"), "-o", "--output"),
    chunk_size: int = typer.Option(512, "--chunk-size"),
    chunk_overlap: int = typer.Option(64, "--chunk-overlap"),
    per_chunk: int = typer.Option(3, "-n", "--per-chunk"),
    model: str = typer.Option("", "--model"),
    base_url: str = typer.Option("", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
    limit_chunks: int = typer.Option(0, "--limit-chunks", help="0 = no limit"),
    paraphrases: int = typer.Option(1, "--paraphrases", help="Paraphrase variants per Q&A"),
    difficulty: str = typer.Option("", "--difficulty", help="Comma-separated: easy,medium,hard"),
) -> None:
    """Generate training samples from source documents."""
    import yatsaury.exporters.hf  # noqa: F401
    import yatsaury.generators.instruction  # noqa: F401
    import yatsaury.generators.qa  # noqa: F401
    import yatsaury.generators.rag  # noqa: F401
    import yatsaury.generators.summary  # noqa: F401
    import yatsaury.schemas.alpaca  # noqa: F401
    import yatsaury.schemas.chatml  # noqa: F401
    import yatsaury.schemas.completion  # noqa: F401
    import yatsaury.schemas.qa  # noqa: F401
    import yatsaury.schemas.rag  # noqa: F401
    import yatsaury.schemas.raw  # noqa: F401
    import yatsaury.schemas.sharegpt  # noqa: F401
    from yatsaury.config import Settings
    from yatsaury.pipeline import Orchestrator, OrchestratorConfig

    settings = Settings()
    config = OrchestratorConfig(
        dataset_types=type,
        schema_names=schema,
        output_formats=format,
        output_dir=output,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        per_chunk=per_chunk,
        llm_base_url=base_url or settings.base_url,
        llm_api_key=api_key or settings.api_key.get_secret_value(),
        llm_model=model or settings.model,
        paraphrases=paraphrases,
        difficulty=[d.strip() for d in difficulty.split(",") if d.strip()],
    )
    orch = Orchestrator(config)
    records = orch.run(input)
    count = len(records)
    noun = "record" if count == 1 else "records"
    typer.echo(f"Generated {count} {noun}.")


@app.command()
def inspect(
    input: list[str] = typer.Option(..., "-i", "--input", help="Input files or URLs"),
    chunk_size: int = typer.Option(512, "--chunk-size"),
    chunk_overlap: int = typer.Option(64, "--chunk-overlap"),
) -> None:
    """Load and chunk sources; print document and chunk statistics."""
    from yatsaury.processing.chunk import chunk_document
    from yatsaury.processing.clean import clean_text
    from yatsaury.sources.base import resolve_loader

    for uri in input:
        loader = resolve_loader(uri)
        doc = loader.load(uri)
        doc = doc.model_copy(update={"raw_text": clean_text(doc.raw_text)})
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=chunk_overlap)
        total_tokens = sum(c.token_count for c in chunks)
        typer.echo(f"Source: {uri}")
        typer.echo(f"  Characters : {len(doc.raw_text)}")
        typer.echo(f"  Chunks     : {len(chunks)}")
        typer.echo(f"  Total tokens: {total_tokens}")
        if chunks:
            typer.echo(f"  Avg tokens/chunk: {total_tokens // len(chunks)}")


@app.command()
def verify(
    input: Path = typer.Option(..., "-i", "--input", help="JSONL file of samples to re-score"),
    output: Path = typer.Option(
        None, "-o", "--output", help="Output JSONL (default: overwrite input)"
    ),
    min_score: float = typer.Option(0.7, "--min-score"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge; use quote check only"),
    model: str = typer.Option("", "--model"),
    base_url: str = typer.Option("", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
    judge_model: str = typer.Option("", "--judge-model"),
) -> None:
    """Re-score an existing JSONL dataset for grounding quality."""
    from yatsaury.config import Settings
    from yatsaury.llm.client import LLMClient
    from yatsaury.models import Sample
    from yatsaury.quality.verify import verify_samples

    settings = Settings()
    llm = LLMClient(
        base_url=base_url or settings.base_url,
        api_key=api_key or settings.api_key.get_secret_value(),
        model=model or settings.model,
    )
    samples = [
        Sample.model_validate_json(line)
        for line in input.read_text().splitlines()
        if line.strip()
    ]
    passing = verify_samples(samples, llm, min_score=min_score,
                             judge_model=judge_model or settings.judge_model,
                             use_judge=not no_judge)
    out = output or input
    out.write_text("\n".join(s.model_dump_json() for s in passing) + "\n")
    typer.echo(f"Kept {len(passing)}/{len(samples)} samples.")


@app.command(name="export")
def export_cmd(
    input: Path = typer.Option(..., "-i", "--input", help="JSONL or reviewed CSV to export"),
    schema: list[str] = typer.Option(["chatml"], "-s", "--schema"),
    format: list[str] = typer.Option(["jsonl"], "-f", "--format"),
    output: Path = typer.Option(Path("./output"), "-o", "--output"),
) -> None:
    """Re-render an existing dataset into a new schema/format (no LLM cost)."""
    from yatsaury.exporters.base import get_exporter
    from yatsaury.exporters.review_csv import CsvReviewExporter
    from yatsaury.models import Sample
    from yatsaury.schemas.base import get_schema

    if input.suffix.lower() == ".csv":
        rows = CsvReviewExporter().load_approved(input)
        samples = []
        for row in rows:
            try:
                samples.append(Sample.model_validate(row))
            except Exception:
                pass
    else:
        lines = [line for line in input.read_text().splitlines() if line.strip()]
        samples = [Sample.model_validate_json(line) for line in lines]

    records: list[dict] = []
    for schema_name in schema:
        adapter = get_schema(schema_name)
        for sample in samples:
            if adapter.supports(sample.dataset_type.value):
                records.append(adapter.render(sample))

    for fmt in format:
        exporter_cls = get_exporter(fmt)
        exporter = exporter_cls()
        for schema_name in schema:
            out_path = output / f"{schema_name}.{fmt}"
            exporter.export(records, out_path)
    typer.echo(f"Exported {len(records)} records.")


@app.command()
def schemas() -> None:
    """List available record schemas and their compatible dataset types."""
    import yatsaury.schemas.alpaca  # noqa: F401
    import yatsaury.schemas.chatml  # noqa: F401
    import yatsaury.schemas.completion  # noqa: F401
    import yatsaury.schemas.qa  # noqa: F401
    import yatsaury.schemas.rag  # noqa: F401
    import yatsaury.schemas.raw  # noqa: F401
    import yatsaury.schemas.sharegpt  # noqa: F401
    from yatsaury.schemas.base import _REGISTRY

    for name, adapter in sorted(_REGISTRY.items()):
        supported = [t for t in ["qa", "instruction", "summary", "rag"] if adapter.supports(t)]
        typer.echo(f"{name:12s}  supports: {', '.join(supported)}")


@app.command()
def web(
    ctx: typer.Context,
) -> None:
    """Launch the Yatsaury web UI."""
    typer.echo("web: not yet implemented")


@app.command(name="config")
def config_cmd(
    ctx: typer.Context,
) -> None:
    """Show or validate current configuration."""
    typer.echo("config: not yet implemented")
