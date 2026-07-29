"""DENSITY command line interface.

Commands: synth, audit, compress, search, replay, bench, serve.
Each command is a thin wrapper over the SDK. Logic lives in the library.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="density",
    help="DENSITY: recall-guaranteed storage tiers for agent traces and embeddings.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def synth(
    gb: float = typer.Option(1.0, "--gb", help="Target corpus size in gigabytes."),
    out: Path = typer.Option(Path("./raw"), "--out", help="Output directory."),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed."),
    dim: int = typer.Option(768, "--dim", help="Embedding dimensionality."),
) -> None:
    """Generate a realistic synthetic agent corpus (traces + embeddings)."""
    from density.bench.synth import generate

    stats = generate(gb, out, seed=seed, dim=dim)
    typer.echo(json.dumps(stats.to_dict(), indent=2))


@app.command()
def audit(
    path: Path = typer.Argument(..., help="Directory of raw traces and embeddings."),
    out: Path = typer.Option(Path("report.html"), "--out", help="Report output path."),
    tiers: str = typer.Option("warm,cold", "--tiers", help="Comma-separated tiers."),
    pricing: Path | None = typer.Option(None, "--pricing", help="pricing.toml path."),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed."),
) -> None:
    """Compress, verify recall, and estimate savings. The report is the product."""
    from density.audit.runner import run_audit

    result = run_audit(
        path,
        out=str(out),
        tiers=tuple(t.strip() for t in tiers.split(",") if t.strip()),
        pricing=str(pricing) if pricing else None,
        seed=seed,
    )
    typer.echo(result.summary_text())


@app.command()
def compress(
    path: Path = typer.Argument(..., help="Directory of raw traces and embeddings."),
    out: Path | None = typer.Option(None, "--out", help="Store directory (default PATH.density)."),
    tiers: str = typer.Option("warm,cold", "--tiers", help="Comma-separated tiers."),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed."),
) -> None:
    """Build a tiered store from raw data, without the audit report."""
    from density.audit.runner import compress_to_store

    store_dir = compress_to_store(
        path,
        out=str(out) if out else None,
        tiers=tuple(t.strip() for t in tiers.split(",") if t.strip()),
        seed=seed,
    )
    typer.echo(f"store written: {store_dir}")


@app.command()
def search(
    store: Path = typer.Argument(..., help="Path to a .density store directory."),
    query: str = typer.Argument(..., help="Query text (demo embedder) or JSON vector."),
    k: int = typer.Option(10, "-k", help="Number of results."),
    tier: str | None = typer.Option(None, "--tier", help="Force a specific tier."),
) -> None:
    """Search a store. Text queries use the demo hashing embedder."""
    import density as sdk

    with sdk.open(store) as st:
        ids, scores = st.search_cli(query, k=k, tier=tier)
        for i, s in zip(ids, scores):
            typer.echo(f"{i}\t{s:.4f}")


@app.command()
def replay(
    store: Path = typer.Argument(..., help="Path to a .density store directory."),
    trace_id: str = typer.Argument(..., help="Trace id to replay."),
    raw: bool = typer.Option(False, "--raw", help="Print original raw lines."),
) -> None:
    """Replay the exact original events of one trace."""
    import density as sdk

    with sdk.open(store) as st:
        if raw:
            for line in st.replay_raw(trace_id):
                typer.echo(line.decode("utf-8", errors="replace"))
        else:
            for event in st.replay(trace_id):
                typer.echo(json.dumps(event, ensure_ascii=False))


@app.command()
def bench(
    quick: bool = typer.Option(False, "--quick", help="Small, fast configuration."),
    out: Path = typer.Option(
        Path("benchmarks/results"), "--out", help="Results directory."
    ),
    seed: int = typer.Option(1337, "--seed", help="Deterministic seed."),
) -> None:
    """Run the reproducible public benchmark and write results JSON."""
    from density.bench.harness import run_bench

    path = run_bench(out_dir=out, quick=quick, seed=seed)
    typer.echo(f"results written: {path}")


@app.command()
def serve(
    store: Path = typer.Argument(..., help="Path to a .density store directory."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8377, "--port"),
) -> None:
    """Serve a store over HTTP: POST /audit, GET /search, GET /replay."""
    import uvicorn

    import density as sdk
    from density.service.api import create_app

    st = sdk.open(store)
    uvicorn.run(create_app(st), host=host, port=port, workers=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
