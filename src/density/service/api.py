"""HTTP surface over one open Store: thin JSON wrappers, no auth.

create_app(store) builds a FastAPI app whose three endpoints are direct
pass-throughs to SDK calls that already exist:

    POST /audit {"path": str, "tiers": optional list}
        -> density.audit.runner.run_audit(...).to_dict(), report written
           next to the given corpus path.
    GET /search?q=...&k=10&tier=optional
        -> Store.search_cli(q, k, tier): q is parsed exactly as the CLI
           parses it (JSON array vector first, else text), because the
           parsing lives in the store method, not here.
    GET /replay/{trace_id}
        -> Store.replay(trace_id): the parsed original events.

No parsing, ranking, or compression logic lives in this module. Every
error returns structured JSON ({"error": {"type": ..., "message": ...}})
and never a stack trace: expected failures are caught per endpoint and
mapped to 400/404, and an app-level handler turns anything unexpected
into an opaque 500.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from density.errors import DensityError, StoreError
from density.tiers.store import Store

__all__ = ["create_app"]

# Store.replay raises StoreError for every failure mode. These two stable
# message prefixes (part of the store's documented contract text) are the
# "resource does not exist" cases and map to 404; every other StoreError
# on replay (corrupt bundle, split corpora, closed store) is a server-side
# problem and maps to 500. StoreError carries no machine-readable code
# field in v1, so prefix matching is the honest option; if the store ever
# grows error codes, this list is the one place to update.
_NOT_FOUND_PREFIXES = ("unknown trace_id", "store holds no traces")


class AuditRequest(BaseModel):
    """POST /audit body. path is a raw corpus directory or file on the
    server's filesystem; tiers defaults to the audit runner's own default
    ("warm", "cold") when omitted."""

    path: str
    tiers: list[str] | None = None


def _error(status: int, exc_type: str, message: str) -> JSONResponse:
    """One structured error shape for every failure path."""
    return JSONResponse(
        status_code=status,
        content={"error": {"type": exc_type, "message": message}},
    )


def create_app(store: Store) -> FastAPI:
    """Build the FastAPI app serving one open Store.

    The caller owns the store's lifecycle: the app never opens or closes
    it. Binding to a socket is the caller's job too (the CLI serve
    command runs uvicorn); nothing here touches the network.
    """
    app = FastAPI(
        title="DENSITY",
        description="Recall-guaranteed storage tiers for agent traces and embeddings.",
    )

    @app.exception_handler(RequestValidationError)
    async def _on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default is 422 with framework-shaped detail; the
        # service contract says bad inputs are 400 with our error shape.
        detail = "; ".join(
            ".".join(str(part) for part in err.get("loc", ())) + ": " + str(err.get("msg", ""))
            for err in exc.errors()
        )
        return _error(400, "ValidationError", f"invalid request: {detail}")

    @app.exception_handler(DensityError)
    async def _on_density_error(request: Request, exc: DensityError) -> JSONResponse:
        # Safety net: endpoints catch and map expected DensityErrors
        # themselves; anything that escapes is still a clean 400.
        return _error(400, type(exc).__name__, str(exc))

    @app.exception_handler(Exception)
    async def _on_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Unexpected bugs must not leak stack traces or internal paths to
        # HTTP clients; the message is deliberately opaque.
        return _error(500, "InternalError", "internal server error")

    # response_model=None on every endpoint: handlers return either a
    # plain dict or an error JSONResponse, a union FastAPI cannot (and
    # need not) turn into a pydantic response model.
    @app.post("/audit", response_model=None)
    def audit(req: AuditRequest) -> JSONResponse | dict:
        """Run a full audit of a raw corpus path; report lands beside it.

        The report is written next to the given path (PATH.report.html
        plus PATH.report.md), never inside it, so re-auditing the same
        corpus stays byte-identical.
        """
        # Imported lazily, mirroring cli.py: the audit stack (jinja2
        # templates, pricing) should not load just to serve /search.
        from density.audit.runner import run_audit

        corpus = Path(req.path)
        out_html = corpus.with_name(corpus.name + ".report.html")
        try:
            kwargs: dict = {"out": str(out_html)}
            if req.tiers is not None:
                kwargs["tiers"] = tuple(req.tiers)
            result = run_audit(corpus, **kwargs)
        except DensityError as exc:
            # AuditError (missing path, unknown tier, empty corpus) and
            # friends are caller mistakes: 400 with the runner's message.
            return _error(400, type(exc).__name__, str(exc))
        return {
            "report_html": result.out_html,
            "report_md": result.out_md,
            "audit": result.to_dict(),
        }

    @app.get("/search", response_model=None)
    def search(
        q: str = Query(..., description="JSON array vector, or query text."),
        k: int = Query(10, description="Number of results."),
        tier: str | None = Query(None, description="Force a tier: hot, warm, cold."),
    ) -> JSONResponse | dict:
        """Top-k search. q is parsed by Store.search_cli, same as the CLI."""
        try:
            ids, scores = store.search_cli(q, k=k, tier=tier)
        except StoreError as exc:
            return _error(400, "StoreError", str(exc))
        # tolist() converts numpy int64/str ids and float32 scores to
        # plain JSON scalars; ids keep the dtype the caller ingested.
        return {"ids": ids.tolist(), "scores": [float(s) for s in scores]}

    @app.get("/replay/{trace_id}", response_model=None)
    def replay(trace_id: str) -> JSONResponse | dict:
        """The parsed original events of one trace, in original order."""
        try:
            events = store.replay(trace_id)
        except StoreError as exc:
            message = str(exc)
            if message.startswith(_NOT_FOUND_PREFIXES):
                return _error(404, "StoreError", message)
            # Corrupt bundles or split-corpora ambiguity: the trace may
            # exist but the store cannot produce it. That is our fault,
            # not the caller's.
            return _error(500, "StoreError", message)
        return {"trace_id": trace_id, "events": events}

    return app
