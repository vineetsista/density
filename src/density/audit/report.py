"""Report rendering: one AuditResult, two faces, identical numbers.

render(result) returns (markdown, html). Both documents are built from
one shared view model of preformatted strings, so a number can never
differ between the two formats: they are the same figures placed twice.
The HTML comes from templates/report.html.j2 through a package-relative
jinja2 loader and is fully self-contained: inline CSS only, no
JavaScript, no external assets, printable via an @media print block.
"""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, select_autoescape

from density.audit.runner import AuditResult, human_bytes, usd

__all__ = ["render"]

_ENV = Environment(
    loader=PackageLoader("density.audit", "templates"),
    autoescape=select_autoescape(("html", "j2")),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}x"


def _recall(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.4f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f} percent"


def _bytes_pair(n: int) -> str:
    """Humanized size with the exact byte count beside it: trust needs both."""
    return f"{human_bytes(n)} ({n:,} bytes)"


def _view(result: AuditResult) -> dict:
    """The shared view model: every figure formatted exactly once."""
    r = result
    headline_name = r.headline_tier
    headline = None
    if headline_name is not None:
        t = r.tier_results[headline_name]
        sv = r.savings.per_tier[headline_name]
        recall10 = None if t.recall is None else t.recall.get("recall@10")
        if t.recall is not None and t.recall10_floor is not None:
            recall_line = f"{_recall(recall10)} ({'PASS' if t.recall_pass else 'MISS'})"
        elif headline_name == "hot" and r.embeddings_present:
            recall_line = "exact by definition"
        else:
            recall_line = "not measured"
        headline = {
            "tier": headline_name,
            "size": _bytes_pair(t.total_bytes),
            "ratio": _ratio(t.ratio),
            "recall": recall_line,
            "monthly": usd(sv.monthly),
            "yearly": usd(sv.yearly),
        }

    tiers = []
    for name in r.tiers:
        t = r.tier_results[name]
        sv = r.savings.per_tier[name]
        if t.recall is not None and t.recall10_floor is not None:
            verdict = "PASS" if t.recall_pass else "MISS"
        elif name == "hot" and r.embeddings_present:
            verdict = "EXACT"
        else:
            verdict = "UNMEASURED"
        tiers.append(
            {
                "name": name,
                "codec": t.codec or ("fp32" if r.embeddings_present else "none"),
                "floor": (
                    "none (exact)" if t.recall10_floor is None else f"{t.recall10_floor:.2f}"
                ),
                "recall10": _recall(None if t.recall is None else t.recall.get("recall@10")),
                "recall1": _recall(None if t.recall is None else t.recall.get("recall@1")),
                "recall100": _recall(None if t.recall is None else t.recall.get("recall@100")),
                "verdict": verdict,
                "target": _pct(t.footprint_target),
                "measured_footprint": _pct(t.footprint_fraction),
                "traces_bytes": (
                    _bytes_pair(t.traces_bytes)
                    if r.traces_present
                    else "not present in this corpus"
                ),
                "vectors_bytes": (
                    _bytes_pair(t.vectors_bytes + t.vectors_aux_bytes)
                    if r.embeddings_present
                    else "not present in this corpus"
                ),
                "total_bytes": _bytes_pair(t.total_bytes),
                "ratio": _ratio(t.ratio),
                "rerank_depth": str(t.rerank_depth),
                "monthly": usd(sv.monthly),
                "yearly": usd(sv.yearly),
            }
        )

    dedup = None
    if r.dedup is not None:
        d = r.dedup
        trace_raw = max(r.traces_raw_bytes, 1)
        dedup = {
            "clusters": f"{d.cluster_count:,}",
            "exact_groups": f"{d.exact_group_count:,}",
            "bytes_saved": _bytes_pair(d.bytes_saved),
            "saved_fraction": _pct(d.bytes_saved / trace_raw),
            # residual_lines is None when the requested tiers built no
            # bundle at all. Printing 0.0 percent there would state a
            # measurement about something that was never measured, which
            # is the one thing this report must not do.
            "residual_measured": d.residual_lines is not None,
            "residual_tier": d.residual_tier or "",
            "residual_rate": (
                "not measured (no trace bundle was built for the requested tiers)"
                if d.residual_lines is None
                else (_pct(d.residual_lines / d.total_lines) if d.total_lines else "n/a")
            ),
            "residual_lines": (
                "" if d.residual_lines is None else f"{d.residual_lines:,}"
            ),
            "total_lines": f"{d.total_lines:,}",
            "top_clusters": [
                {"size": f"{c.size:,}", "sample": c.sample} for c in d.top_clusters
            ],
        }

    m = r.methodology
    verifier = m.get("verifier")
    rt = m.get("round_trip", {})
    sampled_lines = int(rt.get("sampled_lines", 0))
    total_lines = int(rt.get("total_lines", 0))
    sampling_notes = []
    if sampled_lines:
        sampling_notes.append(
            f"Trace round-trip was byte-verified on a seeded sample of "
            f"{sampled_lines:,} of {total_lines:,} lines "
            f"({_pct(sampled_lines / total_lines if total_lines else None)}), "
            "not exhaustively, in every tier bundle."
        )
    if verifier:
        sampling_notes.append(
            f"Recall is a sample estimate: {verifier['n_queries']:,} held-out "
            f"query vectors against {verifier['n_database']:,} database vectors "
            f"(seeded split of {verifier['n_vectors']:,}), ground truth exact "
            f"to k={verifier['k']}."
        )

    pricing = m.get("pricing", {})
    depths_str = ""
    if verifier:
        depths_str = ", ".join(
            f"{name} {depth}" for name, depth in verifier["rerank_depths"].items()
        )
    methodology = {
        "seed": str(m.get("seed")),
        "sample_events": f"{float(m.get('sample_events', 0.0)):.4f}",
        "verifier": verifier,
        "verifier_depths": depths_str,
        "round_trip": rt,
        "pricing_lines": [
            f"s3_gb_month = {pricing.get('s3_gb_month')}",
            f"vectordb_gb_month = {pricing.get('vectordb_gb_month')}",
            f"source = {pricing.get('source')}",
        ],
        "versions": [f"{k} {v}" for k, v in sorted(m.get("versions", {}).items())],
        "accel": "active" if m.get("accel_active") else "inactive (numpy fallback)",
        "cold_rerank_reference": m.get("cold_rerank_reference", ""),
        "units": m.get("units", ""),
        # The render stage is excluded: the report is written during it,
        # so it cannot contain its own duration, and including a partial
        # value would make render(result) unreproducible.
        "timings": [
            {"stage": k, "seconds": f"{v:.3f}"}
            for k, v in r.stage_seconds.items()
            if k != "render"
        ],
    }

    original_parts = []
    if r.traces_present:
        original_parts.append(
            f"traces {_bytes_pair(r.traces_raw_bytes)}, {r.trace_events:,} events "
            f"in {r.trace_files} files, {r.malformed:,} malformed lines quarantined"
        )
    else:
        original_parts.append("traces: not present in this corpus")
    if r.embeddings_present:
        original_parts.append(
            f"vectors {_bytes_pair(r.vectors_raw_bytes)}, "
            f"{r.vector_count:,} x {r.vector_dim} float32"
        )
    else:
        original_parts.append("vectors: not present in this corpus")

    return {
        "path": r.path,
        "seed": str(r.seed),
        "original_size": _bytes_pair(r.original_bytes),
        "original_parts": original_parts,
        "headline": headline,
        "tiers": tiers,
        "traces_present": r.traces_present,
        "embeddings_present": r.embeddings_present,
        "dedup": dedup,
        "before_monthly": usd(r.savings.before_monthly),
        "methodology": methodology,
        "honesty": list(r.honesty),
        "sampling_notes": sampling_notes,
        "all_passed": not r.honesty,
    }


def _markdown(view: dict) -> str:
    """The markdown face: same sections, same order, same numbers."""
    lines: list[str] = []
    lines.append("# DENSITY audit report")
    lines.append("")
    lines.append(f"Corpus: `{view['path']}` (seed {view['seed']})")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Original size: {view['original_size']}")
    for part in view["original_parts"]:
        lines.append(f"- {part}")
    h = view["headline"]
    if h is not None:
        lines.append(f"- Best tier: {h['tier']} at {h['size']}, {h['ratio']} smaller")
        lines.append(f"- Measured recall@10 at the {h['tier']} tier: {h['recall']}")
        lines.append(
            f"- Projected savings at the {h['tier']} tier: {h['monthly']} per month, "
            f"{h['yearly']} per year"
        )
    lines.append("")
    lines.append("## Tier guarantees")
    lines.append("")
    lines.append(
        "| Tier | Codec | Floor recall@10 | Measured recall@10 | Verdict | "
        "Footprint target | Measured footprint | Total bytes | Ratio |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for t in view["tiers"]:
        lines.append(
            f"| {t['name']} | {t['codec']} | {t['floor']} | {t['recall10']} | "
            f"{t['verdict']} | {t['target']} | {t['measured_footprint']} | "
            f"{t['total_bytes']} | {t['ratio']} |"
        )
    lines.append("")
    lines.append("## Compression breakdown")
    lines.append("")
    lines.append("| Tier | Traces | Vectors (incl. aux) | Total | Savings per month | Savings per year |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for t in view["tiers"]:
        lines.append(
            f"| {t['name']} | {t['traces_bytes']} | {t['vectors_bytes']} | "
            f"{t['total_bytes']} | {t['monthly']} | {t['yearly']} |"
        )
    lines.append("")
    d = view["dedup"]
    if d is not None:
        residual = (
            f"{d['residual_rate']} ({d['residual_lines']} of {d['total_lines']} "
            f"lines, {d['residual_tier']} bundle)"
            if d["residual_measured"]
            else d["residual_rate"]
        )
        lines.append(
            f"Dedup: {d['clusters']} near-duplicate clusters, {d['exact_groups']} "
            f"exact groups sharing storage, {d['bytes_saved']} saved "
            f"({d['saved_fraction']} of raw trace bytes). Residual rate: "
            f"{residual}."
        )
        if d["top_clusters"]:
            lines.append("")
            lines.append("Top duplicate clusters:")
            lines.append("")
            lines.append("| Members | Sample (first 120 chars) |")
            lines.append("|---:|---|")
            for c in d["top_clusters"]:
                sample = c["sample"].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {c['size']} | {sample} |")
    else:
        lines.append("Traces: not present in this corpus, nothing to deduplicate.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    m = view["methodology"]
    lines.append(f"- Seed: {m['seed']}; event sample fraction: {m['sample_events']}")
    if m["verifier"]:
        v = m["verifier"]
        lines.append(
            f"- Verifier: {v['n_queries']:,} held-out queries, database "
            f"{v['n_database']:,} of {v['n_vectors']:,} vectors, ground truth "
            f"k={v['k']}, rerank depths {m['verifier_depths']}"
        )
    lines.append(f"- {m['cold_rerank_reference']}")
    lines.append(f"- Pricing constants: {'; '.join(m['pricing_lines'])}")
    lines.append(f"- Cost before compression: {view['before_monthly']} per month")
    lines.append(f"- Versions: {', '.join(m['versions'])}; accel kernels {m['accel']}")
    lines.append(f"- {m['units']}")
    timing = ", ".join(f"{t['stage']} {t['seconds']}s" for t in m["timings"])
    lines.append(f"- Stage timings (wall clock, nondeterministic): {timing}")
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    if view["all_passed"]:
        lines.append(
            "Every guarantee measured in this audit passed. Nothing was skipped "
            "or degraded."
        )
    else:
        for flag in view["honesty"]:
            lines.append(f"- {flag}")
    for note in view["sampling_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def render(result: AuditResult) -> tuple[str, str]:
    """Render one AuditResult. Returns (markdown, html), same numbers in both."""
    view = _view(result)
    markdown = _markdown(view)
    html = _ENV.get_template("report.html.j2").render(**view)
    return markdown, html
