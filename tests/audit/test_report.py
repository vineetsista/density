"""Report rendering tests: both faces, same numbers, self-contained HTML.

Reuses the hand-built corpus from test_runner (never the synth
generator) and runs one audit for the whole module: these tests are
read-only assertions over the rendered markdown and HTML.
"""

from __future__ import annotations

import pytest

from density.audit.report import render
from density.audit.runner import human_bytes, run_audit, usd
from tests.audit.test_runner import SEED, build_corpus


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    corpus_dir, meta = build_corpus(tmp_path_factory.mktemp("report-corpus"))
    out = tmp_path_factory.mktemp("report-out") / "report.html"
    result = run_audit(corpus_dir, out=str(out), seed=SEED)
    html = out.read_text(encoding="utf-8")
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    return result, md, html


def test_html_is_self_contained(rendered):
    _, _, html = rendered
    # A trust document must render from file:// on an airgapped machine:
    # no network fetches, no scripts, no linked assets of any kind.
    for needle in ("http://", "https://", "src=", "<script", "href"):
        assert needle not in html, f"forbidden fragment in report: {needle!r}"
    assert "<style>" in html  # styling is inline, not linked


def test_no_dash_characters_anywhere(rendered):
    result, md, html = rendered
    for doc in (md, html, result.summary_text()):
        assert "\u2014" not in doc  # em dash
        assert "\u2013" not in doc  # en dash


def test_html_sections_in_contract_order(rendered):
    _, _, html = rendered
    order = ['id="headline"', 'id="tiers"', 'id="compression"',
             'id="methodology"', 'id="honesty"']
    positions = [html.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_markdown_mirrors_section_order(rendered):
    _, md, _ = rendered
    order = ["# DENSITY audit report", "## Headline", "## Tier guarantees",
             "## Compression breakdown", "## Methodology", "## Honesty"]
    positions = [md.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_headline_numbers_shared_between_faces(rendered):
    result, md, html = rendered
    best = result.tier_results[result.headline_tier]
    sv = result.savings.per_tier[result.headline_tier]
    shared = [
        f"{best.ratio:.1f}x",
        usd(sv.monthly),
        usd(sv.yearly),
        f"({result.original_bytes:,} bytes)",
        f"({best.total_bytes:,} bytes)",
    ]
    for fragment in shared:
        assert fragment in md, f"missing from markdown: {fragment!r}"
        assert fragment in html, f"missing from html: {fragment!r}"


def test_pass_labels_with_measured_recall(rendered):
    result, md, html = rendered
    # The fixture's warm sq8 tier comfortably clears its 0.99 floor.
    warm = result.tier_results["warm"]
    assert warm.recall_pass is True
    recall_str = f"{warm.recall['recall@10']:.4f}"
    for doc in (md, html):
        assert "PASS" in doc
        assert recall_str in doc
        assert f"{warm.recall10_floor:.2f}" in doc


def test_exact_bytes_beside_humanized_sizes(rendered):
    result, md, html = rendered
    pair = f"{human_bytes(result.original_bytes)} ({result.original_bytes:,} bytes)"
    assert pair in md and pair in html


def test_dedup_and_residual_reported(rendered):
    result, md, html = rendered
    top = result.dedup.top_clusters[0]
    # Samples are truncated to 120 chars and appear in both faces.
    assert len(top.sample) <= 120
    probe = top.sample[:40]
    assert probe in md and probe in html
    assert "Residual rate" in html and "Residual rate" in md


def test_honesty_states_sampling_even_on_full_pass(rendered):
    result, md, html = rendered
    rt = result.methodology["round_trip"]
    note = f"{rt['sampled_lines']:,} of {rt['total_lines']:,} lines"
    for doc in (md, html):
        assert "seeded sample" in doc
        assert note in doc
        assert "sample estimate" in doc


def test_print_stylesheet_present(rendered):
    _, _, html = rendered
    assert "@media print" in html
    assert "tabular-nums" in html


def test_pricing_constants_named_in_methodology(rendered):
    result, md, html = rendered
    p = result.savings.pricing
    for doc in (md, html):
        assert f"s3_gb_month = {p.s3_gb_month}" in doc
        assert f"vectordb_gb_month = {p.vectordb_gb_month}" in doc
        assert "seed" in doc.lower()


def test_render_returns_both_faces(rendered):
    result, md, html = rendered
    md2, html2 = render(result)
    # render() is a pure function of the result: re-rendering reproduces
    # the files the audit wrote.
    assert md2 == md
    assert html2 == html
