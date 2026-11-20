"""Report: collect from records, figures, HTML, and the PDF when plutoprint exists."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from geomulticorr.cli.record import StageRecord, write_record
from geomulticorr.report import collect_run, render_report
from geomulticorr.report import collect as collect_mod
from geomulticorr.report import figures

PAIRS = [
    {
        "pa_key": "a",
        "dt_days": 30,
        "ew_nmad_raw": 1.0,
        "ew_nmad_corr": 0.5,
        "ns_nmad_raw": 0.8,
        "ns_nmad_corr": 0.4,
    },
    {
        "pa_key": "b",
        "dt_days": 400,
        "ew_nmad_raw": 2.0,
        "ew_nmad_corr": 1.5,
        "ns_nmad_raw": None,
        "ns_nmad_corr": None,
    },
]
THUMBS = [
    {"date": "2020-01-01", "sensor": "spot6", "res": 1.5},
    {"date": "2021-06-01", "sensor": "planetscope", "res": 3.0},
]


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "used_config.yaml").write_text(
        f"project: {{path: {tmp_path}}}\noutputs: {{level: 1}}\n"
    )
    write_record(
        tmp_path,
        StageRecord(
            stage="pairs",
            status="ok",
            pzones=["P"],
            summary={
                "pzones": {"P": {"n_pairs": 2, "n_components": 2, "thumbs": THUMBS}}
            },
            pairs=PAIRS,
            warnings=["w1", "w1"],
        ),
    )
    write_record(
        tmp_path,
        StageRecord(
            stage="correct",
            status="partial",
            pzones=["P"],
            summary={
                "masks": [{"CCFilter": {"cc_threshold": 0.5}}],
                "corrections": [{"RampCorrection": {}}],
                "pzones": {"P": {"n_pairs": 2, "n_failed": 1}},
            },
            pairs=PAIRS,
        ),
    )
    return tmp_path


def test_collect_orders_and_counts(run_dir):
    ctx = collect_run(run_dir)
    assert [r["stage"] for r in ctx["stages"]] == ["pairs", "correct"]
    assert ctx["warnings"] == [{"stage": "pairs", "message": "w1", "count": 2}]
    assert ctx["n_failed"] == 0


def test_render_html(run_dir):
    n_figs = len(plt.get_fignums())
    written = render_report(run_dir, pdf=False)
    html = written["html"].read_text()
    assert "Pair network" in html and "Corrections" in html
    assert "2 disconnected components" in html
    assert "CCFilter" in html and "RampCorrection" in html
    assert (run_dir / "assets" / "pairs" / "archive.png").exists()
    assert (run_dir / "assets" / "correct" / "nmad.png").exists()
    assert "pdf" not in written
    assert len(plt.get_fignums()) == n_figs


def test_render_pdf(run_dir):
    pytest.importorskip("plutoprint")
    written = render_report(run_dir, pdf=True)
    pdf = written["pdf"]
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert pdf.stat().st_size > 10_000  # figures embedded, not an empty page


def test_png_artifacts_copied(run_dir):
    import matplotlib

    fig, _ = plt.subplots()
    src = run_dir / "figs" / "chord.png"
    src.parent.mkdir()
    figures.save(fig, src)
    write_record(
        run_dir,
        StageRecord(
            stage="pairs",
            artifacts={"P/pairs_chord.png": "figs/chord.png", "P/plan": "missing.json"},
        ),
    )
    html = render_report(run_dir, pdf=False)["html"].read_text()
    assert "assets/pairs/P_pairs_chord.png" in html
    assert "missing.json" in html and "(missing)" in html
    assert matplotlib.get_backend().lower() == "agg"


def test_no_records_no_report(tmp_path):
    assert render_report(tmp_path, pdf=False) == {}


def test_worst_pairs_and_fmt():
    assert [p["pa_key"] for p in collect_mod.worst_pairs(PAIRS, 1)] == ["b"]
    assert collect_mod.fmt(None) == "—"
    assert collect_mod.fmt(0.123456) == "0.123"
    assert collect_mod.fmt(1e-5) == "1e-05"
    assert collect_mod.fmt(True) == "yes"


def test_table_and_scalar_items():
    cols, rows = collect_mod.table(
        [{"a": 1, "b": {"x": 1}, "k": [21, 21]}, {"a": 2, "c": "z"}]
    )
    assert cols == ["a", "k", "c"] and rows == [[1, [21, 21], None], [2, None, "z"]]
    assert collect_mod.fmt([21, 21]) == "[21, 21]"
    items = dict(
        collect_mod.scalar_items({"n": 1, "d": {"x": 1}, "l": [1, 2], "big": [{}]})
    )
    assert items == {"n": 1, "d": "x: 1", "l": "1, 2"}


@pytest.mark.parametrize(
    "plot, data",
    [
        (figures.plot_archive_timeline, THUMBS),
        (figures.plot_dt_histogram, PAIRS),
        (figures.plot_nmad_before_after, PAIRS),
    ],
)
def test_plot_conventions(plot, data):
    fig, ax = plot(data)
    try:
        assert ax.figure is fig
        fig2, ax2 = plot(data, ax=ax)
        assert fig2 is fig
    finally:
        plt.close(fig)
