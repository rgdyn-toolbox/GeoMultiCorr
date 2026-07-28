#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural smoke tests for the interactive plotly pair views (_pairs_plotly)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

from geomulticorr.utils._pairs_frame import (  # noqa: E402
    empty_pairs_frame,
    pairs_frame_from_indices,
    unique_couples,
)
from geomulticorr.utils import _pairs_plotly as pp  # noqa: E402


@pytest.fixture
def frame(dates_iso, sensors, index_pairs_redundancy):
    return pairs_frame_from_indices(
        dates_iso, index_pairs_redundancy, sensors=sensors, pz="PZ1"
    )


class TestDispatchTables:
    def test_builders_and_labels_agree(self):
        assert set(pp.VIEW_BUILDERS) == set(pp.VIEW_LABELS)

    def test_baseline_is_a_view(self):
        assert "baseline" in pp.VIEW_BUILDERS


class TestBaseline:
    def test_one_trace_per_direction(self, frame):
        fig = pp.figure_baseline(frame)
        assert [t.name for t in fig.data] == ["forward", "backward"]

    def test_point_counts_match_frame(self, frame):
        fig = pp.figure_baseline(frame)
        assert len(fig.data[0].x) == (frame["direction"] == "forward").sum()
        assert len(fig.data[1].x) == (frame["direction"] == "backward").sum()

    def test_title_applied(self, frame):
        assert pp.figure_baseline(frame, title="hello").layout.title.text == "hello"


class TestChord:
    def test_trace_count_is_constant(self, frame):
        """5 traces regardless of pair count — ring, links, hover, nodes, ticks."""
        assert len(pp.figure_chord(frame).data) == 5

    def test_arrows_add_exactly_one_trace(self, frame):
        """Arrowheads are one marker trace, appended last so indices 0-4 hold."""
        fig = pp.figure_chord(frame, show_arrows=True)
        assert len(fig.data) == 6
        assert len(fig.data[5].x) == len(unique_couples(frame))
        assert fig.data[5].marker.symbol == "triangle-up"
        assert fig.data[5].marker.angleref == "up"

    def test_arrow_angles_are_per_point(self, frame):
        fig = pp.figure_chord(frame, show_arrows=True)
        angles = np.asarray(fig.data[5].marker.angle, dtype=float)
        assert len(angles) == len(unique_couples(frame))
        # plotly normalises what it stores to (-180, 180]; the helper emits
        # [0, 360). Equivalent mod 360, which is what the renderer uses.
        assert np.all((angles % 360.0 >= 0.0) & (angles % 360.0 < 360.0))
        assert len(np.unique(angles)) > 1

    def test_arrows_off_by_default(self, frame):
        assert len(pp.figure_chord(frame).data) == 5

    def test_arrow_position_moves_the_heads(self, frame):
        mid = pp.figure_chord(frame, show_arrows=True, arrow_position=0.5).data[5]
        end = pp.figure_chord(frame, show_arrows=True, arrow_position=1.0).data[5]
        assert not np.allclose(mid.x, end.x)

    def test_arrow_heads_at_end_sit_on_the_ring(self, frame):
        """position=1.0 lands on the acquisition angles — the pile-up mid-chord avoids."""
        end = pp.figure_chord(frame, show_arrows=True, arrow_position=1.0).data[5]
        radii = np.hypot(np.asarray(end.x), np.asarray(end.y))
        assert np.allclose(radii, radii[0])

    def test_links_flattened_into_one_trace(self, frame):
        n_pts = 16
        fig = pp.figure_chord(frame, n_bezier_points=n_pts)
        n_curves = len(unique_couples(frame))
        assert len(fig.data[1].x) == n_curves * (n_pts + 1)

    def test_one_hover_anchor_per_chord(self, frame):
        fig = pp.figure_chord(frame)
        assert len(fig.data[2].x) == len(unique_couples(frame))

    def test_dedupe_halves_redundancy(self, frame):
        many = pp.figure_chord(frame, dedupe=False, n_bezier_points=8)
        few = pp.figure_chord(frame, dedupe=True, n_bezier_points=8)
        assert len(few.data[1].x) * 2 == len(many.data[1].x)

    def test_year_labels_are_annotations(self, frame):
        fig = pp.figure_chord(frame)
        assert len(fig.layout.annotations) == 3  # 2020, 2021, 2022
        assert {a.text for a in fig.layout.annotations} == {"2020", "2021", "2022"}

    def test_year_labels_can_be_disabled(self, frame):
        assert len(pp.figure_chord(frame, show_year_labels=False).layout.annotations) == 0

    def test_axes_locked_square(self, frame):
        fig = pp.figure_chord(frame)
        assert fig.layout.yaxis.scaleanchor == "x"
        assert fig.layout.xaxis.fixedrange is True


class TestNetwork:
    def test_direction_groups(self, frame):
        # forward + backward + hover anchors + nodes; mirroring adds annotations,
        # not traces, so this count is unaffected by the mirror setting.
        fig = pp.figure_network(frame, color_by="direction")
        assert len(fig.data) == 4

    def test_default_color_by_is_dt(self, frame):
        """Direction is encoded geometrically once mirrored, so colour shows Δt."""
        assert len(pp.figure_network(frame).data) == len(
            pp.figure_network(frame, color_by="dt").data
        )

    def test_dt_classes_bounded(self, frame):
        fig = pp.figure_network(frame, color_by="dt", n_dt_classes=3)
        assert len(fig.data) <= 3 + 2

    def test_hover_anchor_per_pair(self, frame):
        fig = pp.figure_network(frame, color_by="direction")
        assert len(fig.data[-2].x) == len(frame)

    def test_baseline_shape_drawn(self, frame):
        assert len(pp.figure_network(frame).layout.shapes) == 1

    def test_bad_color_by_raises(self, frame):
        with pytest.raises(ValueError, match="Unknown color_by"):
            pp.figure_network(frame, color_by="nope")


class TestNetworkMirror:
    def _arc_y(self, fig, name):
        trace = next(t for t in fig.data if t.name == name)
        return np.asarray(trace.y, dtype=float)

    def test_backward_arcs_go_below_axis(self, frame):
        fig = pp.figure_network(frame, color_by="direction", mirror_direction=True)
        assert np.nanmin(self._arc_y(fig, "backward")) < 0
        assert np.nanmax(self._arc_y(fig, "forward")) > 0

    def test_unmirrored_keeps_everything_above(self, frame):
        fig = pp.figure_network(frame, color_by="direction", mirror_direction=False)
        for name in ("forward", "backward"):
            assert np.nanmin(self._arc_y(fig, name)) >= -1e-12

    def test_y_range_straddles_zero_when_mirrored(self, frame):
        rng = pp.figure_network(frame, mirror_direction=True).layout.yaxis.range
        assert rng[0] < 0 < rng[1]

    def test_forward_only_frame_unaffected_by_mirror(self, dates_iso, index_pairs_consecutive):
        f = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        on = pp.figure_network(f, mirror_direction=True).layout.yaxis.range
        off = pp.figure_network(f, mirror_direction=False).layout.yaxis.range
        assert tuple(on) == tuple(off)

    def test_two_direction_annotations_when_both_present(self, frame):
        fig = pp.figure_network(frame, mirror_direction=True)
        texts = [a.text for a in fig.layout.annotations]
        assert any("forward" in t for t in texts)
        assert any("backward" in t for t in texts)

    def test_one_annotation_when_forward_only(self, dates_iso, index_pairs_consecutive):
        f = pairs_frame_from_indices(dates_iso, index_pairs_consecutive)
        texts = [a.text for a in pp.figure_network(f, mirror_direction=True).layout.annotations]
        assert len(texts) == 1
        assert "forward" in texts[0]

    def test_no_annotations_when_not_mirrored(self, frame):
        assert len(pp.figure_network(frame, mirror_direction=False).layout.annotations) == 0


class TestDtHist:
    def test_split_by_direction(self, frame):
        assert len(pp.figure_dt_hist(frame).data) == 2

    def test_single_histogram(self, frame):
        assert len(pp.figure_dt_hist(frame, split_direction=False).data) == 1

    def test_median_and_mean_lines(self, frame):
        fig = pp.figure_dt_hist(frame)
        assert len(fig.layout.shapes) == 2


class TestDegenerateInput:
    @pytest.mark.parametrize("view", ["baseline", "chord", "network", "dt_hist"])
    def test_empty_frame_never_raises(self, view):
        fig = pp.VIEW_BUILDERS[view](empty_pairs_frame())
        assert len(fig.data) == 0
        assert "No pairs" in fig.layout.annotations[0].text

    def test_chord_needs_two_dates(self):
        one = pairs_frame_from_indices(["2020-01-01", "2020-01-01"], [(0, 1)])
        fig = pp.figure_chord(one)
        assert "at least two acquisition dates" in fig.layout.annotations[0].text

    def test_single_pair_renders(self, dates_iso):
        frame = pairs_frame_from_indices(dates_iso, [(0, 1)])
        for view in pp.VIEW_BUILDERS:
            assert len(pp.VIEW_BUILDERS[view](frame).data) > 0
