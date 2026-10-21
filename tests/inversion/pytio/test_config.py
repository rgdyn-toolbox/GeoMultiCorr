#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``TIOConfig`` and its Fortran serialisation ``input_tio_text``.

The Fortran reads ``input_tio`` **positionally**, so the tests pin line
indices, not comments. ``LEGACY_TEXT`` is the block GMC hard-coded before the
weights fix, pasted verbatim: ``input_tio_text(TIOConfig.legacy())`` must stay
byte-identical to it so an inversion prepared by an earlier release can be
reproduced exactly.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from geomulticorr.inversion.pytio import (IPONDER, IPONDER_LINE_INDEX,
                                          TIOConfig, input_tio_text)

LEGACY_TEXT = (
    "0.0030  %  smoothing coefficient (threshold = 0.0001)\n"
    "1   %   remove points with large RMS misclosure  (y=0;n=1)\n"
    "1.2 %  threshold on RMS misclosure (in rad) ?\n"
    "1  % range and azimuth sampling ?\n"
    "0 % iterations to correct unwrapping errors (y:nb_of_iterations,n:0)\n"
    "0 % iterations to weight pixels of interferograms with large residual? (y:nb_of_iterations,n:0)\n"
    "0.2 % Scaling value for weighting residuals (in same unit as input files)\n"
    "0 % iterations to mask (tiny weight) pixels of interferograms with large residual? (y:nb_of_iterations,n:0)\n"
    "4 % threshold on residual, defining clearly wrong values (in same unit as input files)\n"
    "1    %   elimination of outliers by the median ? (y=0,n=1)\n"
    "liste_image_inv\n"
    "0    % sort by date (0) ou by another variable (1) ?\n"
    "liste_couple\n"
    "1   % interferogram format (RMG : 0; R4 :1) (date1-date2_pre_inv.unw or date1-date2.r4)\n"
    "3100.   %  include interferograms with bperp lower than maximal baseline\n"
    "1 % Weight input interferograms by coherence or correlation maps ? (y:0,n:1)\n"
    "0 % coherence file format (RMG : 0; R4 :1) (date1-date2.cor or date1-date2-CC.r4)\n"
    "1   %   minimal number of interferams using each image\n"
    "1     % interferograms weighting so that the weight per image is the same (y=0;n=1)\n"
    "0.6 % maximum fraction of discarded interferograms\n"
    "0 %  Would you like to restrict the area of inversion ?(y=1,n=0)\n"
    "1 735 1500 1585  %Give four corners, lower, left, top, right in file pixel coord\n"
    "1  %    referencing of interferograms by bands (1) or corners (2) ?\n"
    "5  %     band NW-SW(1), SW-SE(2), NW-NE(3), average of three bands(4), no referencement(5) ?\n"
    "1   %   Weigthing by image quality (y:0,n:1) ?\n"
    "0   %  Weigthing by interferogram variance (y:0,n:1) ?  or user given weight (2)?\n"
    "1    % use of covariance (y:0,n:1) ? (Obsolete)\n"
    "0   % include a baseline term in inversion ? (y:1;n:0) Requires smoothing !\n"
    "1   % smoothing by Laplacian, computed with a scheme at 3pts (0) or 5pts (1) ?\n"
    "2   % weigthed smoothing by the average time step (y:0; n:1, int:2) ?\n"
    "1    % put the first derivative to zero (y:0; n:1)?\n"
)


class TestLegacyRegression:
    def test_legacy_config_reproduces_the_old_text_byte_for_byte(self):
        assert input_tio_text(TIOConfig.legacy()) == LEGACY_TEXT

    def test_legacy_is_variance_weighting(self):
        cfg = TIOConfig.legacy()
        assert cfg.weight_mode == "variance"
        assert cfg.iponder == 0
        assert cfg.weights_applied_by_solver is False

    def test_legacy_accepts_overrides(self):
        cfg = TIOConfig.legacy(gamma=0.01)
        assert cfg.gamma == 0.01 and cfg.weight_mode == "variance"


class TestDefaults:
    def test_default_asks_the_solver_to_read_the_file_weights(self):
        cfg = TIOConfig()
        assert cfg.weight_mode == "file"
        assert cfg.iponder == 2
        assert cfg.weights_applied_by_solver is True

    def test_iponder_line_position_is_pinned(self):
        lines = input_tio_text().splitlines()
        assert len(lines) == 31
        assert lines[IPONDER_LINE_INDEX].startswith("2 ")
        assert "user given weight (2)" in lines[IPONDER_LINE_INDEX]

    def test_default_differs_from_legacy_only_on_the_iponder_line(self):
        new = input_tio_text().splitlines()
        old = LEGACY_TEXT.splitlines()
        diffs = [i for i, (a, b) in enumerate(zip(new, old)) if a != b]
        assert diffs == [IPONDER_LINE_INDEX]

    def test_iponder_table(self):
        assert IPONDER == {"variance": 0, "none": 1, "file": 2}
        for mode, value in IPONDER.items():
            line = input_tio_text(TIOConfig(weight_mode=mode)).splitlines()[IPONDER_LINE_INDEX]
            assert line.startswith(f"{value} ")


class TestFieldToLineMapping:
    """Every ported knob lands on its own line, rendered the way the Fortran expects."""

    @pytest.mark.parametrize("field, value, index, prefix", [
        ("gamma", 0.01, 0, "0.0100 "),
        ("mask_high_rms", True, 1, "0 "),
        ("rms_threshold", 2.5, 2, "2.5 "),
        ("unwrap_iterations", 3, 4, "3 "),
        ("reweight_iterations", 2, 5, "2 "),
        ("reweight_scale", 0.5, 6, "0.5 "),
        ("mask_iterations", 1, 7, "1 "),
        ("mask_residual_threshold", 6.0, 8, "6 "),
        ("min_pairs_per_image", 2, 17, "2 "),
        ("equalize_image_weights", True, 18, "0 "),
        ("frac_discard", 0.4, 19, "0.4 "),
        ("scheme", "3pt", 28, "0 "),
        ("pond_liss", 0, 29, "0 "),
        ("first_deriv_zero", True, 30, "0 "),
    ])
    def test_field_renders_on_its_line(self, field, value, index, prefix):
        lines = input_tio_text(TIOConfig(**{field: value})).splitlines()
        assert lines[index].startswith(prefix), lines[index]
        # and nothing else moved
        default = input_tio_text().splitlines()
        assert [i for i, (a, b) in enumerate(zip(lines, default)) if a != b] == [index]

    def test_file_names_are_the_two_bare_lines(self):
        lines = input_tio_text(None, "images.txt", "pairs.txt").splitlines()
        assert lines[10] == "images.txt"
        assert lines[12] == "pairs.txt"

    def test_none_config_means_defaults(self):
        assert input_tio_text(None) == input_tio_text(TIOConfig())


class TestValidation:
    def test_bad_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            TIOConfig(scheme="7pt")

    def test_bad_weight_mode(self):
        with pytest.raises(ValueError, match="weight_mode"):
            TIOConfig(weight_mode="user")

    def test_bad_pond_liss(self):
        with pytest.raises(ValueError, match="pond_liss"):
            TIOConfig(pond_liss=3)

    def test_smoothing_property_follows_the_fortran_threshold(self):
        assert TIOConfig(gamma=0.003).smoothing is True
        assert TIOConfig(gamma=0.00001).smoothing is False

    def test_n_robust_iterations_is_the_max(self):
        assert TIOConfig(reweight_iterations=2, mask_iterations=5).n_robust_iterations == 5


class TestToDict:
    def test_is_json_serialisable_and_complete(self):
        cfg = TIOConfig(gamma=0.01, reweight_iterations=2)
        d = cfg.to_dict()
        json.dumps(d)
        assert d["gamma"] == 0.01
        assert d["reweight_iterations"] == 2
        assert d["weight_mode"] == "file"
        assert d["shift"] is None and d["image_quality"] is None

    def test_arrays_become_type_tags(self):
        cfg = TIOConfig(shift=np.zeros(3), image_quality=np.ones(4))
        d = cfg.to_dict()
        assert d["shift"] == "<ndarray>"
        assert d["image_quality"] == "<ndarray>"
        json.dumps(d)
