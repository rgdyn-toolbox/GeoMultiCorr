#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the pure pairing-strategy index helper (_strategy_pair_indices)."""
from __future__ import annotations

import pytest

from geomulticorr.core.pzone import _strategy_pair_indices


class TestConsecutive:
    def test_indices(self):
        assert _strategy_pair_indices(4, "consecutive") == [(0, 1), (1, 2), (2, 3)]

    def test_single_thumb_empty(self):
        assert _strategy_pair_indices(1, "consecutive") == []

    def test_empty(self):
        assert _strategy_pair_indices(0, "consecutive") == []


class TestStep:
    def test_indices(self):
        # every pair within max_step=2 of each other
        assert _strategy_pair_indices(4, "step", max_step=2) == [
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
        ]

    def test_step_one_equals_consecutive(self):
        assert _strategy_pair_indices(5, "step", max_step=1) == \
            _strategy_pair_indices(5, "consecutive")

    def test_missing_max_step_raises(self):
        with pytest.raises(ValueError, match="'step' strategy requires max_step"):
            _strategy_pair_indices(4, "step")


class TestRedundancy:
    def test_indices_max_step_1(self):
        assert _strategy_pair_indices(3, "redundancy", max_step=1) == [
            (0, 1), (1, 0), (1, 2), (2, 1),
        ]

    def test_indices_max_step_2(self):
        # forward + backward for offsets 1 and 2
        assert _strategy_pair_indices(3, "redundancy", max_step=2) == [
            (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
        ]

    def test_missing_max_step_raises(self):
        with pytest.raises(ValueError, match="'redundancy' strategy requires max_step"):
            _strategy_pair_indices(4, "redundancy")


class TestForwardBackward:
    def test_indices(self):
        assert _strategy_pair_indices(3, "forward-backward") == [
            (0, 1), (1, 2), (1, 0), (2, 1),
        ]

    def test_single_thumb_empty(self):
        assert _strategy_pair_indices(1, "forward-backward") == []

    def test_same_membership_as_redundancy_step_1(self):
        """forward-backward is redundancy(max_step=1) up to ordering — documents
        the intentional overlap noted in the plan's Context."""
        fb = set(_strategy_pair_indices(5, "forward-backward"))
        red = set(_strategy_pair_indices(5, "redundancy", max_step=1))
        assert fb == red
        # ...but the emitted order genuinely differs
        assert _strategy_pair_indices(5, "forward-backward") != \
            _strategy_pair_indices(5, "redundancy", max_step=1)


class TestErrors:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy 'nope'"):
            _strategy_pair_indices(4, "nope")
