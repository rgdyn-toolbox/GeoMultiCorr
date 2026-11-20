# -*- coding: utf-8 -*-
"""Cluster profiles and the ``cluster=`` string grammar."""

from __future__ import annotations

import pytest

from geomulticorr.scheduler import profiles
from geomulticorr.scheduler.profiles import (
    CLUSTER_CHOICES,
    CORRELATION_CLUSTER_CHOICES,
    DEFAULT_GRID_CLUSTERS,
    asp_env_lines,
    cluster_base_env,
    get_profile,
    is_grid_cluster,
    oar_header,
    parse_cluster,
    validate_cluster,
)

#: The environment blocks the ASP scripts carried before the profile table.
HISTORICAL_ENV = {
    "gricad": ["\n", "source $HOME/.bashrc\n", "conda activate E\n", "\n"],
    "isterre": [
        "\n",
        "source /soft/env.bash\n",
        "module load stereopipeline/3.6.0\n",
        "\n",
        "micromamba activate E\n",
        "\n",
    ],
}


class TestHistoricalBehaviour:
    def test_cluster_choices_unchanged(self):
        # TIO validates against this tuple; a grid target there would be a no-op.
        assert CLUSTER_CHOICES == (None, "local", "gricad", "isterre")

    @pytest.mark.parametrize("cluster", ["gricad", "isterre"])
    def test_asp_env_block_is_byte_identical(self, cluster):
        assert asp_env_lines(cluster, "E") == HISTORICAL_ENV[cluster]

    @pytest.mark.parametrize("cluster", [None, "local"])
    def test_local_has_no_env_block(self, cluster):
        assert asp_env_lines(cluster) == []

    def test_projects(self):
        assert profiles._OAR_PROJECT == {
            "gricad": "rgdyn",
            "isterre": "iste-equ-risques",
        }


class TestParseCluster:
    def test_bare_cigri_uses_default_grid_clusters(self):
        spec = parse_cluster("cigri")
        assert spec.is_grid
        assert spec.grid_clusters == DEFAULT_GRID_CLUSTERS == ("dahu",)
        assert spec.test_mode is False

    def test_named_grid_clusters(self):
        spec = parse_cluster("cigri:dahu,luke")
        assert spec.grid_clusters == ("dahu", "luke")
        assert spec.test_mode is False

    def test_test_mode(self):
        assert parse_cluster("cigri-test").test_mode is True
        spec = parse_cluster("cigri-test:bigfoot")
        assert (spec.grid_clusters, spec.test_mode) == (("bigfoot",), True)

    def test_dashed_cluster_names_are_allowed(self):
        assert parse_cluster("cigri:dahu-oar3").grid_clusters == ("dahu-oar3",)

    @pytest.mark.parametrize(
        "bad", ["cigri:", "cigri:dahu,", "cigri:dahu,dahu", "cigri:da hu", "cigri:a:b"]
    )
    def test_malformed_grid_spelling_raises(self, bad):
        with pytest.raises(ValueError, match="invalid grid cluster"):
            parse_cluster(bad)

    @pytest.mark.parametrize("cluster", [None, "local", "gricad", "isterre"])
    def test_historical_names(self, cluster):
        spec = parse_cluster(cluster)
        assert not spec.is_grid
        assert spec.grid_clusters == ()

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="cluster must be one of"):
            get_profile("bogus")

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("cigri", True),
            ("cigri-test:dahu", True),
            ("cigri:", True),
            ("gricad", False),
            (None, False),
            ("cigrix", False),
        ],
    )
    def test_is_grid_cluster(self, value, expected):
        assert is_grid_cluster(value) is expected


class TestValidateCluster:
    @pytest.mark.parametrize("cluster", ["cigri", "cigri:dahu,luke", "cigri-test"])
    def test_correlation_accepts_grid_spellings(self, cluster):
        assert validate_cluster(cluster, CORRELATION_CLUSTER_CHOICES) is None

    @pytest.mark.parametrize("cluster", ["cigri", "cigri-test:dahu"])
    def test_default_choices_reject_grid_with_a_reason(self, cluster):
        with pytest.raises(ValueError, match="only available for the pair correlation"):
            validate_cluster(cluster)

    def test_malformed_grid_spelling_raises_in_correlation(self):
        with pytest.raises(ValueError, match="invalid grid cluster"):
            validate_cluster("cigri:", CORRELATION_CLUSTER_CHOICES)


class TestGridScripts:
    @pytest.mark.parametrize("cluster", ["cigri", "cigri:luke", "cigri-test"])
    def test_no_oar_header(self, cluster):
        # CiGri passes the OAR options itself, from the JDL.
        assert oar_header("job", 1, 8, "01:00:00", cluster) == []

    @pytest.mark.parametrize("cluster", ["cigri", "cigri-test:dahu,luke"])
    def test_gricad_environment(self, cluster):
        assert asp_env_lines(cluster, "E") == HISTORICAL_ENV["gricad"]

    def test_base_env(self):
        assert cluster_base_env("cigri") == ["source $HOME/.bashrc\n"]
