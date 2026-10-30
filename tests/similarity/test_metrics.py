"""Similarity metrics (similarity/metrics.py): each pins the invariance it promises."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from geomulticorr.similarity import metrics as sm
from geomulticorr.similarity.metrics import SIMILARITY_KEYS


def _texture(shape=(256, 256), sigma=2.0, seed=0):
    """Band-limited random texture, unit variance — a stand-in for terrain."""
    g = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=shape), sigma)
    return (g - g.mean()) / g.std()


@pytest.fixture
def tex():
    return _texture()


@pytest.fixture
def other():
    return _texture(seed=1)


# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_nmad_of_gaussian_is_sigma(self):
        x = np.random.default_rng(0).normal(scale=2.0, size=200_000)
        assert sm.nmad(x) == pytest.approx(2.0, rel=0.02)

    def test_nmad_empty_is_nan(self):
        assert np.isnan(sm.nmad([np.nan, np.nan]))

    def test_shape_mismatch_raises(self, tex):
        with pytest.raises(ValueError, match="same grid"):
            sm.pearson(tex, tex[:-1])

    def test_mask_shape_mismatch_raises(self, tex):
        with pytest.raises(ValueError, match="mask shape"):
            sm.joint_valid(tex, tex, np.ones((3, 3), bool))

    def test_joint_valid_combines_nan_and_mask(self, tex):
        b = tex.copy()
        b[0, 0] = np.nan
        mask = np.ones_like(tex, bool)
        mask[1, 1] = False
        v = sm.joint_valid(tex, b, mask)
        assert not v[0, 0] and not v[1, 1] and v.sum() == tex.size - 2


# --------------------------------------------------------------------------- #
class TestGlobal:
    def test_pearson_affine_invariant(self, tex):
        assert sm.pearson(tex, 3.0 * tex + 100.0) == pytest.approx(1.0)
        assert sm.pearson(tex, -tex) == pytest.approx(-1.0)

    def test_pearson_independent_near_zero(self, tex, other):
        assert abs(sm.pearson(tex, other)) < 0.1

    def test_pearson_flat_is_nan(self, tex):
        assert np.isnan(sm.pearson(tex, np.ones_like(tex)))

    def test_spearman_monotonic_invariant_pearson_not(self, tex):
        b = np.exp(3 * tex)
        assert sm.spearman(tex, b) == pytest.approx(1.0)
        assert sm.pearson(tex, b) < 0.9

    def test_linfoot_matches_rho_for_gaussian(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 500))
        y = 0.8 * x + 0.6 * rng.normal(size=x.shape)
        mi = sm.mutual_information(x, y)
        assert mi["linfoot"] == pytest.approx(0.8, abs=0.05)
        assert 1.0 < mi["nmi"] < 2.0

    def test_mi_independent_bias_corrected_near_zero(self, tex, other):
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=(400, 500)), rng.normal(size=(400, 500))
        assert sm.mutual_information(x, y)["linfoot"] < 0.08

    def test_mi_sees_non_monotonic_relation(self):
        x = np.random.default_rng(0).uniform(-1, 1, size=(300, 300))
        y = x**2  # symmetric: rank correlation ~0, fully dependent
        assert abs(sm.spearman(x, y)) < 0.05
        assert sm.mutual_information(x, y)["linfoot"] > 0.9

    def test_mi_too_few_samples_nan(self):
        assert np.isnan(
            sm.mutual_information(np.ones((5, 5)), np.ones((5, 5)))["mi_nats"]
        )

    def test_histogram_identical_shapes(self, tex):
        h = sm.histogram_comparison(tex, 2 * tex + 5)
        assert h["hist_intersection"] == pytest.approx(1.0)
        assert h["hist_chi2"] == pytest.approx(0.0, abs=1e-12)
        assert h["gain"] == pytest.approx(2.0) and h["offset"] == pytest.approx(5.0)
        assert h["std_ratio"] == pytest.approx(2.0)

    def test_histogram_blind_to_spatial_arrangement(self, tex):
        """The reason it is a diagnostic, not a predictor."""
        shuffled = np.random.default_rng(0).permutation(tex.ravel()).reshape(tex.shape)
        assert sm.histogram_comparison(tex, shuffled)[
            "hist_intersection"
        ] == pytest.approx(1.0)
        assert abs(sm.pearson(tex, shuffled)) < 0.05


# --------------------------------------------------------------------------- #
class TestLocal:
    def test_local_zncc_affine_invariant(self, tex):
        z = sm.local_zncc(tex, 4 * tex - 2, window=15)
        assert np.nanmin(z) == pytest.approx(1.0, abs=1e-6)

    def test_local_zncc_survives_spatially_varying_gain(self, tex):
        """Per-region gain/offset (haze, illumination) hurts global Pearson only."""
        b = tex.copy()
        for (r, c), (g, o) in zip(
            [(0, 0), (0, 128), (128, 0), (128, 128)],
            [(0.3, 0.0), (1.0, 50.0), (3.0, -20.0), (8.0, 200.0)],
        ):
            b[r : r + 128, c : c + 128] = g * tex[r : r + 128, c : c + 128] + o
        assert np.nanmedian(sm.local_zncc(tex, b, window=15)) > 0.99
        assert sm.pearson(tex, b) < 0.5

    def test_local_zncc_independent(self, tex, other):
        assert abs(np.nanmedian(sm.local_zncc(tex, other, window=31))) < 0.1

    def test_local_zncc_respects_mask_and_nan(self, tex):
        b = tex.copy()
        b[:50] = np.nan
        z = sm.local_zncc(tex, b, window=15)
        # corners fall under min_fraction (a quarter of the window is inside)
        assert np.isnan(z[:50]).all() and np.isfinite(z[100:-10, 10:-10]).all()

    def test_ssim_separates_radiometry_from_structure(self, tex):
        c = sm.ssim_components(tex, 0.3 * tex + 2.0, window=15)
        assert c["ssim_structure"] > 0.99
        assert c["ssim_luminance"] < 0.9 or c["ssim_contrast"] < 0.9

    def test_ssim_empty_nan(self, tex):
        c = sm.ssim_components(tex, tex, 15, mask=np.zeros_like(tex, bool))
        assert all(np.isnan(v) for v in c.values())

    def test_oc_gain_invariant_and_polarity(self, tex):
        assert sm.orientation_correlation(tex, 5 * tex + 3)["oc"] == pytest.approx(1.0)
        flip = sm.orientation_correlation(tex, -tex)
        assert flip["oc"] == pytest.approx(-1.0)
        assert flip["oc_axial"] == pytest.approx(1.0)

    def test_oc_independent(self, tex, other):
        oc = sm.orientation_correlation(tex, other)
        assert abs(oc["oc"]) < 0.1 and abs(oc["oc_axial"]) < 0.1

    def test_oc_too_few_pixels(self):
        assert np.isnan(
            sm.orientation_correlation(np.ones((2, 2)), np.ones((2, 2)))["oc"]
        )


# --------------------------------------------------------------------------- #
class TestCoherence:
    def test_identical_is_coherent_everywhere(self, tex):
        c = sm.spectral_coherence(tex, tex, tile=64, res=3.0)
        assert np.nanmin(c["coherence"]) == pytest.approx(1.0)
        assert c["cutoff_wavelength_m"] == pytest.approx(6.0)  # Nyquist
        assert c["n_tiles"] == 16 and c["noise_floor"] == pytest.approx(1 / 16)

    def test_unaffected_by_blurring_one_side(self, tex):
        """A softer sensor (linear filter) does not lower coherence."""
        c = sm.spectral_coherence(
            tex, ndimage.gaussian_filter(tex, 1.0, mode="wrap"), tile=64
        )
        assert np.nanmedian(c["coherence"][:20]) > 0.95

    def test_noise_sets_a_cutoff(self):
        base = _texture((512, 512), sigma=1.0)
        noisy = base + 1.0 * np.random.default_rng(5).normal(size=base.shape)
        c = sm.spectral_coherence(base, noisy, tile=64, res=1.0)
        assert c["coherence"][0] > 0.8 and c["coherence"][-1] < 0.5
        assert 2.0 < c["cutoff_wavelength_m"] < 64.0

    def test_shift_does_not_destroy_coherence(self, tex):
        c = sm.spectral_coherence(tex, np.roll(tex, (2, -3), axis=(0, 1)), tile=64)
        assert np.nanmedian(c["coherence"][:20]) > 0.8

    def test_independent_is_near_noise_floor(self):
        a, b = _texture((512, 512), seed=7), _texture((512, 512), seed=8)
        c = sm.spectral_coherence(a, b, tile=64)
        assert np.nanmean(c["coherence"]) < 3 * c["noise_floor"]
        assert np.isnan(c["cutoff_wavelength_m"])

    def test_too_few_tiles_nan(self, tex):
        c = sm.spectral_coherence(tex[:100, :100], tex[:100, :100], tile=64)
        assert c["n_tiles"] < 4 and np.isnan(c["coherence"]).all()


# --------------------------------------------------------------------------- #
class TestTrialCorrelation:
    def test_recovers_subpixel_shift(self):
        a = _texture((400, 400), sigma=2.0)
        b = ndimage.shift(a, (0.7, -1.3), order=3, mode="wrap")  # dy, dx
        t = sm.trial_correlation(a, b, kernel=31, search=5, n_tiles=50)
        assert t["trial_valid_frac"] == 1.0
        assert t["trial_dy_median_px"] == pytest.approx(0.7, abs=0.15)
        assert t["trial_dx_median_px"] == pytest.approx(-1.3, abs=0.15)
        assert t["trial_sigma_px"] < 0.1
        assert t["trial_peak_median"] > 0.95

    def test_independent_images_fail(self):
        a, b = _texture((400, 400), seed=1), _texture((400, 400), seed=2)
        t = sm.trial_correlation(a, b, kernel=31, search=5, n_tiles=50, min_peak=0.5)
        assert t["trial_valid_frac"] < 0.2

    def test_shift_beyond_search_rails(self):
        a = _texture((400, 400))
        b = np.roll(a, 9, axis=1)
        t = sm.trial_correlation(a, b, kernel=31, search=4, n_tiles=50)
        assert t["trial_railed_frac"] > 0.8 and t["trial_valid_frac"] < 0.2

    def test_mask_limits_tiles(self, tex):
        mask = np.zeros_like(tex, bool)
        mask[:, :128] = True
        t = sm.trial_correlation(tex, tex, mask, kernel=21, search=3, return_tiles=True)
        assert (t["tiles"]["col"] < 128).all()

    def test_nothing_to_sample(self, tex):
        t = sm.trial_correlation(tex, tex, np.zeros_like(tex, bool), kernel=21)
        assert t["trial_n_tiles"] == 0 and np.isnan(t["trial_valid_frac"])

    @pytest.mark.parametrize("kw", [{"kernel": 20}, {"kernel": 1}, {"search": 0}])
    def test_bad_parameters_raise(self, tex, kw):
        with pytest.raises(ValueError):
            sm.trial_correlation(tex, tex, **kw)

    def test_reproducible(self, tex, other):
        b = tex + 0.5 * other
        t1 = sm.trial_correlation(tex, b, kernel=21, n_tiles=10, rng=4)
        t2 = sm.trial_correlation(tex, b, kernel=21, n_tiles=10, rng=4)
        assert t1 == t2


# --------------------------------------------------------------------------- #
class TestReport:
    def test_keys_and_order(self, tex):
        r = sm.similarity_report(tex, 2 * tex + 1, kernel=21, search=3, tile=64)
        assert tuple(r) == SIMILARITY_KEYS

    def test_keys_unique(self):
        assert len(set(SIMILARITY_KEYS)) == len(SIMILARITY_KEYS)

    def test_similar_beats_dissimilar(self, tex, other):
        good = sm.similarity_report(tex, 2 * tex + 1 + 0.2 * other, kernel=21, tile=64)
        bad = sm.similarity_report(tex, other, kernel=21, tile=64)
        for k in (
            "pearson",
            "linfoot",
            "zncc_local_median",
            "oc",
            "coh_mean",
            "trial_valid_frac",
            "trial_peak_median",
        ):
            assert good[k] > bad[k], k

    def test_empty_mask_gives_nan_row(self, tex):
        r = sm.similarity_report(tex, tex, mask=np.zeros_like(tex, bool))
        assert r["n_pixels"] == 0 and r["trial_n_tiles"] == 0
        assert np.isnan(r["pearson"]) and np.isnan(r["trial_valid_frac"])
