#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# corr_params.py
# creation date: 2026-09-10.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/agpl-3.0.txt
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------- #
r"""Pure-math derivation of ASP correlation parameters.

All functions operate on plain floats and NumPy arrays — no
:class:`geoutils.Raster`, no :class:`~geomulticorr.core.pair.Pair`, no GMC
imports at all.  Same design intent as :mod:`geomulticorr.corrections.fit` and
:mod:`geomulticorr.utils._pairs_geometry`: the maths lives in a dependency-light
module underneath everything that renders or persists it.

The theory these functions implement is documented in
``docs/correlation_parameters.md``; section references below point there.

The central relation (guide §2.1) is that a pair yields a usable measurement when

.. math::

    k\,c\,r \;<\; v\,\Delta t \;<\; (S - \delta_{coreg} - m)\,r

with :math:`r` the ground sample distance (m/px), :math:`c` the subpixel matching
precision (px), :math:`k` a confidence multiplier, :math:`S` the search
half-width (px), :math:`v` the surface velocity (m/yr) and :math:`\Delta t` the
temporal baseline (yr).  Both bounds scale as :math:`1/\Delta t`, so the band
between them has the **resolution-independent** width :math:`S/(k c)` — which is
what :func:`dynamic_range` returns and what makes :func:`n_parameter_sets`
answerable at all.

Public API:

- :func:`expected_displacement` / :func:`displacement_px` — what a pair measures.
- :func:`noise_floor_m` / :func:`min_detectable_velocity` / :func:`snr` — the
  lower bound.
- :func:`required_search_px` / :func:`max_measurable_velocity` — the upper bound.
- :func:`kernel_from_footprint` / :func:`strain_kernel_limit_px` — kernel sizing.
- :func:`dynamic_range` / :func:`n_parameter_sets` — how many parameter sets an
  archive needs.
- :func:`coherence` / :func:`optimal_baseline_days` — decorrelation (guide §5).
- :func:`cost_index` — relative runtime, for comparing candidate settings.
- :func:`geometric_dt_bins` / :func:`assign_dt_bin` — the grouping axis.
- :func:`feasibility_bounds` — the two boundary curves, for the design map.
- :func:`suggest_parameters` — the whole derivation for one pair.
"""
from __future__ import annotations

import math

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from geomulticorr._typing import NDArrayNum

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #

#: Days per Julian year — the same conversion ``Pair.pa_dt_years`` uses
#: (``pair.py``), kept identical so a Δt in years never disagrees between the
#: two.
DAYS_PER_YEAR: float = 365.25

#: Smallest kernel that still carries enough texture for an unambiguous match
#: (guide §3.1, "texture floor").  Below this, block matching degenerates on
#: smooth terrain — snow, scree, shadowed slopes — and the blunder rate rises
#: sharply.
TEXTURE_FLOOR_PX: int = 7

#: Differential displacement across the template, in pixels, above which the
#: template has itself deformed too much to match (guide §3.1, "strain
#: ceiling").  One pixel is the conventional tolerance.
STRAIN_TOLERANCE_PX: float = 1.0

#: Algorithms that force a 9x9 kernel, ``subpixel-kernel 9 9`` and
#: ``cost-mode 4``.  Kept here — rather than only inside
#: :meth:`~geomulticorr.correlation.correlation.ASP.build_correlation_params` —
#: so a caller can *predict* the override instead of discovering it in a log
#: line after the fact.
SGM_ALGORITHMS: tuple[str, ...] = ("asp_sgm", "asp_mgm", "asp_final_mgm")

#: Kernel and cost mode SGM/MGM impose.  See :func:`sgm_kernel_override`.
SGM_KERNEL: tuple[int, int] = (9, 9)
SGM_COST_MODE: int = 4

#: Default relative-cost ceiling above which :func:`suggest_parameters` warns.
#: Dimensionless: 1.0 is ASP's default ``corr-kernel 21 21`` with a 5 px search
#: half-width, so a value of 20 means "twenty times the reference correlation".
DEFAULT_COST_BUDGET: float = 20.0

#: Reference point for :func:`cost_index`.
_COST_REF_KERNEL_PX: float = 21.0
_COST_REF_SEARCH_PX: float = 5.0

#: Exponent in the empirical kernel/precision relation
#: ``sigma ~ sigma_0 (K_0/K)**_KERNEL_PRECISION_EXPONENT`` (guide §3.2).  The
#: benefit is modest and saturating; this is a fitted shape, not a law.
_KERNEL_PRECISION_EXPONENT: float = 0.5

#: Kernel at which a quoted subpixel precision ``c`` is taken to have been
#: measured, used only when :func:`noise_floor_m` is asked to scale ``c`` with
#: kernel size.
_KERNEL_PRECISION_REFERENCE_PX: float = 21.0


# ---------------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------------- #
def days_to_years(dt_days: float | NDArrayNum) -> float | NDArrayNum:
    """Convert a temporal baseline in days to Julian years.

    :param dt_days: Temporal baseline(s) in days.
    :returns: Baseline(s) in years, using :data:`DAYS_PER_YEAR`.
    """
    return np.asarray(dt_days, dtype="float64") / DAYS_PER_YEAR if np.ndim(dt_days) else float(dt_days) / DAYS_PER_YEAR


def to_odd(value: float, *, minimum: int = 1) -> int:
    """Round *value* to the nearest odd integer at or above *minimum*.

    ASP correlation kernels must be odd so the template has a well-defined
    centre pixel.  Rounding is to nearest, then adjusted **upward** to odd — a
    28 px request becomes 29, not 27, so the ground footprint is never quietly
    smaller than asked for.

    :param value: Desired kernel side, in pixels.
    :param minimum: Floor applied after rounding.
    :returns: An odd integer >= ``minimum``.
    """
    if not np.isfinite(value):
        return int(minimum) | 1
    n = int(round(float(value)))
    if n % 2 == 0:
        n += 1
    floor = int(minimum)
    if floor % 2 == 0:
        floor += 1
    return max(n, floor)


def sgm_kernel_override(
    corr_algorithm: str,
    corr_kernel: tuple[int, int],
    subpixel_kernel: tuple[int, int],
    cost_mode: int,
) -> tuple[tuple[int, int], tuple[int, int], int, bool]:
    """Apply the SGM/MGM kernel and cost-mode override.

    Semi-global matching works on small census-transform windows and imposes
    smoothness globally instead, so ASP forces a 9x9 kernel.  **Any kernel the
    caller requested is discarded** in these modes.

    This lives here, rather than only inside ``build_correlation_params``, so
    that every consumer resolves the same values: the parameters file, the
    ``corr_eval`` command that computes the CC map, and
    :func:`suggest_parameters`.  Applying it in only one of those is what let
    ``parallel_stereo`` run 9x9 while ``corr_eval --kernel-size`` still received
    21x21 — a CC map computed over a different window than the disparity it
    describes.

    :param corr_algorithm: ASP ``stereo-algorithm`` value.
    :param corr_kernel: Requested correlation kernel ``(h, w)``.
    :param subpixel_kernel: Requested subpixel kernel ``(h, w)``.
    :param cost_mode: Requested cost mode.
    :returns: ``(corr_kernel, subpixel_kernel, cost_mode, overridden)`` — the
        effective values, and whether the override fired.
    """
    if corr_algorithm in SGM_ALGORITHMS:
        return SGM_KERNEL, SGM_KERNEL, SGM_COST_MODE, True
    return corr_kernel, subpixel_kernel, cost_mode, False


# ---------------------------------------------------------------------------- #
# What a pair measures
# ---------------------------------------------------------------------------- #
def expected_displacement(velocity_m_yr: float, dt_days: float) -> float:
    """Ground displacement a pair is expected to measure, in metres.

    ``D = v * dt``.  This — not the velocity and not the baseline separately —
    is the quantity that decides whether a pair works (guide §1): two pairs with
    the same ``v * dt`` pose the same correlation problem.

    :param velocity_m_yr: Surface velocity, m/yr.
    :param dt_days: Temporal baseline, days.
    :returns: Expected displacement, metres.
    """
    return float(velocity_m_yr) * float(dt_days) / DAYS_PER_YEAR


def displacement_px(velocity_m_yr: float, dt_days: float, resolution_m: float) -> float:
    """Expected displacement expressed in pixels of the **left** image.

    ASP reports disparity in left-image pixels, so a cross-sensor pair must use
    the left image's GSD here (guide §4.4).

    :param velocity_m_yr: Surface velocity, m/yr.
    :param dt_days: Temporal baseline, days.
    :param resolution_m: Ground sample distance, m/px.
    :returns: Expected displacement, pixels.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    return expected_displacement(velocity_m_yr, dt_days) / float(resolution_m)


# ---------------------------------------------------------------------------- #
# The lower bound: noise floor and detectability
# ---------------------------------------------------------------------------- #
def noise_floor_m(
    resolution_m: float,
    c_px: float,
    kernel_px: int | None = None,
) -> float:
    """Displacement uncertainty ``sigma_D``, in metres.

    ``sigma_D = c * r``.  Image correlation has a precision that is a roughly
    fixed *fraction of a pixel*, largely independent of how far the ground
    moved; converting that fraction to metres is what makes resolution matter
    (guide §1).

    When *kernel_px* is given, ``c`` is scaled by the empirical and saturating
    kernel relation of guide §3.2,
    ``c(K) = c * (K_ref/K) ** 0.5`` with ``K_ref = 21``.  Pass ``None`` (the
    default) to take ``c`` at face value — which is the right choice when ``c``
    was measured from stable-ground NMAD at the kernel you intend to use.

    :param resolution_m: Ground sample distance, m/px.
    :param c_px: Subpixel matching precision, pixels.
    :param kernel_px: Kernel side the precision should be scaled to, or None.
    :returns: Displacement uncertainty, metres.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    if c_px <= 0:
        raise ValueError(f"c_px must be positive, got {c_px}")
    c = float(c_px)
    if kernel_px is not None:
        if kernel_px <= 0:
            raise ValueError(f"kernel_px must be positive, got {kernel_px}")
        c *= (_KERNEL_PRECISION_REFERENCE_PX / float(kernel_px)) ** _KERNEL_PRECISION_EXPONENT
    return c * float(resolution_m)


def min_detectable_velocity(
    dt_days: float | NDArrayNum,
    resolution_m: float,
    c_px: float,
    k: float = 3.0,
    kernel_px: int | None = None,
) -> float | NDArrayNum:
    """Slowest velocity a pair of this baseline can resolve, in m/yr.

    ``v_min = k * sigma_D / dt`` — the detection floor of guide §2.1, plotted as
    the lower slope-(-1) line of the design map.  Anything below it is noise,
    not a noisy measurement.

    :param dt_days: Temporal baseline(s), days.  Arrays are supported.
    :param resolution_m: Ground sample distance, m/px.
    :param c_px: Subpixel matching precision, pixels.
    :param k: Confidence multiplier (2 or 3).
    :param kernel_px: Optional kernel for precision scaling, see
        :func:`noise_floor_m`.
    :returns: Minimum detectable velocity, m/yr.
    """
    sigma = noise_floor_m(resolution_m, c_px, kernel_px)
    dt_years = np.asarray(dt_days, dtype="float64") / DAYS_PER_YEAR
    with np.errstate(divide="ignore", invalid="ignore"):
        out = float(k) * sigma / dt_years
    return out if np.ndim(dt_days) else float(out)


def snr(
    velocity_m_yr: float,
    dt_days: float,
    resolution_m: float,
    c_px: float,
    kernel_px: int | None = None,
) -> float:
    """Signal-to-noise ratio of a pair's displacement measurement.

    ``SNR = v * dt / sigma_D``.  Compare against *k*: a pair with ``SNR < k``
    sits below the detection floor.

    :param velocity_m_yr: Surface velocity, m/yr.
    :param dt_days: Temporal baseline, days.
    :param resolution_m: Ground sample distance, m/px.
    :param c_px: Subpixel matching precision, pixels.
    :param kernel_px: Optional kernel for precision scaling.
    :returns: Dimensionless SNR; ``inf`` where the noise floor is zero.
    """
    sigma = noise_floor_m(resolution_m, c_px, kernel_px)
    if sigma == 0:
        return math.inf
    return expected_displacement(velocity_m_yr, dt_days) / sigma


# ---------------------------------------------------------------------------- #
# The upper bound: search range
# ---------------------------------------------------------------------------- #
def required_search_px(
    velocity_m_yr: float,
    dt_days: float,
    resolution_m: float,
    *,
    coreg_px: float = 2.0,
    margin_px: float = 2.0,
    flow_azimuth_deg: float | None = None,
    cross_flow_px: float = 2.0,
) -> tuple[int, int, int, int]:
    """Search box covering the expected displacement, as ASP's four integers.

    Returns ``(x_min, y_min, x_max, y_max)`` for ``--corr-search``, in pixels.

    The box must cover the **sum** of true ground motion and any residual
    misregistration, or the true correlation peak sits outside it and the
    disparity rails against an edge:

    ``S = v * dt / r + coreg_px + margin_px``

    When *flow_azimuth_deg* is given the box is made **asymmetric**, elongated
    along flow and only ``cross_flow_px`` (plus allowances) across it.  This is
    the single largest cost saving available (guide §6.1): a symmetric box
    covering an 80 px reach searches 1600 times the area of one that is 80 px
    along flow and 2 px across, for no additional signal.

    Azimuth follows the compass convention — degrees clockwise from north, so 90
    is due east (+x) and 180 due south (+y in raster row order, since rows
    increase downward/southward).

    :param velocity_m_yr: Surface velocity, m/yr.
    :param dt_days: Temporal baseline, days.
    :param resolution_m: Ground sample distance, m/px.
    :param coreg_px: Residual co-registration allowance, pixels.
    :param margin_px: Safety margin, pixels.
    :param flow_azimuth_deg: Flow direction, degrees clockwise from north, or
        None for a symmetric box.
    :param cross_flow_px: Half-width across flow when an azimuth is given.
    :returns: ``(x_min, y_min, x_max, y_max)`` in pixels.
    """
    d_px = abs(displacement_px(velocity_m_yr, dt_days, resolution_m))
    allowance = float(coreg_px) + float(margin_px)

    # A NaN resolution reaches here whenever a thumb header could not be read.
    # _thumb_resolution degrades to NaN deliberately rather than aborting, so
    # one unreadable file must not take the whole derivation down with an
    # opaque "cannot convert float NaN to integer" from math.ceil below.
    if not math.isfinite(d_px):
        half = int(math.ceil(allowance)) or 1
        return (-half, -half, half, half)

    if flow_azimuth_deg is None:
        half = int(math.ceil(d_px + allowance))
        return (-half, -half, half, half)

    theta = math.radians(float(flow_azimuth_deg))
    # Compass azimuth -> raster axes.  +x is east, +y is south (row index grows
    # downward), so a due-south flow (180 deg) is +y.
    dx = d_px * math.sin(theta)
    dy = -d_px * math.cos(theta)

    cross = float(cross_flow_px) + allowance
    x_lo = int(math.floor(min(dx, 0.0) - cross))
    x_hi = int(math.ceil(max(dx, 0.0) + cross))
    y_lo = int(math.floor(min(dy, 0.0) - cross))
    y_hi = int(math.ceil(max(dy, 0.0) + cross))
    return (x_lo, y_lo, x_hi, y_hi)


def search_half_width_px(corr_search: Sequence[int] | None) -> float:
    """Largest half-extent of a ``--corr-search`` box, in pixels.

    The scalar ``S`` the guide's formulas use.  ``None`` (ASP auto-detection)
    has no defined reach, so this returns ``nan``.

    :param corr_search: ``(x_min, y_min, x_max, y_max)`` or None.
    :returns: ``max(|x_min|, |x_max|, |y_min|, |y_max|)``, or nan.
    """
    if corr_search is None:
        return float("nan")
    return float(max(abs(int(v)) for v in corr_search))


def max_measurable_velocity(
    dt_days: float | NDArrayNum,
    resolution_m: float,
    search_px: float,
    *,
    coreg_px: float = 2.0,
    margin_px: float = 2.0,
) -> float | NDArrayNum:
    """Fastest velocity a search box of *search_px* can follow, in m/yr.

    ``v_max = (S - coreg - margin) * r / dt`` — the search-reach ceiling of
    guide §2.1, the upper slope-(-1) line of the design map.

    :param dt_days: Temporal baseline(s), days.  Arrays are supported.
    :param resolution_m: Ground sample distance, m/px.
    :param search_px: Search half-width ``S``, pixels.
    :param coreg_px: Residual co-registration allowance, pixels.
    :param margin_px: Safety margin, pixels.
    :returns: Maximum measurable velocity, m/yr.
    """
    reach_px = max(float(search_px) - float(coreg_px) - float(margin_px), 0.0)
    dt_years = np.asarray(dt_days, dtype="float64") / DAYS_PER_YEAR
    with np.errstate(divide="ignore", invalid="ignore"):
        out = reach_px * float(resolution_m) / dt_years
    return out if np.ndim(dt_days) else float(out)


# ---------------------------------------------------------------------------- #
# Kernel sizing
# ---------------------------------------------------------------------------- #
def kernel_from_footprint(
    footprint_m: float,
    resolution_m: float,
    *,
    k_min: int = TEXTURE_FLOOR_PX,
    k_max: int | None = None,
) -> int:
    """Kernel side, in pixels, giving a template footprint of *footprint_m*.

    ``K = L / r``, rounded to odd.  This is the conversion that makes a
    heterogeneous archive consistent (guide §4.1): fix the physical scale in
    metres, then convert per sensor.  A fixed *pixel* kernel silently applies
    different physical filters — 21 px is 31.5 m on 1.5 m SPOT and 63 m on 3 m
    PlanetScope — and feeding fields of different effective resolution into one
    inversion combines measurements of different quantities.

    :param footprint_m: Desired template ground footprint, metres.
    :param resolution_m: Ground sample distance, m/px.
    :param k_min: Texture floor, pixels.
    :param k_max: Optional ceiling (e.g. from :func:`strain_kernel_limit_px`).
    :returns: Odd kernel side, pixels, clamped into ``[k_min, k_max]``.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    if footprint_m <= 0:
        raise ValueError(f"footprint_m must be positive, got {footprint_m}")
    kernel = to_odd(float(footprint_m) / float(resolution_m), minimum=k_min)
    if k_max is not None:
        ceiling = to_odd(k_max, minimum=k_min)
        # to_odd rounds up to odd, which can push a ceiling above itself; step
        # back down rather than return a kernel larger than the caller allowed.
        if ceiling > k_max and ceiling - 2 >= k_min:
            ceiling -= 2
        kernel = min(kernel, ceiling)
    return kernel


def footprint_m(kernel_px: int, resolution_m: float) -> float:
    """Ground footprint of a kernel, in metres: ``L = K * r``.

    :param kernel_px: Kernel side, pixels.
    :param resolution_m: Ground sample distance, m/px.
    :returns: Template footprint, metres.
    """
    return float(kernel_px) * float(resolution_m)


def strain_kernel_limit_px(
    strain_rate_per_yr: float,
    dt_days: float,
    *,
    tolerance_px: float = STRAIN_TOLERANCE_PX,
) -> float:
    """Largest kernel that survives the deformation accumulated over *dt_days*.

    ``K <= tolerance / (|grad v| * dt)`` (guide §3.1).  If the displacement
    varies across the template, the surface inside the window has genuinely
    deformed between acquisitions and the template no longer matches anything.

    The consequence is counterintuitive and worth restating: **long baselines
    want smaller kernels in high-strain zones, not larger ones** — the opposite
    of the "long baseline means big window" reflex, which confuses the kernel
    with the search range.

    :param strain_rate_per_yr: Velocity gradient magnitude, 1/yr.
    :param dt_days: Temporal baseline, days.
    :param tolerance_px: Differential displacement tolerated across the
        template, pixels.
    :returns: Kernel ceiling in pixels; ``inf`` when the strain rate is zero.
    """
    strain = abs(float(strain_rate_per_yr)) * float(dt_days) / DAYS_PER_YEAR
    if strain <= 0:
        return math.inf
    return float(tolerance_px) / strain


# ---------------------------------------------------------------------------- #
# How many parameter sets does an archive need?
# ---------------------------------------------------------------------------- #
def dynamic_range(search_px: float, c_px: float, k: float = 3.0) -> float:
    """Multiplicative window of ``v * dt`` one parameter set can span.

    ``D_max / D_min = S / (k * c)`` — guide §2.2.  The pixel size cancels, so
    this depends on **neither the sensor's resolution nor the baseline**: it is
    a pure ratio of search range to subpixel precision.

    Realistic values sit between 30 and 100 — under two orders of magnitude,
    which is why a single setting cannot serve a wide archive.

    :param search_px: Search half-width ``S``, pixels.
    :param c_px: Subpixel matching precision, pixels.
    :param k: Confidence multiplier.
    :returns: Dimensionless dynamic range.
    """
    denominator = float(k) * float(c_px)
    if denominator <= 0:
        return math.inf
    return float(search_px) / denominator


def n_parameter_sets(
    dt_days_min: float,
    dt_days_max: float,
    search_px: float,
    c_px: float,
    k: float = 3.0,
    *,
    velocity_ratio: float = 1.0,
) -> int:
    """Number of distinct parameter sets an archive needs.

    ``ceil( ln(span) / ln(dynamic_range) )`` where *span* is the ratio of
    largest to smallest expected displacement (guide §2.3).  Since velocity is
    largely a site property while baseline is a pairing choice, the span across
    one landform is essentially the span in Δt — pass *velocity_ratio* to widen
    it when a site has genuinely distinct fast and slow domains.

    :param dt_days_min: Shortest temporal baseline in the archive, days.
    :param dt_days_max: Longest temporal baseline, days.
    :param search_px: Search half-width, pixels.
    :param c_px: Subpixel matching precision, pixels.
    :param k: Confidence multiplier.
    :param velocity_ratio: Ratio of fastest to slowest velocity of interest.
    :returns: Number of parameter sets, at least 1.
    """
    if dt_days_min <= 0 or dt_days_max <= 0:
        raise ValueError("temporal baselines must be positive")
    span = (float(dt_days_max) / float(dt_days_min)) * max(float(velocity_ratio), 1.0)
    window = dynamic_range(search_px, c_px, k)
    if not np.isfinite(window) or window <= 1.0 or span <= 1.0:
        return 1
    return max(1, int(math.ceil(math.log(span) / math.log(window))))


# ---------------------------------------------------------------------------- #
# Decorrelation (guide section 5)
# ---------------------------------------------------------------------------- #
def coherence(
    dt_days: float | NDArrayNum,
    tau_days: float,
    *,
    rho_0: float = 1.0,
    seasonal_amplitude: float = 0.0,
) -> float | NDArrayNum:
    """Modelled surface coherence at baseline *dt_days*.

    ``rho = rho_0 * exp(-dt/tau) * (1 - A sin^2(pi dt / T_year))`` — guide §5.1.
    The seasonal factor is what makes an **anniversary pair** (Δt = 1 yr, same
    sun geometry, same snow state) routinely correlate better than a 6-month
    pair despite being twice as long.

    :param dt_days: Temporal baseline(s), days.
    :param tau_days: Decorrelation time, days.
    :param rho_0: Coherence at zero baseline.
    :param seasonal_amplitude: ``A`` in [0, 1]; 0 disables seasonality.
    :returns: Coherence in [0, rho_0].
    """
    if tau_days <= 0:
        raise ValueError(f"tau_days must be positive, got {tau_days}")
    dt = np.asarray(dt_days, dtype="float64")
    decay = float(rho_0) * np.exp(-dt / float(tau_days))
    if seasonal_amplitude:
        season = 1.0 - float(seasonal_amplitude) * np.sin(np.pi * dt / DAYS_PER_YEAR) ** 2
        decay = decay * season
    out = np.clip(decay, 0.0, None)
    return out if np.ndim(dt_days) else float(out)


def optimal_baseline_days(tau_days: float) -> float:
    """Baseline maximising SNR under exponential decorrelation.

    If displacement noise grows as ``sigma(dt) = sigma_0 exp(dt/tau)``, then
    ``SNR = v dt / sigma(dt)`` peaks where ``d/dt [dt exp(-dt/tau)] = 0``, i.e.
    at ``dt = tau``: **the optimal temporal baseline is one decorrelation
    time** (guide §5.2).

    A heuristic that says where to look, not a law — it assumes that specific
    noise-growth model.  But the shape is right: SNR rises linearly, then falls
    exponentially, and the turnover is at tau.

    :param tau_days: Decorrelation time, days.
    :returns: Optimal baseline, days.
    """
    if tau_days <= 0:
        raise ValueError(f"tau_days must be positive, got {tau_days}")
    return float(tau_days)


# ---------------------------------------------------------------------------- #
# Cost
# ---------------------------------------------------------------------------- #
def cost_index(
    kernel_px: int | tuple[int, int],
    corr_search: Sequence[int] | float | None,
    corr_algorithm: str = "asp_bm",
) -> float:
    """Relative correlation cost, 1.0 being ASP defaults.

    Block-matching cost scales as ``K^2 * S_x * S_y`` per pixel (guide §6.3),
    which carries the warning that matters: **doubling the search range
    quadruples the cost**, and doubling the kernel quadruples it again.

    SGM/MGM are charged at their forced 9x9 kernel plus a constant factor for
    the global optimisation, so the number stays comparable across algorithms.

    :param kernel_px: Kernel side, or ``(h, w)``.
    :param corr_search: ``(x_min, y_min, x_max, y_max)``, a scalar half-width,
        or None for ASP auto-detection (charged at the reference width).
    :param corr_algorithm: ASP ``stereo-algorithm`` value.
    :returns: Dimensionless cost relative to ``corr-kernel 21 21`` with a 5 px
        search half-width.
    """
    if corr_algorithm in SGM_ALGORITHMS:
        k_h, k_w = SGM_KERNEL
        sgm_factor = 4.0  # global optimisation over 8 paths, amortised
    else:
        if isinstance(kernel_px, (tuple, list)):
            k_h, k_w = int(kernel_px[0]), int(kernel_px[1])
        else:
            k_h = k_w = int(kernel_px)
        sgm_factor = 1.0

    if corr_search is None:
        s_x = s_y = _COST_REF_SEARCH_PX
    elif isinstance(corr_search, (tuple, list, np.ndarray)):
        x_lo, y_lo, x_hi, y_hi = (float(v) for v in corr_search)
        s_x = max(x_hi - x_lo, 1.0) / 2.0
        s_y = max(y_hi - y_lo, 1.0) / 2.0
    else:
        s_x = s_y = float(corr_search)

    kernel_term = (k_h * k_w) / (_COST_REF_KERNEL_PX**2)
    search_term = (s_x * s_y) / (_COST_REF_SEARCH_PX**2)
    return float(kernel_term * search_term * sgm_factor)


# ---------------------------------------------------------------------------- #
# The grouping axis
# ---------------------------------------------------------------------------- #
def geometric_dt_bins(
    dt_days: Iterable[float],
    ratio: float = 3.0,
) -> tuple[list[float], list[str]]:
    """Geometric temporal-baseline bin edges covering *dt_days*.

    Each bin spans a factor of *ratio* in Δt.  This is the natural grouping axis
    (guide §2.3): since ``v`` is largely a site property, the spread in
    ``v * dt`` across a landform is essentially the spread in Δt, and a
    parameter set spans a fixed *multiplicative* window.

    Edges are anchored on the shortest baseline present, so the binning is a
    pure function of the data and of *ratio* — no hidden global grid.

    :param dt_days: Temporal baselines present in the archive, days.
    :param ratio: Multiplicative width of each bin; must exceed 1.
    :returns: ``(edges, labels)`` with ``len(edges) == len(labels) + 1``.
    """
    if ratio <= 1.0:
        raise ValueError(f"ratio must exceed 1, got {ratio}")
    values = np.asarray([float(v) for v in dt_days], dtype="float64")
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return [], []

    lo = float(values.min())
    hi = float(values.max())
    edges = [lo]
    while edges[-1] < hi:
        edges.append(edges[-1] * float(ratio))
    # A single distinct baseline still needs one bin with a strictly wider top
    # edge, or assign_dt_bin would place it outside every interval.
    if len(edges) == 1:
        edges.append(lo * float(ratio))

    labels = [_bin_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return edges, labels


def _bin_label(lo: float, hi: float) -> str:
    """Human-readable label for one Δt bin, e.g. ``"dt30-90d"``."""
    return f"dt{_round_days(lo)}-{_round_days(hi)}d"


def _round_days(value: float) -> int:
    """Round a bin edge to a whole number of days for labelling."""
    return int(round(float(value)))


def assign_dt_bin(dt_days: float, edges: Sequence[float], labels: Sequence[str]) -> str:
    """Label of the bin containing *dt_days*.

    Bins are half-open ``[lo, hi)`` except the last, which includes its upper
    edge so the longest baseline is never orphaned.

    :param dt_days: Temporal baseline, days.
    :param edges: Bin edges from :func:`geometric_dt_bins`.
    :param labels: Bin labels from :func:`geometric_dt_bins`.
    :returns: The matching label, or ``""`` when there are no bins.
    """
    if not labels:
        return ""
    value = float(dt_days)
    for i, label in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        if lo <= value < hi or (i == len(labels) - 1 and value <= hi):
            return label
    return labels[-1] if value > edges[-1] else labels[0]


# ---------------------------------------------------------------------------- #
# Design-map boundaries
# ---------------------------------------------------------------------------- #
def feasibility_bounds(
    dt_grid_days: NDArrayNum | Sequence[float],
    resolution_m: float,
    c_px: float,
    search_px: float,
    *,
    k: float = 3.0,
    coreg_px: float = 2.0,
    margin_px: float = 2.0,
) -> dict[str, NDArrayNum]:
    """The two boundary curves of the design map.

    Both are ``v ∝ 1/dt`` — parallel straight lines of slope -1 in log-log — and
    the band between them is where a pair is measurable.  Its width,
    :func:`dynamic_range`, is the same everywhere.

    :param dt_grid_days: Baselines to evaluate, days.
    :param resolution_m: Ground sample distance, m/px.
    :param c_px: Subpixel matching precision, pixels.
    :param search_px: Search half-width, pixels.
    :param k: Confidence multiplier.
    :param coreg_px: Co-registration allowance, pixels.
    :param margin_px: Safety margin, pixels.
    :returns: ``{"dt_days", "v_min", "v_max"}`` as float arrays.
    """
    dt = np.asarray(dt_grid_days, dtype="float64")
    return {
        "dt_days": dt,
        "v_min": np.asarray(min_detectable_velocity(dt, resolution_m, c_px, k), dtype="float64"),
        "v_max": np.asarray(
            max_measurable_velocity(dt, resolution_m, search_px, coreg_px=coreg_px, margin_px=margin_px),
            dtype="float64",
        ),
    }


# ---------------------------------------------------------------------------- #
# The whole derivation for one pair
# ---------------------------------------------------------------------------- #
def suggest_parameters(
    resolution_m: float,
    dt_days: float,
    velocity_m_yr: float,
    *,
    footprint_m: float = 60.0,
    c_px: float = 0.15,
    k: float = 3.0,
    coreg_px: float = 2.0,
    margin_px: float = 2.0,
    flow_azimuth_deg: float | None = None,
    strain_rate_per_yr: float | None = None,
    corr_algorithm: str = "asp_bm",
    subpixel_mode: int = 2,
    cost_budget: float = DEFAULT_COST_BUDGET,
    tau_days: float | None = None,
    explicit_search: bool = True,
    explicit_search_above_days: float = 0.0,
) -> dict[str, Any]:
    """Derive ASP correlation parameters for one pair, with diagnostics.

    Implements the workflow of guide §8: convert the physical scales to pixels
    at this pair's resolution, check both bounds and both ceilings, and report
    every violation as a **warning rather than a silent clamp** — a clamped
    parameter that quietly fails to meet its constraint is worse than one the
    user was told about.

    ``corr_seed_mode`` moves with the search box: 0 when an explicit range is
    produced (the low-resolution disparity stage has nothing left to estimate),
    1 when it is not (that stage *is* what produces ASP's own estimate). See
    guide §6.1.

    :param resolution_m: Ground sample distance of the **left** image, m/px.
    :param dt_days: Temporal baseline, days.
    :param velocity_m_yr: Expected surface velocity, m/yr.
    :param footprint_m: Desired template ground footprint, metres.
    :param c_px: Subpixel matching precision, pixels.
    :param k: Confidence multiplier for the detection floor.
    :param coreg_px: Residual co-registration allowance, pixels.
    :param margin_px: Search safety margin, pixels.
    :param flow_azimuth_deg: Flow direction (degrees clockwise from north) for
        an asymmetric search box, or None.
    :param strain_rate_per_yr: Velocity gradient magnitude (1/yr) for the strain
        ceiling, or None to skip that check.
    :param corr_algorithm: ASP ``stereo-algorithm`` value.
    :param subpixel_mode: ASP ``subpixel-mode``; 2 (Bayes EM) avoids the
        pixel-locking of mode 1.
    :param cost_budget: Relative cost above which a warning is emitted.
    :param tau_days: Decorrelation time (days) for a baseline warning, or None.
    :param explicit_search: When False, always leave ``corr_search`` to ASP.
    :param explicit_search_above_days: Derive an explicit box only for baselines
        at or beyond this many days; ``0`` means every pair. Below it, ASP's
        automatic range is kept — see the note in the body on why that is the
        safer default for short baselines.
    :returns: A dict carrying the ASP parameters (``corr_kernel``,
        ``corr_search``, ``corr_seed_mode``, ``subpixel_mode``,
        ``subpixel_kernel``, ``cost_mode``, ``corr_algorithm``) alongside the
        diagnostics that justify them (``snr``, ``cost_index``,
        ``footprint_m``, ``disp_m``, ``disp_px``, ``noise_floor_m``,
        ``search_px``, ``strain_limit_px``, ``coherence``) and a ``warnings``
        list of human-readable strings.
    """
    warnings: list[str] = []

    disp_m = expected_displacement(velocity_m_yr, dt_days)
    disp_px_value = disp_m / float(resolution_m) if resolution_m > 0 else float("nan")

    # --- Kernel: footprint, then the two ceilings -------------------------- #
    strain_limit = (
        strain_kernel_limit_px(strain_rate_per_yr, dt_days)
        if strain_rate_per_yr is not None
        else math.inf
    )
    raw_kernel = to_odd(float(footprint_m) / float(resolution_m), minimum=1)
    if raw_kernel < TEXTURE_FLOOR_PX:
        warnings.append(
            f"footprint {footprint_m:g} m is only {raw_kernel} px at {resolution_m:g} m/px — "
            f"below the {TEXTURE_FLOOR_PX} px texture floor; this sensor is too coarse for "
            f"that template size"
        )
    k_max = None if math.isinf(strain_limit) else strain_limit
    kernel = kernel_from_footprint(footprint_m, resolution_m, k_min=TEXTURE_FLOOR_PX, k_max=k_max)
    if kernel < raw_kernel and not math.isinf(strain_limit):
        warnings.append(
            f"strain ceiling caps the kernel at {strain_limit:.1f} px (requested {raw_kernel} px "
            f"for a {footprint_m:g} m footprint) — accumulated strain over {dt_days:g} d exceeds "
            f"{STRAIN_TOLERANCE_PX:g} px across the template"
        )
    if not math.isinf(strain_limit) and strain_limit < TEXTURE_FLOOR_PX:
        warnings.append(
            f"strain ceiling ({strain_limit:.1f} px) is below the texture floor "
            f"({TEXTURE_FLOOR_PX} px) — this baseline cannot be correlated in that strain "
            f"regime at any kernel size; use a shorter baseline there"
        )

    # --- Search box -------------------------------------------------------- #
    #
    # An explicit box is NOT unconditionally better than ASP's automatic range.
    # ASP derives that range from the low-resolution disparity built out of
    # interest-point matches, and that estimate is most reliable exactly where
    # displacement is small -- short baselines still look alike, so IP matching
    # succeeds and the derived box is tight and correct.  It degrades on long
    # baselines, where decorrelation leaves sparse or mismatched IPs and a few
    # blunders inflate the box.
    #
    # The failure modes are asymmetric, which is what decides the default:
    # auto failing costs RUNTIME, an explicit box failing costs CORRECTNESS,
    # because a low velocity guess clips real motion and the field saturates at
    # the box edge.  So below the threshold we keep ASP's own estimate.
    use_explicit = bool(explicit_search) and dt_days >= float(explicit_search_above_days)

    if use_explicit:
        corr_search = required_search_px(
            velocity_m_yr,
            dt_days,
            resolution_m,
            coreg_px=coreg_px,
            margin_px=margin_px,
            flow_azimuth_deg=flow_azimuth_deg,
        )
        # Nothing left for the low-resolution disparity stage to estimate.
        corr_seed_mode = 0
    else:
        corr_search = None
        # 1, not 0: that stage IS what produces ASP's estimate, and skipping it
        # while passing no box would leave the correlator with nothing to work
        # from.  These two always move together.
        corr_seed_mode = 1
    search_px = search_half_width_px(corr_search)

    # --- Detectability ----------------------------------------------------- #
    sigma = noise_floor_m(resolution_m, c_px)
    snr_value = snr(velocity_m_yr, dt_days, resolution_m, c_px)
    if snr_value < k:
        v_min = min_detectable_velocity(dt_days, resolution_m, c_px, k)
        warnings.append(
            f"SNR {snr_value:.2f} is below k={k:g} — this pair sits under the detection floor "
            f"({disp_m:.2f} m expected against a {sigma:.2f} m noise floor); it needs "
            f"v >= {v_min:.2f} m/yr at this baseline, or a longer baseline"
        )

    # --- Algorithm override ------------------------------------------------ #
    corr_kernel = (kernel, kernel)
    subpixel_kernel = (kernel, kernel)
    corr_kernel, subpixel_kernel, cost_mode, overridden = sgm_kernel_override(
        corr_algorithm, corr_kernel, subpixel_kernel, 2
    )
    if overridden:
        warnings.append(
            f"'{corr_algorithm}' forces corr-kernel {SGM_KERNEL[0]}x{SGM_KERNEL[1]} and "
            f"cost-mode {SGM_COST_MODE} — the derived {kernel}x{kernel} kernel is discarded"
        )

    # --- Cost -------------------------------------------------------------- #
    cost = cost_index(corr_kernel, corr_search, corr_algorithm)
    if cost > cost_budget:
        warnings.append(
            f"relative cost {cost:.1f} exceeds the budget of {cost_budget:g} — consider an "
            f"asymmetric search box along flow, or a smaller kernel"
        )

    # --- Decorrelation ----------------------------------------------------- #
    rho = float("nan")
    if tau_days is not None:
        rho = float(coherence(dt_days, tau_days))
        if dt_days > 2.0 * tau_days:
            warnings.append(
                f"baseline {dt_days:g} d is more than twice the decorrelation time "
                f"({tau_days:g} d); modelled coherence {rho:.2f} — expect a low valid fraction"
            )

    return {
        # ASP parameters, ready to splat into build_correlation_params
        "corr_algorithm": corr_algorithm,
        "corr_kernel": corr_kernel,
        "corr_search": corr_search,
        "corr_seed_mode": corr_seed_mode,
        "subpixel_mode": int(subpixel_mode),
        "subpixel_kernel": subpixel_kernel,
        "cost_mode": cost_mode,
        # Diagnostics
        "disp_m": disp_m,
        "disp_px": disp_px_value,
        "footprint_m": footprint_m_value(corr_kernel, resolution_m),
        "noise_floor_m": sigma,
        "snr": snr_value,
        "search_px": search_px,
        "strain_limit_px": strain_limit,
        "cost_index": cost,
        "coherence": rho,
        "warnings": warnings,
    }


def footprint_m_value(corr_kernel: tuple[int, int], resolution_m: float) -> float:
    """Ground footprint of a ``(h, w)`` kernel, using its larger side."""
    return float(max(corr_kernel)) * float(resolution_m)


# ---------------------------------------------------------------------------- #
# Uniform overrides
# ---------------------------------------------------------------------------- #
def parse_search_box(text: str | Sequence[int]) -> tuple[int, int, int, int]:
    """Parse a ``--corr-search`` box from one or four numbers.

    ``"20"`` means the symmetric box ``(-20, -20, 20, 20)``; ``"-80 -2 20 2"``
    means exactly that. Both forms matter: the asymmetric box along flow is the
    largest cost saving available (guide §6.1) — a symmetric box covering an
    80 px reach searches ~1600× the area — so a one-number-only control would
    put it out of reach.

    Commas and whitespace both separate.

    :param text: The box, as a string or an already-parsed sequence.
    :returns: ``(x_min, y_min, x_max, y_max)``.
    :raises ValueError: On anything that is not one or four integers, or a box
        whose maxima do not exceed its minima.
    """
    if isinstance(text, (tuple, list, np.ndarray)):
        parts = [str(v) for v in text]
    else:
        parts = str(text).replace(",", " ").split()

    try:
        values = [int(float(p)) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse search box '{text}'. Use one number for a symmetric "
            f"box (e.g. '20') or four for an explicit one (e.g. '-80 -2 20 2')."
        ) from exc

    if len(values) == 1:
        half = abs(values[0])
        if half == 0:
            raise ValueError("A symmetric search box must have a non-zero width.")
        return (-half, -half, half, half)
    if len(values) == 4:
        x_lo, y_lo, x_hi, y_hi = values
        if x_hi <= x_lo or y_hi <= y_lo:
            raise ValueError(
                f"Search box '{text}' is empty: maxima must exceed minima "
                f"(got x {x_lo}..{x_hi}, y {y_lo}..{y_hi})."
            )
        return (x_lo, y_lo, x_hi, y_hi)

    raise ValueError(
        f"Search box '{text}' has {len(values)} numbers; expected 1 or 4."
    )


def uniform_conflicts(
    derived: Mapping[str, Any],
    *,
    kernel_px: int | None = None,
    corr_search: Sequence[int] | None = None,
    corr_algorithm: str = "asp_bm",
    cost_budget: float = DEFAULT_COST_BUDGET,
) -> list[str]:
    """Ways a uniform override conflicts with what *derived* asked for.

    A fixed parameter set is a legitimate choice — often the right one — but
    applying it silently would reintroduce exactly the failures the per-group
    derivation exists to prevent. So the override is applied and the conflicts
    are **reported**, never clamped: the user gets the run they asked for plus a
    list of which pairs it is wrong for.

    :param derived: One :func:`suggest_parameters` result.
    :param kernel_px: The uniform kernel side, or None when not overridden.
    :param corr_search: The uniform box, or None when not overridden.
    :param corr_algorithm: ASP algorithm, for the cost estimate.
    :param cost_budget: Relative cost above which the box is called expensive.
    :returns: Human-readable conflict strings; empty when the override fits.
    """
    conflicts: list[str] = []

    if kernel_px is not None:
        limit = float(derived.get("strain_limit_px", math.inf))
        if kernel_px > limit:
            conflicts.append(
                f"uniform kernel {kernel_px} px exceeds this pair's strain ceiling "
                f"of {limit:.1f} px — the template deforms over {derived.get('disp_m', 0):.1f} m "
                f"of motion and matching will degrade along shear margins"
            )
        if kernel_px < TEXTURE_FLOOR_PX:
            conflicts.append(
                f"uniform kernel {kernel_px} px is below the {TEXTURE_FLOOR_PX} px "
                f"texture floor — matching becomes ambiguous on smooth terrain"
            )

    if corr_search is not None:
        reach = search_half_width_px(corr_search)
        needed = abs(float(derived.get("disp_px", 0.0)))
        if np.isfinite(reach) and np.isfinite(needed) and reach < needed:
            conflicts.append(
                f"uniform search reach {reach:.0f} px is short of this pair's expected "
                f"{needed:.1f} px of motion — the disparity will rail at the box edge "
                f"and read as saturation"
            )
        cost = cost_index(
            kernel_px if kernel_px is not None else derived.get("corr_kernel", (21, 21)),
            corr_search,
            corr_algorithm,
        )
        if cost > cost_budget:
            conflicts.append(
                f"uniform box costs {cost:.1f}x the ASP default — applied to every "
                f"pair, not just this one"
            )

    return conflicts


def format_dt_span(lo_days: float, hi_days: float) -> str:
    """A Δt span in the unit a reader thinks in: ``"30–90 d"``, ``"1.0–3.0 yr"``.

    **Presentation only.** The canonical bin label stays
    :func:`_bin_label`'s ``dt30-90d``, because that string becomes a plan-JSON
    group name and a figure-stem fragment — changing its format would break
    replay of every existing plan for no functional gain. Identity stays stable;
    only what a human reads changes.

    :param lo_days: Lower edge, days.
    :param hi_days: Upper edge, days.
    :returns: The span with a unit suffix.
    """
    lo, hi = float(lo_days), float(hi_days)
    if hi >= DAYS_PER_YEAR:
        scale, unit, digits = DAYS_PER_YEAR, "yr", 1
    elif hi >= 90:
        scale, unit, digits = 30.4375, "mo", 1
    else:
        scale, unit, digits = 1.0, "d", 0

    lo_s, hi_s = f"{lo / scale:.{digits}f}", f"{hi / scale:.{digits}f}"
    # A group holding one baseline should say it once, not twice.
    return f"{hi_s} {unit}" if lo_s == hi_s else f"{lo_s}–{hi_s} {unit}"
