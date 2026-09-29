#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
# Copyright (C) 2026 GeoMultiCorr developers | All rights reserved.
#
# This file is part of the GeoMultiCorr (GMC) project.
# https://github.com/rgdyn-toolbox/GeoMultiCorr
#
# config.py
# creation date: 2026-09-14.
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
"""Configuration of the TIO inversion — the single source of solver settings.

Field names follow the physics, not the Fortran flags; the mapping to the
original ``input_tio`` questions is given in each docstring line, and
:func:`input_tio_text` renders the exact text ``invers_pixel_omp`` reads on
stdin, so the Fortran and the Python backends are driven by **one** object.

Vendored from the standalone ``pytio`` port (a Python re-implementation of
``invers_pixel_brit_omp.f90``, M.-P. Doin, CNRS/ISTerre; optical variant
Bontemps et al. 2018). Two things changed on the way in:

* ``weight_mode`` defaults to ``"file"``. The Fortran reads the third column of
  ``liste_couple`` **only** when its ``iponder`` line is ``2``; GMC used to
  hard-code ``0`` (variance weighting), so every weight it ever wrote was
  ignored by the solver. :meth:`TIOConfig.legacy` reproduces that setup for
  re-running old inversions bit-for-bit.
* :func:`input_tio_text` lives here rather than in ``tio_inversion`` — it is
  the Fortran-facing serialisation of this dataclass and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

#: no-data value written in all outputs (Fortran `nodata_output_value`)
NODATA_OUT = 9999.0
#: input no-data conventions: NaN, |d| < EPS_ZERO, d < NODATA_IN
EPS_ZERO = 1e-5
NODATA_IN = -9990.0
#: weight of the rows linking cumulative increments to the smooth series
LINK_WEIGHT = 5e-4
#: two dates closer than this (years) are treated as identical (Fortran t_threshold)
T_THRESHOLD = 0.001

#: ``input_tio`` line 26 — "Weigthing by interferogram variance (y:0,n:1) ?
#: or user given weight (2)?". The only line that decides whether the
#: ``liste_couple`` weights are read at all.
IPONDER: dict[str, int] = {"variance": 0, "none": 1, "file": 2}


@dataclass
class TIOConfig:
    # --- temporal smoothing -------------------------------------------------
    #: smoothing weight gamma**2 (`gam_liss2`); < 1e-4 disables smoothing
    gamma: float = 0.003
    #: Laplacian stencil: '3pt' (iliss=0) or '5pt' (iliss=1)
    scheme: str = "5pt"
    #: row scaling by the local time step (ipondliss): 0, 1 or 2 (see guide)
    pond_liss: int = 2
    #: impose zero first derivative at the first epoch (ider_zero=0)
    first_deriv_zero: bool = False

    # --- pixel/data selection ----------------------------------------------
    #: max fraction of no-data pairs before a pixel is masked (`frac_interf`)
    frac_discard: float = 0.6
    #: min number of pairs linking each image (`irepIm`)
    min_pairs_per_image: int = 1
    #: mask pixels whose closure RMS exceeds `rms_threshold` (ifermout=0)
    mask_high_rms: bool = False
    #: threshold for the above (`seuil_ferm`)
    rms_threshold: float = 1.2

    # --- weighting ----------------------------------------------------------
    #: 'file' (iponder=2, the ``liste_couple`` weights GMC writes — default),
    #: 'variance' (iponder=0, icov=1: 1/sigma of each pair map, computed by the
    #: solver itself) or 'none' (iponder=1)
    weight_mode: str = "file"
    #: equalize the total weight received by each image (ipondimage=0)
    equalize_image_weights: bool = False

    # --- robust iterations ---------------------------------------------------
    #: iterations reweighting large residuals, w=1/(scale**2+r**2) (`ponder_rms`)
    reweight_iterations: int = 0
    #: scale of the reweighting, ~ measurement noise sigma (`scale_rms`)
    reweight_scale: float = 0.2
    #: iterations masking residuals > mask_residual_threshold (`mask_rms`)
    mask_iterations: int = 0
    #: threshold defining clearly wrong values (`thres_rms`)
    mask_residual_threshold: float = 4.0
    #: iterations adding +-2*pi to the worst residual if > 4.5 (`corr_unw`, SAR)
    unwrap_iterations: int = 0

    # --- solver ---------------------------------------------------------------
    #: relative singular-value cutoff of the SVD solve (Fortran rcond)
    rcond: float = 1e-7

    # --- optional per-pair / per-image inputs ---------------------------------
    #: constant subtracted from each pair map (referencing); default zeros
    shift: Optional[np.ndarray] = None
    #: per-image quality weights applied to the link rows (iqual=0); default ones
    image_quality: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.scheme not in ("3pt", "5pt"):
            raise ValueError("scheme must be '3pt' or '5pt'")
        if self.weight_mode not in IPONDER:
            raise ValueError("weight_mode must be 'variance', 'file' or 'none'")
        if self.pond_liss not in (0, 1, 2):
            raise ValueError("pond_liss must be 0, 1 or 2")

    @property
    def smoothing(self) -> bool:
        """Fortran `ilin`: smoothing block present."""
        return self.gamma >= 1e-4

    @property
    def n_robust_iterations(self) -> int:
        return max(self.reweight_iterations, self.mask_iterations)

    @property
    def iponder(self) -> int:
        """The ``input_tio`` line-26 value this configuration renders to."""
        return IPONDER[self.weight_mode]

    @property
    def weights_applied_by_solver(self) -> bool:
        """Whether the ``liste_couple`` weights reach the least squares."""
        return self.weight_mode == "file"

    @classmethod
    def legacy(cls, **overrides) -> "TIOConfig":
        """The configuration GMC hard-coded before the weights fix.

        ``weight_mode="variance"`` with every other default: the solver computes
        its own 1/σ per pair map and ignores ``liste_couple``'s third column.
        Use it to reproduce an inversion prepared by an earlier release —
        :func:`input_tio_text` of this config is byte-identical to the text
        those releases wrote.
        """
        return cls(weight_mode="variance", **overrides)

    def to_dict(self) -> dict:
        """JSON-able view for the run-parameters trace.

        The two optional arrays are recorded as a type tag (they are data, not
        configuration), everything else as a plain scalar.
        """
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("shift", "image_quality"):
                out[f.name] = None if value is None else f"<{type(value).__name__}>"
            else:
                out[f.name] = value
        return out


def _g(value: float) -> str:
    """Shortest round-trip text for a float (``0.6``, ``4``, ``1.2``)."""
    return f"{value:g}"


def _yes_no(flag: bool) -> int:
    """The Fortran ``(y=0;n=1)`` convention."""
    return 0 if flag else 1


def input_tio_text(
    cfg: TIOConfig | None = None,
    liste_image_inv_fn: str = "liste_image_inv",
    liste_couple_fn: str = "liste_couple",
) -> str:
    """Render the ``input_tio`` parameter block ``invers_pixel_omp`` reads on stdin.

    One line per Fortran question, in the order the program asks them — the
    file is read sequentially, so the **position** of each line is the
    contract, not its comment. Lines whose options are not ported to the
    Python kernel (median pre-filter, corner/band referencing, covariance
    whitening, baseline term, per-image quality column) keep the constant
    values GMC has always written.

    ``input_tio_text(TIOConfig.legacy())`` is byte-identical to the text
    earlier releases hard-coded; a test pins that.

    :param cfg: The solver configuration; ``None`` means :class:`TIOConfig`
        defaults.
    :param liste_image_inv_fn: Name of the image list read by the solver.
    :param liste_couple_fn: Name of the pair list (``date1 date2 weight``).
    :returns: The text, newline-terminated.
    """
    cfg = cfg or TIOConfig()
    lines = [
        f"{cfg.gamma:.4f}  %  smoothing coefficient (threshold = 0.0001)",
        f"{_yes_no(cfg.mask_high_rms)}   %   remove points with large RMS misclosure  (y=0;n=1)",
        f"{_g(cfg.rms_threshold)} %  threshold on RMS misclosure (in rad) ?",
        "1  % range and azimuth sampling ?",
        f"{cfg.unwrap_iterations} % iterations to correct unwrapping errors (y:nb_of_iterations,n:0)",
        f"{cfg.reweight_iterations} % iterations to weight pixels of interferograms with large residual? (y:nb_of_iterations,n:0)",
        f"{_g(cfg.reweight_scale)} % Scaling value for weighting residuals (in same unit as input files)",
        f"{cfg.mask_iterations} % iterations to mask (tiny weight) pixels of interferograms with large residual? (y:nb_of_iterations,n:0)",
        f"{_g(cfg.mask_residual_threshold)} % threshold on residual, defining clearly wrong values (in same unit as input files)",
        "1    %   elimination of outliers by the median ? (y=0,n=1)",
        f"{liste_image_inv_fn}",
        "0    % sort by date (0) ou by another variable (1) ?",
        f"{liste_couple_fn}",
        "1   % interferogram format (RMG : 0; R4 :1) (date1-date2_pre_inv.unw or date1-date2.r4)",
        "3100.   %  include interferograms with bperp lower than maximal baseline",
        "1 % Weight input interferograms by coherence or correlation maps ? (y:0,n:1)",
        "0 % coherence file format (RMG : 0; R4 :1) (date1-date2.cor or date1-date2-CC.r4)",
        f"{cfg.min_pairs_per_image}   %   minimal number of interferams using each image",
        f"{_yes_no(cfg.equalize_image_weights)}     % interferograms weighting so that the weight per image is the same (y=0;n=1)",
        f"{_g(cfg.frac_discard)} % maximum fraction of discarded interferograms",
        "0 %  Would you like to restrict the area of inversion ?(y=1,n=0)",
        "1 735 1500 1585  %Give four corners, lower, left, top, right in file pixel coord",
        "1  %    referencing of interferograms by bands (1) or corners (2) ?",
        "5  %     band NW-SW(1), SW-SE(2), NW-NE(3), average of three bands(4), no referencement(5) ?",
        "1   %   Weigthing by image quality (y:0,n:1) ?",
        f"{cfg.iponder}   %  Weigthing by interferogram variance (y:0,n:1) ?  or user given weight (2)?",
        "1    % use of covariance (y:0,n:1) ? (Obsolete)",
        "0   % include a baseline term in inversion ? (y:1;n:0) Requires smoothing !",
        f"{1 if cfg.scheme == '5pt' else 0}   % smoothing by Laplacian, computed with a scheme at 3pts (0) or 5pts (1) ?",
        f"{cfg.pond_liss}   % weigthed smoothing by the average time step (y:0; n:1, int:2) ?",
        f"{_yes_no(cfg.first_deriv_zero)}    % put the first derivative to zero (y:0; n:1)?",
    ]
    return "\n".join(lines) + "\n"


#: Line index (0-based) of the ``iponder`` question in :func:`input_tio_text`.
#: Pinned by a test because the Fortran reads the file positionally.
IPONDER_LINE_INDEX = 25
