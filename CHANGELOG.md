# Changelog

## [0.5.2] — 2026-08-31

### Added
- **`TIOInversion.save_weights_figure()` and a Save-figure button in
  `explore_weights`.** Writes to `<project>/<pzone>/figures/inversion/` (created on
  demand; project-wide with a warning on a legacy-layout project). Like the pair
  figures, the file name is a pure function of the weighting parameters — no
  timestamp — so re-running a notebook refreshes the same files rather than
  accumulating copies, and only the parameters that *affect* the chosen mode appear in
  it. Static formats go through a new matplotlib twin, never `kaleido`.
- **A run-parameters trace, written per direction** by `prepare_inversion` to
  `inverse_{EW,NS}/inverse_{EW,NS}_parameters.json`. Per-pair weights already reached
  disk in four places, but the *recipe* did not: `save_pair_weight`'s `params` field
  only ever held `{"weight_mode": …}`, so `slope`, `sharpness`, `cc_gamma` and the rest
  were lost when the kernel died — along with the filter pipeline, the NMAD threshold,
  the launch profile and which pairs went in (`liste_couple` holds dates, so two
  sensors on one date are indistinguishable). The file carries the **full unpruned**
  twelve parameters, so `write_liste_couple(**params)` reproduces the run from the file
  alone. A write failure is logged, never raised: losing a trace must not cost an
  otherwise fully prepared inversion.
- `explore_weights(interactive=False)` returns `(frame, fig)` — the weights frame and a
  built-but-never-displayed plotly figure — importing no ipywidgets and needing no
  display, so the figure is reachable from scripts and batch jobs.
- Three pure helper modules mirroring the pair-figure stack, so the interactive figure
  and its static twin draw provably the same thing: `utils/_weights_frame.py` (the
  frame contract and the shared colour/marker/mode tables), `utils/_weights_plotly.py`
  and `utils/_weights_export.py`.

### Changed
- **`explore_weights` now stashes one parameters dict**, `_last_weights_params`,
  mirroring `Session.explore_pairs_strategy`'s `_last_pairs_params`, in place of three
  loose attributes none of which was initialised. It holds exactly the twelve keys both
  writers accept, so `inv.prepare_inversion(**inv._last_weights_params)` is a legal
  splat. `_last_weights` keeps its shape and meaning — parameters are the input you
  replay, results are the output you consume — and the vectors are now computed *from*
  the dict, so the two cannot disagree.
- `explore_weights` moved off `go.FigureWidget`, which since plotly 6 is an *anywidget*
  whose front-end JS is not bundled with the VSCode Jupyter extension. It now redraws a
  plain `go.Figure` into a `widgets.Output`, the way the pairing explorer already did.
- `explore_weights(**defaults)` **validates its keyword arguments** and raises
  `TypeError` on an unknown one; `sharpnes=8` used to be silently ignored. It also
  accepts `dt_range`, which it previously could not reproduce at all.
- Recognising the explorer's own weights no longer depends on object *identity*, so
  `dict(inv._last_weights)` is no longer mislabelled `"explicit"`. The comparison is
  numpy-safe: a dict of ndarrays used to make a plain `==` raise "truth value of an
  array is ambiguous", and a provenance *label* must never abort a write.
- `_pair_quality_metrics()` is memoised. It opens one stats JSON per pair and the
  explorer called it twice per redraw — 920 file opens per slider tick at 460 pairs.
  The cache is cleared by `filter_pairs_by_nmad`, which changes the pair list.
- **`TIOInversion.post_process()` now writes direction-suffixed GeoTIFF names** —
  `TOT_<date>_EW.tif` / `TOT_<date>_NS.tif` / `TOT_<date>_magn.tif` — instead of the
  same `TOT_<date>.tif` in both `inverse_EW/` and `inverse_NS/` (and unsuffixed in
  `inverse_magn/`). The old name never collided on disk — the per-direction
  directories already kept them apart — but it lost the direction the moment a file
  was copied elsewhere or listed flat. All three components now go through one
  `_tot_tif_name()` name-builder instead of deriving the pattern independently in
  `_tot_to_geotiff` and twice more in `post_process`, and
  `InversionExtractor._discover_tif_paths` strips the suffix back off when building
  its date keys, so `EW`/`NS`/`magn` still align on the same `YYYYMMDD` string. It
  also tolerates a pre-change unsuffixed file already on disk — stripping a suffix
  that isn't there is a no-op.

### Deprecated
- `TIOInversion._last_weight_mode` and `._last_combine` are now read-only views onto
  `_last_weights_params`, each emitting a `DeprecationWarning`. **Removal in 0.7.0**,
  the same horizon as `TIOInversion.prepare()`.

## [0.5.1] — 2026-08-31

### Fixed
- **The GMC log handler pinned `sys.stdout` at import time**, so it never saw a later
  swap. Inside a `rich.live.Live` region rich replaces `sys.stdout` with a `FileProxy`
  precisely so ordinary writes render *above* the bar; the handler kept writing to the
  original stream, underneath rich's cursor bookkeeping, and smeared the progress bar —
  it looked stuck or duplicated while the work proceeded and files landed correctly.
  `LazyStdoutHandler` now resolves `sys.stdout` at emit time, which fixes **every**
  progress bar in the package and also makes `contextlib.redirect_stdout` and notebook
  output capture work on GMC logs. As a side benefit, log lines are now ANSI-decoded by
  rich, so colours survive and bracketed text such as `[MedianCentering]` is no longer
  at risk of being read as console markup.
- Progress bars now degrade usefully with no terminal. `Live.refresh()` is a no-op
  unless the console is a terminal or a Jupyter kernel, so a redirected run (`nohup`,
  OAR, SLURM) rendered *nothing* until the loop ended and then dumped one final frame.
  Such runs now get throttled one-line progress records instead.

### Changed
- **`TIOInversion.prepare()` → `TIOInversion.prepare_inversion()`**, so the verb says
  what is being prepared (`Session.prepare_pairs_correlation` is the correlation-side
  counterpart). `prepare()` remains as a deprecated forwarder emitting both a
  `DeprecationWarning` and a GMC warning; **it will be removed in 0.7.0**.
- `extract_pairs_raw_displacements`, `apply_pairs_corrections` and `prepare_inversion`
  print **one header describing the whole run** instead of 1–5 log lines per pair. A
  3-step correction pipeline over 200 pairs previously emitted up to ~1800 lines, which
  buried the progress bar. The per-pair messages are unchanged when the same helpers are
  called directly on a single pair, and all three functions accept `verbose=True` to
  restore the old behaviour. Warnings and errors are unaffected either way.
- `TIOInversion.prepare_inversion` announces its steps *before* doing them; the previous
  code logged "Creating directory tree" after the tree was already created.

### Added
- `geomulticorr.core.console.BatchProgress` — one reusable two-row progress reporter
  (item label on row 1, bar on row 2) replacing the ~20-line rich `Live`/`Progress`
  block that was copy-pasted at nine call sites. Handles the label, the advance, the
  early-`continue` branches, the non-tty fallback and the log quieting. The six
  remaining call sites move onto it in a follow-up change.
- `geomulticorr._logging.quiet()` — context manager silencing low-severity GMC records
  for the duration of a block, with `extra=ALWAYS` as an escape hatch. Ranks records via
  the new `severity()` rather than by raw level number, because `LIST=31` and
  `LAUNCH=32` sit *above* `WARNING=30` and would slip past a plain `setLevel(WARNING)`.

## [0.5.0] — 2026-08-30

### Fixed
- **Grid alignment was checked by shape alone.** `TopoCorrection`,
  `TopoRampCorrection`, `SlopeRampCorrection` and the plotting helper `_dem_on_grid`
  guarded with `dem.data.shape != raster.data.shape`. Two rasters can share a shape and
  still sit on different ground, so a DEM with a matching shape but a different origin
  or CRS was used **as-is** — no error, no warning, a wrong topographic correction.
  Most likely to bite precisely when a DEM was prepared "at the right resolution and
  extent". All four sites now compare `(shape, transform, crs)`. When the old check
  would have passed and the new one does not, a `WARNING` names both grids and states
  that the pair's earlier output should be re-run; a routine regrid logs at `INFO`.
- **Moving-area polygons were rasterized without any CRS handling.**
  `_rasterize_moving_areas` passed geometries straight to
  `rasterio.features.geometry_mask`, which reads coordinates in the transform's
  coordinate space. Polygons in another CRS landed outside the grid, so the moving mask
  came back all-`False`, inverted to an all-`True` *stable* mask, and **the corrections
  were fitted on moving terrain — silently**. Every notebook used this path
  (`StableAreaMask(<path>)`); only the `gu.Vector` branch was safe. The target CRS is
  now threaded through `_resolve_stable_mask` and `StableAreaMask`, and a CRS-less
  GeoDataFrame warns instead of being assumed correct.
- `Session.get_dems()` called `pz.get_dem()` twice per pzone — once in the filter and
  once in the body — reading every DEM off disk twice with `load_data=True`.
- `TopoCorrection`, `TopoRampCorrection` and `SlopeRampCorrection` now declare
  `_REQUIRED_KWARGS = ("dem",)`. No correction class declared it, so
  `apply_pairs_corrections`' upfront validation never caught a missing `dem=`; the run
  failed deep inside the per-pair loop after wasted I/O.

### Features
- **Persisted reference DEM.** `Session.build_reference_dem()` crops a DEM to the pzone
  AOI, regrids it onto the reference grid, and writes it to the path `Pzone.get_dem()`
  already read — connecting `sieve_bulk(image_type="dem")`, `Pzone.get_dem()` and
  `DEMFinder.download_dem()`, which all existed but were wired to nothing. Once stored,
  the DEM shares the pair grid and the corrections stop re-warping it (previously twice
  per pair, plus once more for the control figure).
- **Moving-area polygon import.** `Session.load_moving_areas()` reprojects outlines to
  the session CRS, trims them to the reference grid, and stores them at
  `<pzone>/vector/<pz>_moving-areas.gpkg`; `get_moving_areas()` returns a `gu.Vector`
  ready for `StableAreaMask`. Polygons live with the project instead of at an absolute
  path in a notebook.
- New `geomulticorr/utils/_grid.py`: `grid_key`, `grids_match`, `describe_grid`,
  `regrid_to_ref`, `write_binary_raster` — no GMC imports, so `corrections` and `core`
  both use it. The 1-bit writer is now shared by the reference grid and the per-pair
  masks rather than duplicated.

### Changed
- `<pzone>/reference_raster/` added to the new layout (`PZ_KIND_REFERENCE_RASTER`,
  7 subfolders). Legacy-layout projects collapse it to the pzone root, as they already
  do for `reference_dem`/`vector`/`inversion`.
- `insert_pzone_from_file` now shares its input normalisation with the new loaders via
  `Session._normalize_vector_source()`. A CRS-less source is declared as the session CRS
  with a warning instead of failing in `reproject`.

## [0.4.2] — 2026-08-14

### Features
- **Per-pair stable-ground masks.** The boolean keep-mask that `FilterPipeline.compute()`
  produces was used to fit the corrections and then discarded, leaving no record of which
  pixels a correction was fitted on. `apply_pairs_corrections()` now writes it per pair, per
  component, into the pair folder as `{pa_key}-F_EWmask.tif` / `{pa_key}-F_NSmask.tif`
  (new `Pair.save_correction_masks()`; paths on `pair.pa_ew_mask_path` /
  `pair.pa_ns_mask_path`, also returned as `"ew_mask"` / `"ns_mask"` in the results dict).
  Written only when a `filter_pipeline` is supplied; the `overwrite=False` skip guard now
  treats a corrected-but-maskless pair as needing reprocessing.

  Masks are binary, so they are stored as **1-bit GeoTIFFs** (`NBITS=1` + DEFLATE): a
  1500×1500 mask is ~14 kB instead of the megabytes a float32 write would produce. GDAL,
  rasterio and QGIS expand `NBITS=1` transparently, so consumers see an ordinary uint8 0/1
  raster with no nodata — `0` means "not kept", not "no data".

### Changed
- **BREAKING (new-layout projects): the pzone-level `masks/` folder is removed.** It never
  held per-pair data. `PZ_KIND_MASKS` is gone and `NEW_LAYOUT_PZ_SUBDIRS` is now 6 entries;
  `pz_dir(pz, "masks")` raises `ValueError` so stale call sites fail loudly. The pzone-level
  moving-areas raster/vector pair (`<pz>_moving-areas_round-0.{tif,gpkg}`) now lives in
  `vector/`, and `migrate_to_new_structure()` moves it there. **Legacy-layout projects are
  unaffected** — `pz_dir(name, "vector")` already collapsed to the pzone root, which is
  exactly where those files sat. Existing new-layout projects with a populated `masks/`
  folder should move its contents into `vector/`.

## [0.4.1] — 2026-08-14

### Performance
- **Light control figures (~6× faster per pair).** `apply_pairs_corrections` spent ~83 % of its
  wall time in matplotlib, drawing 10+ Mpx rasters into panels that render under 1000 px. Every
  public `plot_*` now decimates for display first (`_preview` / `_preview_mask`, `preview_px`
  default 1400 px on the long axis), the hexbin grid drops from 1000 to 200 bins, and the
  redundant `bbox_inches="tight"` second render pass is gone. Measured on 3708×3158 PlanetScope
  output: the three figures written per pair go from **39.6 s to 6.5 s**, with smaller files and
  no visible difference. New `plot_level="light"|"full"` parameter on
  `Session.apply_pairs_corrections` (`PLOT_LEVELS` in `session.py`); `"full"` restores native
  resolution, 1000-bin hexbin and 150 dpi. Statistics written to the pair stats JSON are
  unchanged — only the numbers annotated inside the figures come from the decimated sample.

## [0.4.0] — 2026-08-11

### Features
- **Folder structure refactor (Option C: new-default, backward-compatible)** — New projects use a flatter, Pzone-centric layout (`<pzone>/{optical,image_correlation,reference_dem,masks,vector,inversion,figures}/`) instead of the nested `raster_data_<project>/<pzone>/` wrapper. Single unified path resolver `Session.pz_dir(pz_name, kind)` handles both layouts transparently. Legacy projects (v0.3.x and earlier) continue working unchanged, with opt-in migration via `Session.migrate_to_new_structure()`.
- **Per-pzone figures** — For new-layout projects, `save_pairs_figure()` routes single-pzone frames to `<pzone>/figures/<subdir>/` (fallback to project-wide `figures/` for multi-pzone). Explicit `pz_name` parameter overrides automatic routing.
- **Migration helper** — `Session.migrate_to_new_structure(dry_run=True/False)` consolidates legacy projects to the new layout: moves folders, rewrites geodatabase path columns (th_path, pa_path, pa_disparity_f_path), backs up the geodatabase, and updates the session state.
- **Fixed `copy_geodb()` bug** — Was using `shutil.copytree()` (requires directory) on a `.gpkg` file (single file); now uses `shutil.copy2()`.

### Other
- bump: 0.3.0 → 0.4.0

## [0.3.0] — 2026-07-28

### Features
- pair figure export to html/png/jpg/pdf/svg with deterministic names. Self-contained interactive html via plotly plus publication-quality raster and vector output via the matplotlib twins — no kaleido dependency. File names are a pure function of the parameters ({pzone}_{view}_{strategy}[_maxstep{n}] [_dt{min}-{max}]), so re-running a sweep refreshes files instead of accumulating copies. (6d54850)
- multi-view pairing-strategy explorer (chord, network, dt histogram). Plot-kind dropdown with four views over one cached candidate frame, so switching view never re-pairs. Chord places dates proportionally to time with a colour ring, year ticks and optional arrowheads; network mirrors backward pairs below the timeline. Every repeated element is batched into one artist: ~1040 ms -> ~50 ms per redraw at 460 pairs. Also replaces the deprecated matplotlib cm.get_cmap, removed in 3.11. interactive=False returns (frame, fig) for scripts and batch jobs: no ipywidgets import, no display required. BREAKING CHANGE: plot_pairs_chord keeps its signature but produces a different figure (time-proportional angles, colour ring, no colorbar — code reading fig.axes[1] will break). plot_pairs_network mirrors backward pairs by default; pass mirror_direction=False for the previous layout. figure_network's color_by default moved from "direction" to "dt". (fc5883e)
- shared pair-layout geometry and canonical pairs-frame contract. Pure-numpy layout maths (decimal year, circular time scale, Bezier sampling, chord/arc polylines, colour-ring mesh, arrowhead tangents, signed arc heights) plus a single pairs-frame contract built either from candidate index pairs or from committed pairs, so a preview and a commit render identically. Consolidates three duplicated decimal-year implementations. (d3b1152)

### Other
- bump: 0.2.7 → 0.3.0 (20e3d50)

## [0.2.7] — 2026-07-27

### Improvements
- separate module for print control tables (a9b1d1a)

## [0.2.6] — 2026-07-27

### Features
- separate module for print control tables (53b3959)

## [0.2.5] — 2026-07-15

### Features
- `explore_pairs_strategy()` interactively with Plottly; update: session.py and pzone.py for pair indexing and update interactively (8639c1c)

## [0.2.4] — 2026-07-09

### Improvements
- separate control plots for each correction method; update: session.py for saving control plots (6b2fe66)

## [0.2.3] — 2026-07-09

### Features
- combine weights using spatial method (NMAD and CC); UnitTest for weights method (2212b56)

## [0.2.2] — 2026-07-08

### Features
- UnitTest for `Destriping()` methods and `FitFourierStripProfile()` (61051b5)

## [0.2.0] — 2026-07-06

### Features
- major update weighting calculation (6 methods) and unit test (1e8277c)
- `save_pair_weight()` on unit test (bd7b721)
- major update weighting calculation (6 methods); sync weights and inversion stats with GPKG database; `explore_weights()` interactively function on Geojson and GPKG database; update: in function documentation (f0765a7)
- `save_pair_weight()` function on Geojson and GPKG database (f279a58)
- `save_pair_weight()` on init file (4afada9)
- sync_pairs_weight() Sync important statistics in GPKG database (7a97343)

### Improvements
- ipywidgets, ploty for interactive data analysis (6096436)

## [0.1.2] — 2026-07-02

### Features
- unit test for stats.py module. Using `Mock` for generate synthetic data until we get the examples. (21d91c8)
- Enriching default statistics on CC map. Estimation of general CC quality score based on fixed thresholds (0.25; 0.5; 0.75) and the proportion of valid pixels. add functions: `fraction_above() and count_above()`. Function helper in black format (47a9bc2)

### Bug Fixes
- homogenize template geodatabase and `pair.to_pdserie()` function for better sync (31ee1f9)

### Improvements
- small modifications on raster statistics keys after `save_corrected_stats()`. add new key final_corrected_stats() after corrections. Even it is redundant, it allows to isolate final stats; add: `sync_pairs_stats()`, `sync_raw_stats()` and `sync_corrected_stats()`; add: sync_geodb argument to `extract_pairs_raw_displacements()`, `apply_pairs_corrections()`, for syncing to GPKG database (65169be)

## [0.1.1] — 2026-06-30

### Improvements
- `README` with new `DirectionalBiasCorrection()` method (052851a)

## [0.1.0] — 2026-06-30

### Features
- tutorial notebooks for VieuxMarinet (157cff9)
- and  for already cropped images. (a4b2435)
- plot helpres to DirectionalBiasCorrection filter. (b43492d)
- DirectionalBiasCorrection filter with directional-profile fit. Reduce orthorectification bands on displacement rasters. (f651a9e)
- corr-search example in function doc. (2af00d1)
- auxiliary files with excluded file list. (c8444b9)
- exclude_patterns option in . usefull for PlanetScope images; add:  to update pairs status in database; add: summary_tables for printing in console; add: fancy progress bar with richtext library (4e390b1)
- base notebooks for developments (ef636eb)
- inversion extractor class (3256285)
- downsample and target resoludion at extract_raw_displacements() (7ebebb7)
- sync_pairs_after_cluster method; fix: avoid clean directory in cluster mode; fix: oarsub runs always with full path (9e75090)
- base full GMC project (f7c93ee)
- temporary logo folder (bf7a1bc)
- SPOT 6/7 data download exemples (439739a)
- .env Environment variables (b96eb65)
- tot to tif; sigmode weight function; write_launch_script; post_proces() (3964ba1)
- function documentation; fix: functions in agreement with geoutils (1ce6aba)
- pre, proc modules for launching TIO_inversion. (79f6264)
- GMC local ressources; tempate database, QGIS project (b101744)
- base folder for motion analysis (e5c9816)
- save_corrected_stats(), compute_array_stats() (187fa5c)
- hpc headers based on local available ressources (637b8c3)
- plot functions (a1c67ad)
- import utils.py functions (69b0d7f)
- init module for tests/utils (1bc1f21)
- init module for tests/stats (f239aea)
- init module for tests/inversion (59282c2)
- init module for tests/fixtures (4389b01)
- conftest.py rules for core module. (119a4ab)
- init module for tests/core/ (b305b38)
- test_masks.py with synthetic data (defecd5)
- test_fit.py with synthetic data (731e9b1)
- test_corrections.py with synthethic data (2c76dfd)
- conftest.py rules with synthethic data (a2c701b)
- init module for test/corrections (9696c7a)
- init module for importing (084d0b8)
- mask.py add mask class for filtering values (5fd2be3)
- fit.py mofule for pure mathemathical calculations. (c1f8d34)
- corrections clases and module. (012f3a1)
- functions help; fix: black text formatter. (1213dd7)
- caplog_gmc fixture for cleaner tests; function help using reST style; black formatter (76ed280)
- _logging.py small edits; fix:init.py import modules (ab1a665)
- extract_pairs_raw_displacements();fix:sync pa_status launch_pairs_correlation (e9273e5)
- license statement AGPLv3; fix:old asp structure extract_raw_displacements() (46b8188)
- example notebooks (e1a8e4c)
- plot_show_raw_results; histogram; rounded_limits (a21cf3b)
- license statement AGPL3 (c22cce1)
- automatic version handling (62a1f99)
- python wrappers for image correlation using ASP (85aa89d)

### Bug Fixes
- add GMC logo and docs/_static structure (d3f344e)
- out_dtype ad out_nodata in geoutils. Change gu.save to gu.to_file in agreement with the latest version of GeoUtils. (51e73de)
- out and error log files (hpc) in correct pair directory (e0525ff)
- fix canonical grid for inversion in mixed sensors example (fd6dd05)
- differences in grid size in TopoCorrection.fit(), TopoRampCOrrection.fit() and SlopeRampCorrection.fit(). Force to reproject for match grid size (3bc046e)
- full path on binary files (77d5601)
- updated micromamba env (96cb3b1)
- fix missing name from PlanetScope images (9493679)
- keep correlation parameters and bash job (d858a5c)
- apply_pair_corrections() pipeline and function help (6530d6c)
- fix EW, NS, NCC paths (5cb8e77)
- import missing functions (318e464)
- add geoutils0.2.5; pytest libs; python formatter; gdal libs; jupyterlab (fc2b86d)
- modif in general conftest.py on gmc._logging (2abdd07)
- pa_cc_raw_path in compute_pair_raw_stats (6ed7ef2)
- colors on prints; help on inner methods (dee3e4e)
- add corr_eval within same processing script; fix:symlink to cropped images (4c6dedb)
- fix automatic version handling, path to __init__.py (fc8c009)
- typo error on __init__.py (e0ed10f)
- remove - from hamonize sensor func (946c75b)

### Improvements
- ignore tests/__pycache__ (d252989)
- refresh GMC logo assets (light/dark variants) (905d168)
- small update in README.md (d4ff4a7)
- GMC logger icons using ANSI outputs (070877b)
- GMC logger icons using ANSI outputs (584d03c)
- readme (71b3f47)
- rich text library (a1625e0)
- inversion extractor class (c3f4da0)
- unit test useless files (e6ba000)
- documentation for each function. (db2117f)
- sensor_filter in get_valid_pairs_with_strategy_overview() and get_pairs_with_strategy() (2ef0331)
- sensor_filter in pairs_with_strategy(); insert gaoraster_bank in draw_polygon_manually(); add: apply_pair_corrections() in a single function. (fef633a)
- import inversion modules (42d7da0)

### Testing
- add correlation module integration and unit tests (bcf3e0e)
- add correlation module integration and unit tests; test_integration.py: ASP correlation workflow tests (8becccb)
- add correlation module integration and unit tests; test_asp.py: ASP class instantiation tests (44bd3d3)
- add correlation module integration and unit tests; test_command_building.py: ASP command building tests (e6eb1f2)
- add correlation module integration and unit tests; test_parameters.py: Correlation parameters tests (9c05cc4)
- add correlation module integration and unit tests; conftest.py: Shared fixtures for correlation tests (d75c2e9)

### Other
- Add/update notebooks (bbfa162)
- first test in _logging.py (63e284f)
- small deletes (9fbe963)
- Remove correlation.py and supported_sensors.py (e8f4b8e)
- set up version 0.1.0 o __init__.py (859c7e6)
- move notebooks to top-level notebooks/ directory (ccc6cbb)
- restructure into subpackages (core, correlation, stats, inversion, gee, utils) (f0d0850)
- remove src/ layout, old tests, and stale tracked files (b1c94a5)
- add utilities for normalizing satellite sensors (f1f0282)
- add logging module for fancy prints (46e077e)
- add generic typping aliases for internal use (f42606d)
- Update licence AGPL-3.0 (acc08ec)
- Replace bash installer with Python implementation (install_ASP.py) (54ae771)
- Add tool config to pyproject.toml (199bcd1)
- Add tool config to pyproject.toml (380291d)
- Add zenodo badge on README (12189f5)

