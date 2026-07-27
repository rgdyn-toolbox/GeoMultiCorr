# Changelog

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

