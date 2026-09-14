# Contributing to GeoMultiCorr

Thank you for your interest in contributing to GeoMultiCorr! We welcome contributions from the community. This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read our [Code of Conduct](.github/CODE_OF_CONDUCT.md) before participating. We are committed to providing a welcoming and inclusive community.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/GeoMultiCorr.git
cd GeoMultiCorr
git remote add upstream https://github.com/rgdyn-toolbox/GeoMultiCorr.git
```

### 2. Set Up Development Environment

```bash
# Create and activate conda environment
mamba env create -f gmc_env.yml
conda activate gmc_env

# Install ASP (Ames Stereo Pipeline)
bash install_ASP.sh

# Unset PROJ_LIB if you encounter CRSError issues
unset PROJ_LIB
```

See [CLAUDE.md](CLAUDE.md) for detailed environment setup and troubleshooting.

### 3. Create a Feature Branch

```bash
git checkout -b feature/description-of-change
```

Use descriptive branch names. Examples:
- `feature/add-correlation-params-explorer`
- `fix/ramp-correction-edge-case`
- `docs/improve-readme`

## Making Changes

### Code Style

- **Python style:** Follow [PEP 8](https://pep8.org/). Use `black` for formatting (installed in gmc_env).
- **Type hints:** Use type hints in new code (see `geomulticorr/_typing.py` for conventions).
- **Comments:** Minimal comments; only explain *why*, not *what*. The code structure should be self-documenting.
- **Docstrings:** Concise one-liners for public functions; longer docstrings only when behavior is non-obvious.

### Testing

Before submitting, ensure all tests pass:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/core/test_session_layout.py

# Run with coverage
pytest tests/ --cov=geomulticorr --cov-report=html
```

**Add tests for new features.** Our test baseline: **1564 tests, ~1563 passing**. See `tests/` for structure and patterns.

### Git Commits

We use **Conventional Commits** with GitSmartPipeline (`gsp`). Prefix your commits with:

- `add:` — new feature
- `fix:` — bug fix
- `update:` — enhancement to existing feature
- `remove:` — delete code/files
- `docs:` — documentation only
- `test:` — test changes only
- `refactor:` — code reorganization (no behavior change)

**Example:**

```bash
gsp commit add "Correlation parameters explorer UI" \
  --description "Interactive widget for tuning ASP correlation parameters across archive"
```

This creates a conventional commit with proper message structure. See [gsp tool](https://github.com/rgdyn-toolbox/GitSmartPipeline) for more details.

## Submitting Changes

### 1. Push to Your Fork

```bash
git push origin feature/description-of-change
```

### 2. Create a Pull Request

On GitHub:
1. Go to [GeoMultiCorr](https://github.com/rgdyn-toolbox/GeoMultiCorr)
2. Click **New Pull Request**
3. Select your branch
4. Fill in the PR template with:
   - **Summary:** What does this PR do?
   - **Test plan:** How should reviewers test it?
   - **Related issues:** Link to any relevant issues

### 3. Keep Your Branch Up to Date

```bash
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin feature/description-of-change
```

## Pull Request Review

All PRs must:
- ✅ Pass CI/tests (GitHub Actions)
- ✅ Have at least one approval from a maintainer
- ✅ Follow code style guidelines
- ✅ Update `CHANGELOG.md` if user-facing

Reviewers may request changes. Update your PR by pushing new commits (don't force-push unless asked).

## Updating CHANGELOG.md

For user-facing changes, add an entry under the **Unreleased** section:

```markdown
## Unreleased

### Added
- New correlation parameters explorer for interactive ASP tuning

### Fixed
- Ramp correction edge case with all-NaN stable ground
```

Use the format from existing entries. Sections: `Added`, `Changed`, `Fixed`, `Removed`, `Security`.

## Release Process

Releases are managed by maintainers using GitSmartPipeline:

```bash
gsp bump minor              # v0.6.0 → v0.7.0
gsp tag v0.7.0
gsp push --with-tags
gsp release v0.7.0 --title "v0.7.0 — Feature Name"
```

Contributors: keep version bumping in `pyproject.toml` unchanged until maintainers prepare a release.

## Documentation

- Update docstrings for API changes
- Add notebooks for new workflows (examples in `notebooks/`)
- Update `CLAUDE.md` for architectural changes
- Render markdown docs in `docs/` as needed

## Questions or Need Help?

- **Issues:** Open a GitHub issue for bugs or feature requests
- **Discussions:** Use GitHub Discussions for questions
- **Email:** [diego.cusicanqui.vg@gmail.com](mailto:diego.cusicanqui.vg@gmail.com)

## Attribution

By contributing, you agree that your contributions may be distributed under the same license as GeoMultiCorr (AGPL-3.0). Your name and contributions will be visible in git history and on GitHub.

Thank you for helping make GeoMultiCorr better! 🚀
