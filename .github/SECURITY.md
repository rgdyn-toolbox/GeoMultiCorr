# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in GeoMultiCorr, please **do not open a public GitHub issue**. Instead, report it confidentially by email.

### How to Report

Send a detailed email to: [diego.cusicanqui.vg@gmail.com](mailto:diego.cusicanqui.vg@gmail.com) with:

- **Title:** Security Vulnerability Report
- **Description:** Clear description of the vulnerability
- **Steps to reproduce:** How to trigger the issue
- **Impact:** Severity and potential consequences
- **Affected versions:** Which versions are affected
- **Suggested fix:** (Optional) If you have a solution in mind

### What to Expect

- **Response time:** We will acknowledge your report within 48 hours
- **Investigation:** We will investigate and assess the vulnerability
- **Timeline:** We will work with you on a disclosure timeline
- **Patch:** Once fixed, we will release a patch and credit you (unless you prefer anonymity)

### Responsible Disclosure

Please allow us reasonable time to prepare and release a fix before public disclosure. Typically:
- **Critical issues:** 7–14 days
- **High severity:** 14–30 days
- **Medium/Low:** 30+ days or with the next release

## Supported Versions

| Version | Status | Support |
|---------|--------|---------|
| v0.6.x  | Current | Security updates |
| v0.5.x  | Older  | No longer supported |
| < v0.5  | Ancient | No support |

Only the latest minor version (v0.6.x) receives security patches. Users are encouraged to upgrade to the latest version.

## Security Best Practices for Users

When using GeoMultiCorr:

1. **Keep dependencies updated:**
   ```bash
   mamba update -n gmc_env --all
   ```

2. **Use with verified data sources:** External services (Planet API, GEE, etc.) require authentication—store credentials securely (never in version control).

3. **Report suspicious behavior:** If you notice unexpected behavior, contact us before publicizing.

## Known Issues

None currently. All discovered vulnerabilities are patched before release.

## Security of External APIs

GeoMultiCorr integrates with external services:
- **Planet Labs API** — Requires API key (keep confidential)
- **Google Earth Engine** — Requires authentication via OAuth
- **Copernicus Data Hub** — Requires credentials
- **DINAMIS/Théia STAC** — Requires free account

**Always:**
- Use `.env` files (not version control) for credentials
- Never commit API keys or tokens
- Rotate credentials regularly if compromised

See `CLAUDE.md` for integration details.

## Contact

- **Security reports:** [diego.cusicanqui.vg@gmail.com](mailto:diego.cusicanqui.vg@gmail.com)
- **General questions:** Same email or GitHub Discussions
- **Repository:** [GeoMultiCorr on GitHub](https://github.com/rgdyn-toolbox/GeoMultiCorr)
