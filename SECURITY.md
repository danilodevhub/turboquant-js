# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in turboquant-js, please report it responsibly.

**Do not open a public issue.** Instead, email **[security@danilodevhub.com](mailto:security@danilodevhub.com)** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You should receive an acknowledgment within **48 hours**. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

As a pure-computation library with zero dependencies, the most likely security concerns are:

- Numeric overflows or precision issues that could produce incorrect results in safety-critical applications
- Denial of service through crafted inputs causing excessive memory or CPU usage

## Disclosure Policy

- We will credit reporters in the release notes (unless anonymity is requested)
- We aim to release a fix within **7 days** of confirming a vulnerability
- Public disclosure will be coordinated with the reporter
