# Security policy

## Supported versions

| Version | Supported |
|---|---|
| Latest 1.x release on PyPI | Yes |
| Older 1.x releases | No. Upgrade to the latest release. |
| 0.7.x and earlier (the [`legacy`](https://github.com/ao/pyroid/tree/legacy) branch) | No. That code is frozen and gets no fixes. |

## Reporting a vulnerability

**Please don't open a public issue, discussion or pull request for a vulnerability.**

Report it privately through GitHub's private vulnerability reporting:
**<https://github.com/ao/pyroid/security/advisories/new>**
(or the **Security** tab, then **Report a vulnerability**).

Please include:
- the pyroid version, Python version (and whether it is a free-threaded build, e.g. 3.14t) and platform;
- a minimal reproducer: the input text or file and the call that triggers the problem;
- what happens (crash, wrong memory access, unexpected file or network access, ...) and what you expected.

We aim to acknowledge a report within 3 business days and to ship a fix within 30 days. We coordinate disclosure with you, publish a GitHub Security Advisory (with a CVE where one applies), and credit you unless you'd rather stay anonymous.

## Scope

pyroid is a native (Rust) extension that processes untrusted text, so we treat these as vulnerabilities:
- memory-safety bugs reachable from the Python API: segfaults, out-of-bounds reads or writes, use-after-free, other undefined behaviour;
- data races or crashes on free-threaded Python builds (3.14t, 3.15t);
- a panic or bug that aborts the Python process instead of raising an exception;
- crafted model or data files that cause memory corruption when loaded;
- unexpected file-system or network access;
- problems with the published artifacts themselves (for example, wheels whose provenance doesn't match this repository).

Out of scope:
- denial of service through deliberately huge inputs or models (resource limits are the caller's job);
- loading pickles, or anything that executes code by design, from untrusted sources;
- vulnerabilities in the unsupported 0.7.x line;
- bugs in third-party packages that pyroid is compared against (ftfy, fastText, datatrove, ...). Report those upstream.

## Verifying a release

Every release is built and published from this repository's [`release.yml`](.github/workflows/release.yml) workflow with PyPI Trusted Publishing. Each file on PyPI carries a [PEP 740](https://peps.python.org/pep-0740/) attestation, shown on the PyPI file page, and the files also have GitHub build-provenance attestations:

```bash
gh attestation verify pyroid-1.0.0-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --repo ao/pyroid
```
