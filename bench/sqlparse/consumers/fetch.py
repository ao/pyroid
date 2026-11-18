"""Download and unpack PyPI sdists into /target/src/<name>-<version>/ (untrusted data).

    python -I fetch.py django==6.1.3 django-debug-toolbar==8.0.0 ...

The consumers' test suites come from their sdists; nothing in them is imported from here.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tarfile
import urllib.request
import zipfile

OUT = pathlib.Path("/target/src")


def fetch(spec: str) -> pathlib.Path:
    name, _, ver = spec.partition("==")
    url = f"https://pypi.org/pypi/{name}/{ver}/json" if ver else f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url) as r:
        meta = json.load(r)
    ver = meta["info"]["version"]
    dest = OUT / f"{name}-{ver}"
    if dest.exists():
        return dest
    sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    with urllib.request.urlopen(sdist["url"]) as r:
        data = r.read()
    tmp = dest.with_suffix(".part")
    tmp.mkdir(parents=True, exist_ok=True)
    if sdist["filename"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(tmp, filter="data")
    tmp.rename(dest)
    return dest


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        d = fetch(spec)
        print(spec, "->", next(p for p in d.iterdir() if p.is_dir()))
