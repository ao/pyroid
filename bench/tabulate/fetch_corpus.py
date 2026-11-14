#!/usr/bin/env python3
"""Fetch the public CSV files the tabulate corpus test and benchmark use, and check their sha256.

    python bench/tabulate/fetch_corpus.py [DIR]          # default $PYROID_TABULATE_CORPUS
    python bench/tabulate/fetch_corpus.py DIR --pin      # (re)write the manifest

URLs are pinned to commits; the manifest (parity/data/tabulate-corpus/manifest.json) records
each file's URL, size and sha256. The data itself is not in the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "parity" / "data" / "tabulate-corpus" / "manifest.json"

SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/71e2436a092d714350de0fc409ca8a8714e7e78f/"
VEGA = "https://raw.githubusercontent.com/vega/vega-datasets/192178cf90a83fedccbab80c1f067c29cac6a697/data/"
DATAHUB = "https://raw.githubusercontent.com/datasets/"

FILES = {
    **{f"seaborn-{n}": SEABORN + f"{n}.csv" for n in (
        "anagrams", "anscombe", "attention", "brain_networks", "car_crashes", "diamonds", "dots",
        "dowjones", "exercise", "flights", "fmri", "geyser", "glue", "healthexp", "iris", "mpg",
        "penguins", "planets", "seaice", "taxis", "tips", "titanic",
    )},
    **{f"vega-{n}": VEGA + f"{n}.csv" for n in (
        "airports", "bird-migration", "birdstrikes", "co2-concentration", "disasters",
        "flights-airport", "gapminder-health-income", "github", "global-temp", "iowa-electricity",
        "la-riots", "population_engineers_hurricanes", "seattle-weather", "sp500-2000", "sp500",
        "species", "stocks", "us-employment", "weather", "windvectors", "zipcodes",
    )},
    "datahub-airport-codes": DATAHUB + "airport-codes/0ebaf2a8ab955eb542f84b7b5960ecf563f64edc/data/airport-codes.csv",
    "datahub-gdp": DATAHUB + "gdp/36d761dc5617982d3e50b213e02b9a1b47c4e958/data/gdp.csv",
    "datahub-sp500-constituents": DATAHUB + "s-and-p-500-companies/05c998c3e714b8dc9b621239bbdef92bccefc376/data/constituents.csv",
    "datahub-country-codes": DATAHUB + "country-codes/2d84034a0cae19ea5444b78c2131304a5c7f4bb9/data/country-codes.csv",
    "datahub-covid-countries-aggregated": DATAHUB + "covid-19/bdd76cfa0d80a132898345bc21939f89b2e29137/data/countries-aggregated.csv",
}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pin = "--pin" in sys.argv
    out = Path(args[0] if args else os.environ.get("PYROID_TABULATE_CORPUS", "tabulate-corpus"))
    out.mkdir(parents=True, exist_ok=True)
    pinned = json.loads(MANIFEST.read_text())["files"] if MANIFEST.is_file() and not pin else {}
    manifest = {}
    for name, url in sorted(FILES.items()):
        path = out / f"{name}.csv"
        if not path.is_file():
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed https URLs)
                path.write_bytes(r.read())
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if pinned and pinned[name]["sha256"] != digest:
            raise SystemExit(f"{name}: sha256 {digest} != pinned {pinned[name]['sha256']}")
        manifest[name] = {"url": url, "bytes": len(data), "sha256": digest}
        print(f"{name:40} {len(data):>9} {digest[:16]}")
    if pin:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({"files": manifest}, indent=1) + "\n")


if __name__ == "__main__":
    main()
