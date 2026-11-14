Inputs of the dateutil suite (`tests/test_dateutil.py`):

| file | what |
|---|---|
| `issue-strings.json` | 747 distinct strings reported as bugs or surprises in python-dateutil's and pandas' GitHub issue trackers, each with the issue's URL and, for 153 of them, the `parse` keywords it used |
| `corpus-manifest.json` | the real-string corpus the `corpus-*` checks read from `$PYROID_DATEUTIL_CORPUS`: 35 public sources (loghub system logs, web server logs, email and HTTP headers, CSV date columns, CLDR-generated date formats), 9,246,745 strings, with each source's download URLs, their sha256 and the sha256 of the extracted strings. The strings themselves stay out of the repository. |
