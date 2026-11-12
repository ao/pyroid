# warc suite data (warcio 1.8.2, datatrove 0.10.1)

Only `ref-digests.json` is committed: the digests of warcio's and datatrove's traces on every input under every
configuration of the suite (`python -m pyroid_parity.warc_pins write`, then `check`). The inputs live in
`$PYROID_DATA/warc/`, pinned by sha256 in [`bench/manifest.json`](../../../bench/manifest.json) and fetched by
`bench/fetch.py --only warc` (`pyroid_parity.warc_corpus`). The synthetic inputs are generated at run time
(`pyroid_parity.warc_synth`); `ref-digests.json` pins the sha256 of the generated set.

## Common Crawl slices (`warc/cc/`, 86.9 MB)

Each file is one HTTP range of a file under `https://data.commoncrawl.org/`; the manifest entry has the URL and
range. `members`: the range starts and ends on gzip member boundaries; `tail`: it ends inside a member (a
truncated download); `head`: it starts inside one. Records as warcio reads the raw file:

| files | crawls | cut | records |
|---|---|---|---:|
| `CC-MAIN-<crawl>.warc.gz`, the first 3 MB of file 00001 of a segment | 2013-20, 2013-48, 2014-15, 2015-27, 2016-40, 2017-30, 2018-30, 2018-51, 2019-35, 2020-50, 2021-49, 2022-49, 2023-50, 2024-51, 2025-51, 2026-43 | members | 8,687 |
| `CC-MAIN-<crawl>-mid.warc.gz`, 3 MB from 40 % into the file | 2017-30, 2021-49, 2025-51 | members | 1,143 |
| `CC-MAIN-2015-27-empty.warc.gz`, a whole 431-byte file (one warcinfo record) | 2015-27 | members | 1 |
| `CC-MAIN-2013-48-tail.warc.gz`, `CC-MAIN-2022-49-tail.warc.gz`, 2 MB | 2013-48, 2022-49 | tail | 591 |
| `CC-MAIN-2024-51-head.warc.gz`, 229 KB | 2024-51 | head | 0 (ArchiveLoadFailed) |
| `CC-MAIN-<crawl>.warc.wet.gz`, 2 MB, `conversion` records | 2013-20, 2016-40, 2019-35, 2023-50, 2026-43 (+ a 1.5 MB `tail`) | members, tail | 3,643 |
| `CC-MAIN-<crawl>.warc.wat.gz`, 2 MB, JSON `metadata` records | 2013-48, 2018-30, 2022-49, 2026-43 | members | 4,228 |
| `CC-MAIN-2012.arc.gz`, 3 MB of `parse-output/segment/1346823845675/1346864469604_0.arc.gz` | the 2012 crawl (ARC) | members | 270 |

The crawls up to CC-MAIN-2016-40 (and the 2012 ARC crawl) have no `WARC-Identified-Payload-Type`, from
CC-MAIN-2017-30 on every response has one; datatrove asks libmagic about each record without one, which includes
every WET record. The 2008-2010 ARC crawls (`crawl-001`, `crawl-002`) were not found: their
file names are not listed anywhere data.commoncrawl.org serves, and anonymous S3 listing is refused. The 2012 file
name comes from the `arcFileDate` / `arcFileParition` fields of that segment's `metadata-00000`. Read through
`gzip.GzipFile`, as datatrove does, the ARC file stops after its first record with `ArchiveLoadFailed`: its
records are not followed by a newline, and in plain mode warcio always consumes the line after a record (with its
"Record not followed by newline" warning), here the next record's header line.

The pages are third-party web content under the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).

## warcio's test files (`warc/warcio-1.8.2/`, 44.7 KB)

`test/data` of https://github.com/webrecorder/warcio at the commit tagged v1.8.2 (`b9036802`), each file checked
against its git blob sha1: 16 files, 53 records (ARC, WARC, gzip, chunked, digests, a non-chunked gzip, truncated
and wget-damaged files). Apache-2.0, © Ilya Kreymer and contributors.
