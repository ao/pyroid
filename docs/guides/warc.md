# `pyroid.warc`: warcio's reader and datatrove's WARC loop in Rust

`pyroid.warc` reads WARC and ARC files as [warcio](https://github.com/webrecorder/warcio) 1.8.2
(Apache-2.0) does, and turns them into documents as datatrove 0.10.1's `WarcReader` does. The
reader is a statement-by-statement port of warcio's: the same records, headers, payloads,
offsets, digest checks, exceptions and messages. Gzip members are inflated with zlib-rs, and
headers are parsed and payloads decoded in Rust; a file given by path is read without the GIL.

```bash
pip install pyroid            # no dependencies
pip install 'pyroid[warc]'    # faust-cchardet and python-magic, for datatrove's decoding
```

## Change one import

```diff
-from warcio.archiveiterator import ArchiveIterator
+from pyroid.warc import ArchiveIterator

 with open("CC-MAIN-20260920-00000.warc.gz", "rb") as f:
     for record in ArchiveIterator(f):
         if record.rec_type == "response":
             url = record.rec_headers.get_header("WARC-Target-URI")
             html = record.content_stream().read()
```

`WARCIterator`, `ARCIterator`, `StatusAndHeaders`, `ArcWarcRecord`, `DigestChecker` and
`ArchiveLoadFailed` are there too, with warcio's arguments, attributes and methods
(`read_to_end`, `get_record_offset`, `get_record_length`, `check_digests=`, `arc2warc=`, ...).
`content_stream()` de-chunks and decompresses gzip, deflate and, when brotlicffi >= 1.2.1.0 is
installed (as warcio requires), `br`.

A file object is read as warcio reads it, one `fh.read(block_size)` per block, with the GIL held.
Pass a path instead and the file is opened and read in Rust with the GIL released:

```python
from pyroid.warc import ArchiveIterator

for record in ArchiveIterator("CC-MAIN-20260920-00000.warc.gz"):
    ...
```

Or keep warcio's imports and call [`pyroid.patch("warcio")`](#pyroidpatchwarcio) once.

## Documents, as datatrove reads them

`iter_documents` is the loop of datatrove's `WarcReader.read_file`: for every record of the
file, datatrove's `process_record`, and a `(record index, document)` pair for each record it
keeps.

```python
import pyroid.warc as pw

for ri, doc in pw.iter_documents("CC-MAIN-20260920-00000.warc.gz"):
    doc["id"], doc["url"], doc["date"], doc["text"]
```

It reads the file the way datatrove does. A `.gz` file goes through CPython's `gzip.GzipFile`,
which pyroid emulates in Rust, so warcio's parser sees the same bytes and a damaged file raises
the same `EOFError`, `gzip.BadGzipFile` or `zlib.error` at the same record. Then it keeps the
`response` and `conversion` records datatrove keeps (by `WARC-Identified-Payload-Type`, or
python-magic's guess when a record has none), decodes the payload as UTF-8 or else as
faust-cchardet guesses, and takes the id, URL and date from the record's headers. The documents,
exceptions, stderr writes and `warcio.recordloader` log warnings are datatrove's, at the same
document. cchardet and magic are imported at the first document, as datatrove imports them at its
first record.

A binary file object works too (`compression="gzip"` puts the `GzipFile` emulation over it). For
any other compression, open the file yourself and pass the decompressed stream.

## Arrow batches

`iter_batches` gives the same records as Arrow record batches, without pyarrow:

```python
import pyarrow as pa
import pyroid.warc as pw

table = pa.Table.from_batches([pa.record_batch(b) for b in pw.iter_batches("x.warc.gz")])
table.column_names  # ['id', 'url', 'date', 'mime_type', 'content_type', 'html', 'text']

import polars as pl

df = pl.concat(pl.DataFrame(b) for b in pw.iter_batches("x.warc.gz", batch_size=10_000))
```

A row is a record `iter_documents` looks at. `html` is the decoded payload (binary), `text`
datatrove's decoding of it, null where datatrove skips the record. `b.column("text")` gives a
[`pyroid.Column`](../concepts/inputs.md), which the other pyroid modules take as input.

## `pyroid.patch("warcio")`

With warcio installed, `pyroid.patch("warcio")` puts pyroid's iterator in warcio's own modules:
`warcio.ArchiveIterator` and `warcio.archiveiterator.ArchiveIterator`, `WARCIterator` and
`ARCIterator`. Records are then warcio's own classes (`StatusAndHeaders`, an `ArcWarcRecord`
subclass whose `content_stream()` is pyroid's), errors are warcio's `ArchiveLoadFailed`, and
warnings go to warcio's logger. datatrove's `WarcReader` imports `ArchiveIterator` from
`warcio.archiveiterator` when it opens each file, so stock datatrove reads WARC files with pyroid
after the patch:

```python
import pyroid

pyroid.patch("warcio")

from datatrove.pipeline.readers import WarcReader  # datatrove's own reader, pyroid's iterator
```

`pyroid.unpatch("warcio")` puts warcio's classes back. Code that imported `ArchiveIterator`
before the patch keeps warcio's. Without warcio, `import warcio` gives a shim with pyroid's
classes. See [`pyroid.patch()`](patch.md).

## datatrove's `WarcReader`

[`pyroid.integrations.datatrove.WarcReader`](datatrove.md) takes datatrove's arguments and runs
`iter_documents` in `read_file`, on local files and on any fsspec filesystem:

```diff
-from datatrove.pipeline.readers import WarcReader
+from pyroid.integrations.datatrove import WarcReader

 WarcReader("s3://commoncrawl/crawl-data/CC-MAIN-2026-39/segments/", glob_pattern="*/warc/*")
```

## FineWeb from WARC files

`pyroid clean IN.warc.gz... OUT --recipe fineweb-warc` runs datatrove's `examples/fineweb.py`
from WARC files to the deduplicated, PII-replaced documents: this reader, the URL filter,
Trafilatura, fastText language ID, the quality filters, MinHash and PII. See
[the command line guide](cli.md#pyroid-clean-fineweb-from-warc-files).

## Differences from warcio

The records, documents and errors are warcio's and datatrove's
([parity](../compatibility.md#warc)). One case differs: `check_digests` hashes md5, sha1,
sha224, sha256, sha384 and sha512, under the names `hashlib.new` takes for them (`sha1`,
`SHA-1`, `sha-256`, ...). Any other algorithm name in `WARC-Block-Digest` or
`WARC-Payload-Digest` counts as unknown, where hashlib (through OpenSSL) might know it and hash.

And the API, in small ways: `iter(it)` returns a native iterator (warcio returns a generator),
the iterator's arguments are read once (changing `it.no_record_parse` later has no effect), and a
`block_size` that is not an int raises `TypeError` when the iterator is made. The
[reference](../reference/warc.md) has the details.

## Speed

On AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C), CPython 3.13.18, over 8 CC-MAIN-2026-39 WARC
files (7.5 GB, 172,376 HTML documents), from `bench/fineweb_warc.py reader` on one core:

| reader | CPU s | per file |
|---|---:|---:|
| warcio 1.8.2 over `gzip.open`, datatrove's `process_record` | 132.1 | 16.5 |
| `pyroid.warc.iter_documents` | 64.5 | 8.1 |
| `pyroid.warc.iter_batches` | 85.1 | 10.6 |

All three gave the same documents. Part of the time is the same in each: cchardet's guess for
the pages that are not UTF-8, and making the Python strings.

Reading every record with `ArchiveIterator` and `content_stream().read()` is mostly inflating
gzip members, in both readers, so pyroid gains less there. Over the first 210 MB of a
CC-MAIN-2026-43 WARC file (848 MB inflated), warcio 1.8.2 took 2.37 s, 1.76 s of it in CPython's
zlib, and `pyroid.warc` took 1.28 s, 1.02 s of it in zlib-rs (1.9×; AWS EC2 c8g.16xlarge, shared,
indicative; CPython 3.12). That is close to the speedup of the inflater alone.

datatrove's FineWeb pipeline over the same files, end to end at 8 processes on the same machine
(`bench/fineweb_warc.py`; results in `bench/results/aws-c7i.4xlarge/2026-11-10-warc/`):

| run | wall s | CPU s | documents kept |
|---|---:|---:|---:|
| stock datatrove 0.10.1 | 848.3 | 6,593 | 23,503 |
| datatrove with pyroid's steps, `WarcReader` included | 89.4 | 678 | 23,505 |
| `pyroid clean --recipe fineweb-warc --workers 8` | 79.6 | 600 | 23,505 |

At one process, datatrove with pyroid's steps took 640.5 s and the bench's native chain
(`bench/fineweb_warc.py native`) 607.8 s, with the same documents.

The two documents stock datatrove did not keep are pages its 1 s extraction timeout dropped. With
those pages left out the same way, pyroid's documents equal stock datatrove's byte for byte.
