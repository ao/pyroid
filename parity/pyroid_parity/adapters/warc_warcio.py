"""warcio 1.8.2 behind the ``warc`` contract: its reader classes, datatrove's read path as ``iter_documents``."""

from __future__ import annotations

import types
from typing import Any

from pyroid_parity.impl import GenericWarc


def make() -> Any:
    from warcio.archiveiterator import ArchiveIterator, ARCIterator, WARCIterator
    from warcio.digestverifyingreader import DigestChecker
    from warcio.exceptions import ArchiveLoadFailed
    from warcio.recordloader import ArcWarcRecord
    from warcio.statusandheaders import StatusAndHeaders

    from pyroid_parity.warc_datatrove import reference_documents

    ns = types.SimpleNamespace(
        __name__="warcio",
        ArchiveIterator=ArchiveIterator,
        WARCIterator=WARCIterator,
        ARCIterator=ARCIterator,
        StatusAndHeaders=StatusAndHeaders,
        ArcWarcRecord=ArcWarcRecord,
        DigestChecker=DigestChecker,
        ArchiveLoadFailed=ArchiveLoadFailed,
        WARCIO_VERSION="1.8.2",
        iter_documents=reference_documents,
    )
    obj = GenericWarc(ns)
    try:
        from datatrove.pipeline.readers import WarcReader
    except ImportError:
        obj.reader_cls = None
    else:
        obj.reader_cls = WarcReader
    obj.notes = ["reference: warcio 1.8.2; iter_documents = datatrove 0.10.1's WarcReader.read_file path"]
    return obj
