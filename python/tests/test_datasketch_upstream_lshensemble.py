# Ported from datasketch 2.0.0's test suite (MIT, © 2015 ekzhu): test/test_lshensemble.py,
# run against pyroid's classes. The Redis cases go through pyroid's delegation to datasketch
# and need redis and mockredis. Bodies stay as upstream wrote them.
# ruff: noqa: PT009, UP031
import pickle
import unittest
import warnings
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from pyroid.datasketch.lshensemble import MinHashLSHEnsemble
from pyroid.datasketch.minhash import MinHash


def fake_redis(**kwargs: Any) -> Any:
    mockredis = pytest.importorskip("mockredis")
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


class TestMinHashLSHEnsemble(unittest.TestCase):
    def test_init(self) -> None:
        lsh = MinHashLSHEnsemble(threshold=0.8)
        self.assertTrue(lsh.is_empty())

    def _data(self, count: int) -> Iterator[tuple[int, MinHash, int]]:
        sizes = np.random.randint(1, 100, count)
        for key, size in enumerate(sizes):
            m = MinHash()
            for i in range(size):
                m.update(("%d" % i).encode("utf8"))
            yield (key, m, int(size))

    def test_index(self) -> None:
        lsh = MinHashLSHEnsemble(threshold=0.8)
        lsh.index(self._data(64))
        self.assertFalse(lsh.is_empty())
        self.assertTrue(41 in lsh)

    def test_query(self) -> None:
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(64))
        lsh.index(data)
        for key, minhash, size in data:
            keys = lsh.query(minhash, size)
            self.assertTrue(key in keys)

    def test_pickle(self) -> None:
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(32))
        lsh.index(data)
        buf = pickle.dumps(lsh)
        lsh2 = pickle.loads(buf)
        for _key, minhash, size in data:
            keys1 = lsh.query(minhash, size)
            keys2 = lsh2.query(minhash, size)
            self.assertTrue(set(keys1) == set(keys2))

    @pytest.mark.thread_unsafe
    def test_index_redis(self) -> None:
        pytest.importorskip("redis")
        pytest.importorskip("mockredis")
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lsh = MinHashLSHEnsemble(threshold=0.8, storage_config=storage_config)
            lsh.index(self._data(64))
            self.assertFalse(lsh.is_empty())
            self.assertTrue(41 in lsh)

    @pytest.mark.thread_unsafe
    def test_query_redis(self) -> None:
        pytest.importorskip("redis")
        pytest.importorskip("mockredis")
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lsh = MinHashLSHEnsemble(threshold=0.9, storage_config=storage_config)
            data = list(self._data(64))
            lsh.index(data)
            for key, minhash, size in data:
                keys = lsh.query(minhash, size)
                self.assertTrue(key in keys)
