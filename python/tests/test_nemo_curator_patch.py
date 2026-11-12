"""pyroid.patch("nemo_curator"): pyroid's stages on NeMo Curator's modules, and NeMo's back after.

Patching is process-wide state, so each check runs in a fresh interpreter. Skips without NeMo
Curator (Python < 3.14 only).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("nemo_curator.stages.text.filters", reason="NeMo Curator is not installed")

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")

_CLEAN = ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS")


def _child(code: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def test_patch_swaps_and_unpatch_restores() -> None:
    proc = _child(
        """
        import warnings
        import pyroid
        import nemo_curator.stages.text.filters as F
        import nemo_curator.stages.text.filters.heuristic.string as S
        import nemo_curator.stages.text.modifiers as M
        before = (F.ScoreFilter, S.WordCountFilter, M.Modify, M.UrlRemover)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyroid.patch("nemo_curator")
        import pyroid.integrations.nemo_curator as ours
        assert [w.category.__name__ for w in caught] == ["PatchWarning"], caught
        assert F.ScoreFilter is ours.ScoreFilter and S.WordCountFilter is ours.WordCountFilter
        assert M.Modify is ours.Modify and M.UrlRemover is ours.UrlRemover
        # hydra resolves _target_ paths with getattr on the imported module: pyroid's classes.
        from hydra.utils import get_class
        for path, cls in [
            ("nemo_curator.stages.text.filters.score_filter.ScoreFilter", ours.ScoreFilter),
            ("nemo_curator.stages.text.filters.heuristic.repetition.RepeatingTopNGramsFilter",
             ours.RepeatingTopNGramsFilter),
            ("nemo_curator.stages.text.filters.fasttext.fasttext_filters.FastTextLangId",
             ours.FastTextLangId),
            ("nemo_curator.stages.text.modifiers.unicode.unicode_reformatter.UnicodeReformatter",
             ours.UnicodeReformatter),
            ("nemo_curator.stages.text.modifiers.string.url_remover.UrlRemover", ours.UrlRemover),
        ]:
            assert get_class(path) is cls, path
        # pyroid's own module still holds NeMo's classes as bases and plans.
        assert ours.ScoreFilter.__mro__[1] is before[0]
        import pyroid.integrations
        assert pyroid.integrations.patched() == ("nemo_curator",)
        pyroid.unpatch("nemo_curator")
        assert (F.ScoreFilter, S.WordCountFilter, M.Modify, M.UrlRemover) == before
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_patched_yaml_pipeline_stage_runs_pyroid() -> None:
    proc = _child(
        """
        import warnings
        warnings.simplefilter("ignore")
        import pyarrow as pa, pyroid
        pyroid.patch("nemo_curator")
        import hydra.utils
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "_target_": "nemo_curator.stages.text.filters.score_filter.ScoreFilter",
            "filter_obj": {
                "_target_": "nemo_curator.stages.text.filters.heuristic.string.WordCountFilter",
                "min_words": 2,
            },
            "score_field": "wc",
        })
        stage = hydra.utils.instantiate(cfg)
        import pyroid.integrations.nemo_curator as ours
        assert type(stage) is ours.ScoreFilter
        assert type(stage.filter_obj[0]) is ours.WordCountFilter
        from nemo_curator.tasks import DocumentBatch
        batch = DocumentBatch(dataset_name="t", data=pa.table({"text": ["a b c", "d"]}))
        out = stage.process(batch)
        print(out.to_pandas()["wc"].tolist())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[3]"
