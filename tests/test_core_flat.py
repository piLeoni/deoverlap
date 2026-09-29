"""Smoke tests for the flat wire format entry point (shared with Node)."""

from deoverlap import _core


def test_deoverlap_flat_matches_nested():
    nested = [[[[0.0, 0.0], [2.0, 0.0]]], [[[1.0, 0.05], [3.0, 0.05]]]]
    flat = [
        ([0.0, 0.0, 2.0, 0.0], [0, 4], b"\x00"),
        ([1.0, 0.05, 3.0, 0.05], [0, 4], b"\x00"),
    ]
    kw = dict(
        prefer="first",
        angle=90.0,
        self_overlap=False,
        min_length=0.0,
        drop=None,
        keep_duplicates=False,
        mask=[],
        mask_capsules=[],
        progress=None,
    )
    kept_n, _, _, wholly_n, _ = _core.deoverlap(nested, 0.1, **kw)
    flat_kw = {k: v for k, v in kw.items() if k != "mask_capsules"}
    kept_f, _, _, wholly_f = _core.deoverlap_flat(flat, 0.1, **flat_kw)
    assert wholly_n == wholly_f
    assert [i for i, _ in kept_n] == [i for i, _ in kept_f]
    assert kept_n[0][0] == kept_f[0][0]
    assert len(kept_f[0][1].coords) == 4
