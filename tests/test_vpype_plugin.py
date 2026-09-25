"""Tests for the vpype ``deoverlap`` plugin."""

from __future__ import annotations

import numpy as np
import pytest

vpype = pytest.importorskip("vpype")
vpype_cli = pytest.importorskip("vpype_cli")


def _line(x0, y0, x1, y1):
    return np.array([complex(x0, y0), complex(x1, y1)], dtype=complex)


def test_plugin_registered():
    from deoverlap.vpype_plugin import deoverlap_cmd

    assert deoverlap_cmd.name == "deoverlap"


def test_plugin_crops_near_parallel_lines():
    doc = vpype.Document()
    lc = vpype.LineCollection()
    lc.append(_line(0, 0, 10, 0))
    lc.append(_line(1, 0.05, 11, 0.05))  # 0.05 mm away, parallel
    doc.add(lc, 1)

    result = vpype_cli.execute("deoverlap -t 0.1mm --keep longest", document=doc)
    layer = result.layers[1]
    # Longest wins; the short offset line should be gone or heavily cropped.
    total = sum(float(np.sum(np.abs(np.diff(line)))) for line in layer)
    # Original combined length ≈ 20; after deoverlap should be closer to 10–12.
    assert total < 15
    assert len(layer) >= 1


def test_plugin_respects_layer_flag():
    doc = vpype.Document()
    a = vpype.LineCollection([_line(0, 0, 5, 0), _line(0, 0.05, 5, 0.05)])
    b = vpype.LineCollection([_line(0, 0, 5, 0), _line(0, 0.05, 5, 0.05)])
    doc.add(a, 1)
    doc.add(b, 2)

    result = vpype_cli.execute("deoverlap -t 0.1mm -l 1", document=doc)
    # Layer 1 cleaned; layer 2 untouched (still 2 lines).
    assert len(result.layers[2]) == 2
    assert len(result.layers[1]) <= 2
