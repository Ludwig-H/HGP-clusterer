import numpy as np
import pytest

import hgp_clusterer.core as core


def test_hypergraphpercol_handles_missing_faces(monkeypatch):
    def fake_builder(*args, **kwargs):
        return [], [], [], [], [], 0

    monkeypatch.setattr(core, "_build_graph_KSimplexes", fake_builder)
    data = np.zeros((5, 2), dtype=float)
    labels = core.HypergraphPercol(data, K=3, complex_chosen="rips")
    assert labels.shape == (5,)
    assert np.all(labels == -1)


def test_hypergraphpercol_multi_clusters_when_no_faces(monkeypatch):
    def fake_builder(*args, **kwargs):
        return [], [], [], [], [], 0

    monkeypatch.setattr(core, "_build_graph_KSimplexes", fake_builder)
    data = np.zeros((3, 2), dtype=float)
    labels, multi = core.HypergraphPercol(
        data,
        K=4,
        complex_chosen="rips",
        return_multi_clusters=True,
    )
    assert labels.tolist() == [-1, -1, -1]
    assert multi == [[(-1, 1.0)]] * 3
