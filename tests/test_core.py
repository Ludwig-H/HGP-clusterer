import numpy as np
import pytest

from hgp_clusterer import HGPClusterer
import hgp_clusterer.core as core


def test_hgp_handles_missing_faces(monkeypatch):
    def fake_builder(*args, **kwargs):
        return [], [], [], [], [], 0

    monkeypatch.setattr(core, "_build_graph_KSimplexes", fake_builder)
    data = np.zeros((5, 2), dtype=float)
    clusterer = HGPClusterer(K=3, complex_chosen="rips")
    clusterer.fit(data)
    labels = clusterer.labels_
    assert labels.shape == (5,)
    assert np.all(labels == -1)


def test_hgp_handles_missing_faces_empty_data(monkeypatch):
    def fake_builder(*args, **kwargs):
        return [], [], [], [], [], 0

    monkeypatch.setattr(core, "_build_graph_KSimplexes", fake_builder)
    data = np.zeros((3, 2), dtype=float)
    clusterer = HGPClusterer(
        K=4,
        complex_chosen="rips"
    )
    clusterer.fit(data)
    labels = clusterer.labels_
    assert labels.tolist() == [-1, -1, -1]

