import numpy as np

from meanrev_stablecoin.estimation.bootstrap import _select_continuous_path_design


def test_path_design_never_bridges_observed_gaps():
    timestamps = np.array([0, 300, 600, 1500, 1800, 2100], dtype=np.int64)
    values = timestamps.astype(float)
    segments = _select_continuous_path_design(values, timestamps, maximum_pairs=4)
    for segment in segments:
        assert np.all(np.diff(segment) == 300)
    assert sum(len(segment) - 1 for segment in segments) <= 4
