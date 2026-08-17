import numpy as np
import pandas as pd

from meanrev_stablecoin.pairs import build_exact_horizon_pairs


def test_exact_time_matching_skips_gap():
    df = pd.DataFrame({"timestamp": [0, 300, 900, 1200], "log_price": [1.0, 2.0, 3.0, 4.0]})
    i, j = build_exact_horizon_pairs(df, 300)
    assert np.array_equal(i, [0, 2])
    assert np.array_equal(j, [1, 3])


def test_multistep_matching_is_timestamp_based():
    df = pd.DataFrame({"timestamp": [0, 300, 900, 1200], "log_price": [1.0, 2.0, 3.0, 4.0]})
    i, j = build_exact_horizon_pairs(df, 900)
    assert np.array_equal(i, [0, 1])
    assert np.array_equal(j, [2, 3])

