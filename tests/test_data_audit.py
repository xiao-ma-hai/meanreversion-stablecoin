from meanrev_stablecoin.constants import (
    EXPECTED_FIRST_TIMESTAMP,
    EXPECTED_LAST_TIMESTAMP,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
)
from meanrev_stablecoin.data_io import load_kraken_ohlcvt, sha256_file


def test_raw_hash_rows_range_and_duplicates(root):
    path = root / "data/raw/USDTUSD_5.csv"
    assert sha256_file(path) == EXPECTED_SHA256
    df = load_kraken_ohlcvt(path)
    assert len(df) == EXPECTED_ROWS
    assert int(df.timestamp.iloc[0]) == EXPECTED_FIRST_TIMESTAMP
    assert int(df.timestamp.iloc[-1]) == EXPECTED_LAST_TIMESTAMP
    assert not df.timestamp.duplicated().any()
    assert (df.timestamp % 300 == 0).all()


def test_raw_file_is_not_rewritten(root):
    path = root / "data/raw/USDTUSD_5.csv"
    assert path.exists()
    assert sha256_file(path) == EXPECTED_SHA256

