import numpy as np
import pandas as pd

from reorder.features import NumericScaler, decode_bitmask, gaps_matrix, parse_pg_array


def test_decode_bitmask_orders_oldest_to_newest():
    # bit 0 = most recent order, bit 2 = third most recent
    seq = decode_bitmask(np.array([0b101]), seq_len=4)
    assert seq.tolist() == [[0, 1, 0, 1]]  # oldest ... newest


def test_decode_bitmask_handles_all_20_bits():
    seq = decode_bitmask(np.array([2**20 - 1]))
    assert seq.shape == (1, 20) and seq.sum() == 20


def test_parse_pg_array_variants():
    assert parse_pg_array("{1,2.5,3}") == [1.0, 2.5, 3.0]
    assert parse_pg_array([4, 5]) == [4.0, 5.0]
    assert parse_pg_array("{}") == []


def test_gaps_matrix_left_pads():
    gaps, exists = gaps_matrix(pd.Series(["{0,7,14}"]), seq_len=5)
    assert gaps.tolist() == [[0, 0, 0, 7, 14]]
    assert exists.tolist() == [[0, 0, 1, 1, 1]]


def test_scaler_roundtrip(candidates):
    s = NumericScaler().fit(candidates)
    x1 = s.transform(candidates)
    s2 = NumericScaler.from_dict(s.to_dict())
    np.testing.assert_allclose(x1, s2.transform(candidates))
    assert np.abs(x1.mean(axis=0)).max() < 1e-4
