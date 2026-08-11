import numpy as np
import torch

from reorder import nn
from reorder.features import NumericScaler

VOCAB = {"n_products": 50, "n_aisles": 12, "n_departments": 6}


def test_forward_shapes(candidates):
    scaler = NumericScaler().fit(candidates)
    inp = nn.make_inputs(candidates, scaler, VOCAB)
    model = nn.ReorderNet(**VOCAB)
    num, cats, seq = inp.batch(np.arange(len(inp)))
    assert seq.shape == (len(candidates), 20, 3)
    out = model(num, cats, seq)
    assert out.shape == (len(candidates),)


def test_predict_returns_probabilities(candidates):
    scaler = NumericScaler().fit(candidates)
    p = nn.predict(nn.ReorderNet(**VOCAB), nn.make_inputs(candidates, scaler, VOCAB))
    assert p.shape == (len(candidates),) and ((p > 0) & (p < 1)).all()


def test_save_load_roundtrip(tmp_path, candidates):
    scaler = NumericScaler().fit(candidates)
    model = nn.ReorderNet(**VOCAB)
    nn.save(model, scaler, VOCAB, {"history": []}, tmp_path)
    model2, scaler2, vocab2 = nn.load(tmp_path)
    inp = nn.make_inputs(candidates, scaler2, vocab2)
    torch.testing.assert_close(torch.from_numpy(nn.predict(model, inp)),
                               torch.from_numpy(nn.predict(model2, inp)))
