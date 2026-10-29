import pytest
import torch

from cofai.engine.bitrate import bits_from_coded_unit
from cofai.latent_codecs.dsvfc import DSVFC


@pytest.mark.parametrize("token_res", [(6, 10), (8, 12)])
def test_real_stream_roundtrip_survives_checkpoint_reload(token_res):
    """Cover odd latent widths, entropy-buffer loading, and engine bit accounting."""
    torch.manual_seed(2482)
    config = {"h_dim": 64, "y_dim": 16, "z_dim": 16, "groups": 8}
    codec = DSVFC(**config).eval()
    codec.update()
    height, width = token_res
    h = torch.randn(1, height * width, 64)
    with torch.inference_mode():
        coded = codec.compress(h, token_res)
        decoded = codec.decompress(**coded)["h_hat"]
        restored = DSVFC(**config).eval()
        restored.load_state_dict(codec.state_dict(), strict=True)
        redecoded = restored.decompress(**coded)["h_hat"]
        recoded = restored.compress(h, token_res)
    assert decoded.shape == h.shape
    assert torch.isfinite(decoded).all()
    assert torch.equal(decoded, redecoded)
    assert coded["strings"] == recoded["strings"]
    bits = bits_from_coded_unit(coded)
    assert all(value > 0 for value in bits.values())
    assert bits == {
        key: sum(len(stream[0]) for stream in streams) * 8
        for key, streams in coded["strings"].items()
    }
