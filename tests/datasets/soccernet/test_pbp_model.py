import pytest
import torch

from real_time_vlm_benchmark.datasets.soccernet.pbp_model import construct_sent_seq


@pytest.mark.parametrize(
    "index,seq_len",
    [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3)],
)
@pytest.mark.parametrize("emb_dim", [4, 8])
def test_construct_sent_seq(emb_dim: int, index: int, seq_len: int) -> None:
    encoded_sents = torch.rand(index + 5, emb_dim)
    sent_seq = construct_sent_seq(encoded_sents, index, seq_len)
    assert sent_seq.size() == (seq_len, emb_dim)

    if index < seq_len:
        pad_len = seq_len - index - 1
        assert torch.all(sent_seq[:pad_len] == 0)
        assert sent_seq[pad_len:].equal(encoded_sents[: index + 1])
    else:
        assert sent_seq.equal(encoded_sents[(index + 1) - seq_len : index + 1])
