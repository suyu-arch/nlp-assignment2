"""Check leakage, causal prediction and token-weighted evaluation."""
import math
from pathlib import Path
import tempfile
import unittest

import torch

from nlp_lm.data import BOS, EOS, PAD, UNK, SentenceChunks, collate_chunks, encode_sentences, prepare_data
from nlp_lm.engine import run_epoch
from nlp_lm.models import MODEL_NAMES, build_model


class ProtocolTests(unittest.TestCase):
    def test_validation_cannot_change_vocabulary(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "corpus.txt", Path(folder) / "data.pt"
            source.write_text("甲甲\n乙乙\n仅验证集\n", encoding="utf-8")
            prepare_data(source, output, val_sentences=1, min_freq=1)
            bundle = torch.load(output, weights_only=True)
            self.assertEqual(bundle["stats"]["train_sentences"], 2)
            self.assertNotIn("仅", bundle["vocab"])
            self.assertEqual(bundle["val"]["tokens"].tolist(), [BOS, UNK, UNK, UNK, UNK, EOS])

    def test_chunks_predict_each_target_exactly_once(self):
        data, _ = encode_sentences(["甲乙丙丁戊", "甲"], ["<pad>", "<unk>", "<bos>", "<eos>", *"甲乙丙丁戊"])
        chunks = SentenceChunks(data, seq_len=2)
        targets = torch.cat([y for _, y in chunks]).tolist()
        self.assertEqual(targets, [4, 5, 6, 7, 8, EOS, 4, EOS])

    def test_all_models_cannot_see_future(self):
        config = dict(embedding_dim=8, hidden_dim=16, seq_len=6, layers=1, heads=2, dropout=0.0)
        original = torch.tensor([[BOS, 4, 5, 6, 7, 8]])
        modified = torch.tensor([[BOS, 4, 5, 9, 10, 11]])
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                torch.manual_seed(42)
                model = build_model(name, 12, config).eval()
                with torch.no_grad():
                    before, after = model(original), model(modified)
                torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=1e-5)
                self.assertFalse(torch.allclose(before[:, 3:], after[:, 3:]))

    def test_ppl_weights_tokens_and_ignores_padding(self):
        class FixedLM(torch.nn.Module):
            def forward(self, tokens):
                # token 4 predicts target 1 with p=.5; token 5 with p=.25.
                probability = torch.where(tokens == 4, 0.5, 0.25)
                return torch.stack([1 - probability, probability], dim=-1).log()

        batches = [collate_chunks([(torch.tensor([4]), torch.tensor([1]))]),
                   collate_chunks([(torch.tensor([5, 5, 5]), torch.tensor([1, 1, 1])),
                                   (torch.tensor([4]), torch.tensor([PAD]))])]
        result = run_epoch(FixedLM(), batches, torch.device("cpu"))
        self.assertEqual(result["tokens"], 4)
        self.assertAlmostEqual(result["loss"], (-math.log(.5) - 3 * math.log(.25)) / 4, places=6)
        self.assertAlmostEqual(result["ppl"], math.exp(result["loss"]), places=6)


if __name__ == "__main__":
    unittest.main()
