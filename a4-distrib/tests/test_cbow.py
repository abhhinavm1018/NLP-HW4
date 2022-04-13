import torch
from torch import optim
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader

from .rnn_attn_rc_test_case import RNNAttnRCTestCase
from rnn_attention_rc.models.cbow import CBOW
from rnn_attention_rc.data import collate_fn, read_data, SQuADDataset


class TestCBOW(RNNAttnRCTestCase):
    def test_forward(self):
        lr = 0.5
        batch_size = 16
        embedding_dim = 50

        # Read SQuAD train set
        train_dataset, vocab, validation_dataset = read_data(
        self.squad_train, self.squad_validation, 150,
        15, 10)
        validation_dataset = SQuADDataset(validation_dataset, 150, 15, vocab) 


        # Random embeddings for test
        test_embed_matrix = torch.rand(vocab.get_vocab_size(), embedding_dim)
        test_cbow = CBOW(test_embed_matrix)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,
                                          test_cbow.parameters()),
                                   lr=lr)

        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

        for batch in loader:
            passage = batch["passage_tokens"]
            question = batch["question_tokens"]
            span_start = batch["span_start"]
            span_end = batch["span_end"]
            output_dict = test_cbow(passage, question)
            softmax_start_logits = output_dict["softmax_start_logits"]
            softmax_end_logits = output_dict["softmax_end_logits"]
            loss = nll_loss(softmax_start_logits, span_start.view(-1))
            loss += nll_loss(softmax_end_logits, span_end.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
