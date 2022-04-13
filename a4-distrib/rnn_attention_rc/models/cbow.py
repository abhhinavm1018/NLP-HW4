import torch
import torch.nn as nn

def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + 1e-13).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

class CBOW(nn.Module):
    def __init__(self, embedding_matrix):
        """
        Parameters
        ----------
        embedding_matrix: FloatTensor
            FloatTensor matrix of shape (num_words, embedding_dim),
            where each row of the matrix is a word vector for the
            associated word index.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(CBOW, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        # Create Embedding object
        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)
        # Load our embedding matrix weights into the Embedding object,
        # and make them untrainable (requires_grad=False)
        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)

        # Affine transform for predicting start index.
        self.start_output_projection = nn.Linear(3 * self.embedding_dim, 1)
        # Affine transform for predicting end index.
        self.end_output_projection = nn.Linear(3 * self.embedding_dim, 1)

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, passage, question):
        """
        The forward pass of the CBOW model.

        Parameters
        ----------
        passage: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, passage_length)
            representing the words in the passage for each batch.

        question: Variable(LongTensor)
            A Variable(LongTensor) of shape (batch_size, question_length)
            representing the words in the question for each batch.

        Returns
        -------
        An output dictionary consisting of:
        start_logits: Variable(FloatTensor)
            The first element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        end_logits: Variable(FloatTensor)
            The second element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Each value is the score
            assigned to a given token. Masked indices are assigned very
            small scores (-1e7).

        softmax_start_logits: Variable(FloatTensor)
            The third element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer begins.
            Masked indices have probability mass of -inf.

        softmax_end_logits: Variable(FloatTensor)
            The fourth element in the returned tuple. Variable(FloatTensor) of
            shape (batch_size, max_passage_size). Exactly the same as
            start_logits, but with a masked log softmax applied. Represents
            a probability distribution over the passage, indicating the
            probability that any given token is where the answer end.
            Masked indices have probability mass of -inf.
        """
        # Mask: FloatTensor with 0 in positions that are
        # padding (word index 0) and 1 in positions with actual words.
        # Shape: (batch_size, max_passage_size)
        passage_mask = (passage != 0).type(
            torch.cuda.FloatTensor if passage.is_cuda else
            torch.FloatTensor)
        # Shape: (batch_size, max_question_size)
        question_mask = (question != 0).type(
            torch.cuda.FloatTensor if question.is_cuda else
            torch.FloatTensor)
        # The number of non-padding words in each question.
        # Shape: (batch_size,)
        question_lengths = question_mask.sum(dim=1)

        # Part 1: Embed the passages and the questions.
        # Embed the passage.
        # Shape: (batch_size, max_passage_size, embedding_dim)
        embedded_passage = self.embedding(passage)
        # Embed the question.
        # Shape: (batch_size, max_question_size, embedding_dim)
        embedded_question = self.embedding(question)

        # Part 2: Encode the question by averaging the embeddings of
        # their constituent words.
        # Note that simply using torch.mean here would compute an average
        # for each question that includes padding, which we want to exclude.
        # Shape: (batch_size, embedding_dim)
        encoded_q = (torch.sum(embedded_question, dim=1) /
                     question_lengths.unsqueeze(1))

        # Part 3: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.

        # Reshape the question encoding to make it amenable to concatenation
        # with the embedded passage.
        # Shape: (batch_size, max_passage_size, embedding_dim)
        tiled_encoded_q = encoded_q.unsqueeze(dim=1).expand_as(
            embedded_passage)
        # Shape: (batch_size, max_passage_size, 3 * embedding_dim)
        combined_x_q = torch.cat([embedded_passage, tiled_encoded_q,
                                  embedded_passage * tiled_encoded_q], dim=-1)


        # Part 4: Compute logits for answer start index.
        # Shape after start_output_projection: (batch_size, max_passage_size,
        #                                       1)
        # Shape after squeeze: (batch_size, max_passage_size)
        # Shape after masked_fill: (batch_size, max_passage_size)
        # Shape after masked_log_softmax: (batch_size, max_passage_size)
        start_logits = self.start_output_projection(combined_x_q).squeeze(-1)
        assert start_logits.dim() == passage_mask.dim()
        passage_mask = passage_mask.bool()
        start_logits = start_logits.masked_fill(~passage_mask, -1e7)
        softmax_start_logits = masked_log_softmax(start_logits, passage_mask)

        # Part 5: Compute logits for answer end index.
        # Shape after end_output_projection: (batch_size, max_passage_size, 1)
        # Shape after squeeze: (batch_size, max_passage_size)
        # Shape after masked_fill: (batch_size, max_passage_size)
        # Shape after masked_log_softmax: (batch_size, max_passage_size)
        end_logits = self.end_output_projection(combined_x_q).squeeze(-1)
        assert end_logits.dim() == passage_mask.dim()
        end_logits = end_logits.masked_fill(~passage_mask, -1e7)
        softmax_end_logits = masked_log_softmax(end_logits, passage_mask)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "softmax_start_logits": softmax_start_logits,
            "softmax_end_logits": softmax_end_logits
        }
