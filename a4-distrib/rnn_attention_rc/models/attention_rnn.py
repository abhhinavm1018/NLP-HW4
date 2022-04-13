# This list of imports is likely incomplete --- add anything you need.
# TODO: Your code here.
import torch.nn as nn


class AttentionRNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size,
                 dropout):
        """
        Parameters
        ----------
        embedding_matrix: FloatTensor
            FloatTensor matrix of shape (num_words, embedding_dim),
            where each row of the matrix is a word vector for the
            associated word index.

        hidden_size: int
            The size of the hidden state in the RNN.

        dropout: float
            The dropout rate.
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(AttentionRNN, self).__init__()

        # Create Embedding object
        # TODO: Your code here.

        # Load our embedding matrix weights into the Embedding object,
        # and make them untrainable (requires_grad=False)
        # TODO: Your code here.

        # Make a RNN to encode the passage. Note that batch_first=True.
        # TODO: Your code here.

        # Make a RNN to encode the question. Note that batch_first=True.
        # TODO: Your code here.

        # Affine transform for attention.
        # TODO: Your code here.

        # Affine transform for predicting start index.
        # TODO: Your code here.

        # Affine transform for predicting end index.
        # TODO: Your code here.

        # Dropout layer
        # TODO: Your code here.

        # Stores the number of gradient updates performed.
        self.global_step = 0

    def forward(self, passage, question):
        """
        The forward pass of the RNN-based model with attention.

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
        # Make a mask for the passage. Shape: ?
        # TODO: Your code here.

        # Make a mask for the question. Shape: ?
        # TODO: Your code here.

        # Make a LongTensor with the length (number non-padding words
        # in) each passage.
        # Shape: ?
        # TODO: Your code here.

        # Make a LongTensor with the length (number non-padding words
        # in) each question.
        # Shape: ?
        # TODO: Your code here.

        # Part 1: Embed the passages and the questions.
        # 1.1 Embed the passage.
        # TODO: Your code here.
        # Shape: ?

        # 1.2. Embed the question.
        # TODO: Your code here.
        # Shape: ?

        # Part 2. Encode the embedded passages with the RNN.
        # 2.1. Sort embedded passages by decreasing order of passage_lengths.
        # TODO: Your code here.

        # 2.2. Pack the passages with torch.nn.utils.rnn.pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.

        # 2.3. Encode the packed passages with the RNN.
        # TODO: Your code here.

        # 2.4. Unpack (pad) the passages with
        # torch.nn.utils.rnn.pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.

        # 2.5. Unsort the unpacked, encoded passage to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.

        # Part 3. Encode the embedded questions with the RNN.
        # 3.1. Sort the embedded questions by decreasing order
        #      of question_lengths.
        # TODO: Your code here.

        # 3.2. Pack the questions with pack_padded_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # TODO: Your code here.

        # 3.3. Encode the questions with the RNN.
        # TODO: Your code here.

        # 3.4. Unpack (pad) the questions with pad_packed_sequence.
        # Hint: Make sure you have the proper value for batch_first.
        # Shape: ?
        # TODO: Your code here.

        # 3.5. Unsort the unpacked, encoded question to restore the
        # initial ordering.
        # Hint: Look into torch.index_select or NumPy/PyTorch fancy indexing.
        # Shape: ?
        # TODO: Your code here.

        # Part 4. Calculate attention weights and attend to question.
        # 4.1. Expand the encoded question to shape suitable for attention.
        # Hint: Think carefully about what the shape of the attention
        # input vector should be. torch.unsqueeze and torch.expand
        # might be useful.
        # Shape: ?
        # TODO: Your code here.

        # 4.2. Expand the encoded passage to shape suitable for attention.
        # Hint: Think carefully about what the shape of the attention
        # input vector should be. torch.unsqueeze and torch.expand
        # might be useful.
        # Shape: ?
        # TODO: Your code here.

        # 4.3. Build attention_input. This is the tensor passed through
        # the affine transform.
        # Hint: Think carefully what the shape of this tensor should be.
        # torch.cat might be useful.
        # Shape: ?
        # TODO: Your code here.

        # 4.4. Apply affine transform to attention input to get
        # attention logits. You will need to slightly reshape it
        # into a tensor of the shape you expect.
        # Shape: ?
        # TODO: Your code here.

        # 4.5. Masked-softmax the attention logits over the last dimension
        # to normalize and make the attention logits a proper
        # probability distribution.
        # Shape: ?
        # TODO: Your code here.

        # 4.6. Use the attention weights to get a weighted average
        # of the RNN output from encoding the question for each
        # passage word.
        # Hint: torch.bmm might be helpful.
        # Shape: ?
        # TODO: Your code here.

        # Part 5: Combine the passage and question representations by
        # concatenating the passage and question representations with
        # their product.
        # 5.1. Concatenate to make the combined representation.
        # Hint: Use torch.cat
        # Shape: ?
        # TODO: Your code here.

        # Part 6: Compute logits for answer start index.

        # 6.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.

        # 6.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your start_logits.
        # Hint: torch.Tensor.masked_fill might be helpful.
        # Shape: ?
        # TODO: Your code here.

        # 6.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_start_logits.
        # Hint: masked_log_softmax function in cbow.py might be helpful.
        # Shape: ?
        # TODO: Your code here.

        # Part 7: Compute logits for answer end index.

        # 7.1. Apply the affine transformation, and edit the shape.
        # Shape after affine transformation: ?
        # Shape after editing shape: ?
        # TODO: Your code here.

        # 7.2. Replace the masked values so they have a very low score (-1e7).
        # This tensor is your end_logits.
        # Hint: torch.Tensor.masked_fill might be helpful.
        # Shape: ?
        # TODO: Your code here.

        # 7.3. Apply a padding-aware log-softmax to normalize.
        # This tensor is your softmax_end_logits.
        # Hint: masked_log_softmax function in cbow.py might be helpful.
        # Shape: ?
        # TODO: Your code here.

        # Part 8: Output a dictionary with the start_logits, end_logits,
        # softmax_start_logits, softmax_end_logits.
        # TODO: Your code here. Remove the NotImplementedError below.
        # return {
        #     "start_logits":,
        #     "end_logits":,
        #     "softmax_start_logits":,
        #     "softmax_end_logits":,
        # }
        raise NotImplementedError
