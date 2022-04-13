# utils.py

from typing import List, Tuple, Optional
import numpy as np
import torch

import logging
import json
logger = logging.getLogger(__name__)


class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """

    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if index not in self.ints_to_objs:
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if object not in self.objs_to_ints:
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if object not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


class Vocab(Indexer):
    def get_vocab_size(self):
        return len(self.objs_to_ints)

    def get_token_from_index(self, index):
        return self.get_object(index)

    def get_index_to_token_vocabulary(self):
        return self.ints_to_objs

    def get_token_to_index_vocabulary(self):
        return self.objs_to_ints

    def get_index_from_token(self, token):
        """
            map to <UNK> if the token does not exist in the dictionary
        """
        index = self.index_of(token)
        if index == -1:
            return self.index_of("<UNK>")
        else:
            return index

    def save_to_files(self, vocab_dir):
        fw = open(vocab_dir, 'w')
        fw.write(json.dumps({'objs_to_ints':self.objs_to_ints, 'ints_to_objs':self.ints_to_objs}, indent=4))
        fw.close()

    def from_files(self, vocab_dir):
        vocab = json.load(open(vocab_dir))
        self.objs_to_ints = vocab['objs_to_ints']
        self.ints_to_objs = vocab['ints_to_objs']
        return self
        



# borrow from allennlp squadreader
# find token span smartly, even if the character spans don't exactly match (due to tokenization)
# we can still find the most similar span answer
def char_span_to_token_span(
    token_offsets: List[Optional[Tuple[int, int]]], character_span: Tuple[int, int]
) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).
    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.
    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.
    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether there was an error while matching the token spans exactly. If this is ``True``, it
        means there was an error in either the tokenization or the annotated character span. If this
        is ``False``, it means that we found tokens that match the character span exactly.
    """
    error = False
    start_index = 0
    while start_index < len(token_offsets) and (
        token_offsets[start_index] is None or token_offsets[start_index][0] < character_span[0]
    ):
        start_index += 1

    # If we overshot and the token prior to start_index ends after the first character, back up.
    if (
        start_index > 0
        and token_offsets[start_index - 1] is not None
        and token_offsets[start_index - 1][1] > character_span[0]
    ) or (
        start_index <= len(token_offsets)
        and token_offsets[start_index] is not None
        and token_offsets[start_index][0] > character_span[0]
    ):
        start_index -= 1

    if start_index >= len(token_offsets):
        raise ValueError("Could not find the start token given the offsets.")

    if token_offsets[start_index] is None or token_offsets[start_index][0] != character_span[0]:
        error = True

    end_index = start_index
    while end_index < len(token_offsets) and (
        token_offsets[end_index] is None or token_offsets[end_index][1] < character_span[1]
    ):
        end_index += 1
    if end_index == len(token_offsets):
        # We want a character span that goes beyond the last token. Let's see if this is salvageable.
        # We consider this salvageable if the span we're looking for starts before the last token ends.
        # In other words, we don't salvage if the whole span comes after the tokens end.
        if character_span[0] < token_offsets[-1][1]:
            # We also want to make sure we aren't way off. We need to be within 8 characters to salvage.
            if character_span[1] - 8 < token_offsets[-1][1]:
                end_index -= 1

    if end_index >= len(token_offsets):
        raise ValueError("Character span %r outside the range of the given tokens.")
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logger.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logger.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


def move_to_cuda(batch, cuda_device):
    """
        move tensor to specified cuda device
    """
    device = torch.device('cuda:%d'%cuda_device)
    for key, tensor in batch.items():
        if key in ['passage_tokens', 'question_tokens', 'span_start', 'span_end']:
            batch[key] = tensor.to(device=device)

    return batch
    
def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.
    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = torch.div(best_spans, passage_length, rounding_mode="trunc")
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)