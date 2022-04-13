import json
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import logging
import mmap

from .utils import Vocab, char_span_to_token_span

from nltk.tokenize.destructive import NLTKWordTokenizer
from nltk.tokenize.regexp import WordPunctTokenizer
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
        handle data batching explicitly (given a list of items)
    """
    metadatas = []
    passages = []
    questions = []
    span_starts = []
    span_ends = []
    for metadata, passage, question, span_start, span_end in batch:
        metadatas.append(metadata)
        passages.append(passage)
        questions.append(question)
        span_starts.append(span_start)
        span_ends.append(span_end)

    p_tensor = torch.from_numpy(np.concatenate(passages, axis=0)).long()
    q_tensor = torch.from_numpy(np.concatenate(questions, axis=0)).long()
    start = torch.LongTensor(span_starts).unsqueeze(1)
    end = torch.LongTensor(span_ends).unsqueeze(1)
    return {'metadata':metadatas, "passage_tokens":p_tensor, "question_tokens":q_tensor, "span_start":start, "span_end":end}


class SQuADDataset(Dataset):
    """
        docstring for SQuADDataset:
        keep metadata the same, convert tokens from strings into numpy array (using vocab)        
    """
    def __init__(self, data, max_passage_length, max_question_length, vocab):
        super(SQuADDataset, self).__init__()
        self.data = data
        self.max_passage_length = max_passage_length
        self.max_question_length = max_question_length
        self.vocab = vocab

    def __getitem__(self, index):
        metadata = self.data[index]['metadata']
        passage = np.zeros((1,self.max_passage_length))
        for i, tok in enumerate(self.data[index]['passage_tokens']):
            passage[0][i] = self.vocab.get_index_from_token(tok)
        
        question = np.zeros((1,self.max_question_length))
        for i, tok in enumerate(self.data[index]['question_tokens']):
            question[0][i] = self.vocab.get_index_from_token(tok)

        return metadata, passage, question, self.data[index]['span_start'], self.data[index]['span_end']

    def __len__(self):
        return len(self.data)



def load_embeddings(glove_path, vocab):
    """
    Create an embedding matrix for a Vocabulary.
    """
    vocab_size = vocab.get_vocab_size()
    words_to_keep = set(vocab.get_index_to_token_vocabulary().values())
    glove_embeddings = {}
    embedding_dim = None

    logger.info("Reading GloVe embeddings from {}".format(glove_path))
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file,
                         total=get_num_lines(glove_path)):
            fields = line.strip().split(" ")
            word = fields[0]
            if word in words_to_keep:
                vector = np.asarray(fields[1:], dtype="float32")
                if embedding_dim is None:
                    embedding_dim = len(vector)
                else:
                    assert embedding_dim == len(vector)
                glove_embeddings[word] = vector

    all_embeddings = np.asarray(list(glove_embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    logger.info("Initializing {}-dimensional pretrained "
                "embeddings for {} tokens".format(
                    embedding_dim, vocab_size))
    embedding_matrix = torch.FloatTensor(
        vocab_size, embedding_dim).normal_(
            embeddings_mean, embeddings_std)
    # Manually zero out the embedding of the padding token (0).
    embedding_matrix[0].fill_(0)
    # This starts from 1 because 0 is the padding token, which
    # we don't want to modify.
    for i in range(1, vocab_size):
        word = vocab.get_token_from_index(i)

        # If we don't have a pre-trained vector for this word,
        # we don't change the row and the word has random initialization.
        if word in glove_embeddings:
            embedding_matrix[i] = torch.FloatTensor(glove_embeddings[word])
    return embedding_matrix


def read_data(squad_train_path, squad_dev_path, max_passage_length,
              max_question_length, min_token_count):
    """
    Read SQuAD data, and filter by passage and question length.
    """
    train_dataset = _generate_examples(squad_train_path)
    logger.info("Read {} training examples".format(len(train_dataset)))

    # Filter out examples with passage length greater than max_passage_length
    # or question length greater than max_question_length
    logger.info("Filtering out examples in train set with passage length "
                "greater than {} or question length greater than {}".format(
                    max_passage_length, max_question_length))
    train_dataset = [
        instance for instance in tqdm(train_dataset) if
        len(instance["passage_tokens"]) <= max_passage_length and
        len(instance["question_tokens"]) <= max_question_length]
    logger.info("{} training examples remain after filtering".format(
        len(train_dataset)))


    # Make a vocabulary object from the train set
    train_vocab = Vocab()
    # Make position 0 a PAD token, which can be useful if you
    train_vocab.add_and_get_index("<PAD>")
    # Make position 1 the UNK token
    train_vocab.add_and_get_index("<UNK>")
    word_counter = Counter()

    for instance in train_dataset:
        for tok in instance["passage_tokens"]:
            word_counter[tok] += 1

    for word in word_counter:
        if word_counter[word] >= min_token_count:
            train_vocab.add_and_get_index(word)


    # Read SQuAD validation set
    logger.info("Reading SQuAD validation set at {}".format(
        squad_dev_path))
    validation_dataset = _generate_examples(squad_dev_path)
    logger.info("Read {} validation examples".format(
        len(validation_dataset)))

    # Filter out examples with passage length greater than max_passage_length
    # or question length greater than max_question_length
    logger.info("Filtering out examples in validation set with passage length "
                "greater than {} or question length greater than {}".format(
                    max_passage_length, max_question_length))
    validation_dataset = [
        instance for instance in tqdm(validation_dataset) if
        len(instance["passage_tokens"]) <= max_passage_length and
        len(instance["question_tokens"]) <= max_question_length]
    logger.info("{} validation examples remain after filtering".format(
        len(validation_dataset)))
    

    return train_dataset, train_vocab, validation_dataset


def read_test_data(squad_test_path, max_passage_length, max_question_length):
    """
    Read SQuAD test data
    """
    test_dataset = _generate_examples(squad_test_path)
    logger.info("Read {} test examples".format(
        len(test_dataset)))
    # Filter out examples with passage length greater than
    # max_passage_length or question length greater than
    # max_question_length
    logger.info("Filtering out examples in test set with "
                "passage length greater than {} or question "
                "length greater than {}".format(
                    max_passage_length, max_question_length))
    test_dataset = [
        instance for instance in tqdm(test_dataset) if
        (len(instance["passage_tokens"]) <=
         max_passage_length) and
        (len(instance["question_tokens"]) <=
         max_question_length)]
    logger.info("{} test examples remain after filtering".format(
        len(test_dataset)))

    return test_dataset

def _generate_examples(filepath):
    """
        This function reads in the json files, and returns the examples in the form we wanted.
        The output is a list of examples.
        Each example contains the tokenized passage, question, the start/end token index, and some metadata (for evaluation, and debug).
    """
    logger.info("generating examples from = %s", filepath)
    key = 0
    data = []
    filt = 0
    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)
        for article in squad["data"]:
            title = article.get("title", "")
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].lower()  # do not strip leading blank spaces GH-2585

                # tokenize context
                tok = WordPunctTokenizer()
                context_spans = list(tok.span_tokenize(context))
                context_tokens = [context[span[0]:span[1]] for span in context_spans]
                

                for qa in paragraph["qas"]:
                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]
                    
                    question = qa["question"].lower()
                    question_spans = list(tok.span_tokenize(question))
                    question_tokens = [question[span[0]:span[1]] for span in question_spans]

                    char_span = (answer_starts[0], answer_starts[0]+len(answers[0]))
                    token_span, error = char_span_to_token_span(context_spans, char_span)

                    data.append({
                        "title": title,
                        "passage_tokens": context_tokens,
                        "question_tokens": question_tokens,
                        
                        'metadata':{
                            "original_passage": context,
                            "token_offsets": context_spans,
                            "answer_texts": answers,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                            "id": qa["id"],
                        },
                        "span_start": token_span[0],
                        "span_end": token_span[1],
                    })

                    key += 1

    return data

def text_to_instance(passage, question):
    """
    for demo: convert text form passage/question pair into the data form we want
    """
    tok = WordPunctTokenizer()
    context_spans = list(tok.span_tokenize(passage))
    context_tokens = [passage[span[0]:span[1]] for span in context_spans]

    question_spans = list(tok.span_tokenize(question))
    question_tokens = [question[span[0]:span[1]] for span in question_spans]


    data = [{
        "passage_tokens": context_tokens,
        "question_tokens": question_tokens,
        'metadata':{
            "original_passage": passage,
            "token_offsets": context_spans,
        },
        "span_start": 0,
        "span_end": 1,
    }]
    return data



def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines



if __name__ == '__main__':
    squad_train_path = 'squad/train_small.json' 
    squad_dev_path = 'squad/val_small.json' 
    max_passage_length = 150
    max_question_length =50 
    min_token_count = 3

    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)

    train_dataset, train_vocab, validation_dataset = read_data(squad_train_path, squad_dev_path, max_passage_length,
              max_question_length, min_token_count)

    squad = SQuADDataset(train_dataset, max_passage_length, max_question_length, train_vocab)

    loader = DataLoader(squad, batch_size=4, shuffle=False, collate_fn=None, num_workers=4)

    for batch in loader:
        passage = batch["passage_tokens"].long()
        question = batch["question_tokens"].long()
        span_start = batch["span_start"]
        span_end = batch["span_end"]
        metadata = batch.get("metadata", {})



