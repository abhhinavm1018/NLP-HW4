import argparse
import logging
import os
import shutil
import sys

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.nn.functional import nll_loss
from tqdm import tqdm

from datasets import load_metric

sys.path.append(os.path.join(os.path.dirname(__file__)))
from rnn_attention_rc.data import load_embeddings, read_data, SQuADDataset, read_test_data, collate_fn
from rnn_attention_rc.demo import run_demo
from rnn_attention_rc.models.attention_rnn import AttentionRNN
from rnn_attention_rc.models.cbow import CBOW
from rnn_attention_rc.models.rnn import RNN
from rnn_attention_rc.utils import move_to_cuda, get_best_span, Vocab

logger = logging.getLogger(__name__)

# Dictionary of model type strings to model classes
MODEL_TYPES = {
    "attention": AttentionRNN,
    # For compatibility with serialization
    "attentionrnn": AttentionRNN,
    "cbow": CBOW,
    "rnn": RNN
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    parser.add_argument("--squad-train-path", type=str,
                        default=os.path.join(
                            project_root, "squad", "train_small.json"),
                        help="Path to the SQuAD training data.")
    parser.add_argument("--squad-dev-path", type=str,
                        default=os.path.join(
                            project_root, "squad", "val_small.json"),
                        help="Path to the SQuAD dev data.")
    parser.add_argument("--squad-test-path", type=str,
                        default=os.path.join(
                            project_root, "squad", "test_small.json"),
                        help="Path to the SQuAD test data.")
    parser.add_argument("--glove-path", type=str,
                        default=os.path.join(project_root, "glove",
                                             "glove.6B.50d.txt"),
                        help="Path to word vectors in GloVe format.")
    parser.add_argument("--load-path", type=str,
                        help=("Path to load a saved model from and "
                              "evaluate on test data. May not be "
                              "used with --save-dir."))
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--model-type", type=str, default="cbow",
                        choices=["cbow", "rnn", "attention"],
                        help="Model type to train.")
    parser.add_argument("--min-token-count", type=int, default=10,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--max-passage-length", type=int, default=150,
                        help="Maximum number of words in the passage.")
    parser.add_argument("--max-question-length", type=int, default=15,
                        help="Maximum number of words in the question.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and Attention models.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="The learning rate to use.")
    parser.add_argument("--log-period", type=int, default=50,
                        help=("Update training metrics every "
                              "log-period weight updates."))
    parser.add_argument("--validation-period", type=int, default=500,
                        help=("Calculate metrics on validation set every "
                              "validation-period weight updates."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda", action="store_true",
                        help="Train or evaluate with GPU.")
    parser.add_argument("--demo", action="store_true",
                        help="Run the interactive web demo.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to use for web demo.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to use for web demo.")
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load a model from checkpoint and evaluate it on test data.
    if args.load_path:
        logger.info("Loading saved model from {}".format(args.load_path))

        # If evaluating with CPU, force all tensors to CPU.
        # This lets us load models trained on the GPU and evaluate with CPU.
        saved_state_dict = torch.load(args.load_path,
                                      map_location=None if args.cuda
                                      else lambda storage, loc: storage)

        # Extract the contents of the state dictionary.
        model_type = saved_state_dict["model_type"]
        model_weights = saved_state_dict["model_weights"]
        model_init_arguments = saved_state_dict["init_arguments"]
        model_global_step = saved_state_dict["global_step"]

        # Reconstruct a model of the proper type with the init arguments.
        saved_model = MODEL_TYPES[model_type.lower()](**model_init_arguments)
        # Load the weights
        saved_model.load_state_dict(model_weights)
        # Set the global step
        saved_model.global_step = model_global_step

        logger.info("Successfully loaded model!")

        # Move model to GPU if CUDA is on.
        if args.cuda:
            saved_model = saved_model.cuda()

        # Load the serialized train_vocab.
        vocab_dir = os.path.join(os.path.dirname(args.load_path),
                                 "train_vocab")
        logger.info("Loading train vocabulary from {}".format(vocab_dir))
        train_vocab = Vocab().from_files(vocab_dir)
        logger.info("Successfully loaded train vocabulary!")

        if args.demo:
            # Run the demo with the loaded model.
            run_demo(saved_model, train_vocab, args.host, args.port,
                     args.cuda)
            sys.exit(0)

        # Evaluate on the SQuAD test set.
        logger.info("Reading SQuAD test set at {}".format(
            args.squad_test_path))
        test_dataset = read_test_data(args.squad_test_path, args.max_passage_length, args.max_question_length)
        test_dataset = SQuADDataset(test_dataset, args.max_passage_length, args.max_question_length, train_vocab)

        # Evaluate the model on the test set.
        logger.info("Evaluating model on the test set")
        (loss, span_start_accuracy, span_end_accuracy,
         span_accuracy, em, f1) = evaluate(
             saved_model, test_dataset, args.batch_size,
             train_vocab, args.cuda)
        # Log metrics to console.
        logger.info("Done evaluating on test set!")
        logger.info("Test Loss: {:.4f}".format(loss))
        logger.info("Test Span Start Accuracy: {:.4f}".format(
            span_start_accuracy))
        logger.info("Test Span End Accuracy: {:.4f}".format(span_end_accuracy))
        logger.info("Test Span Accuracy: {:.4f}".format(span_accuracy))
        logger.info("Test EM: {:.4f}".format(em))
        logger.info("Test F1: {:.4f}".format(f1))
        sys.exit(0)

    if not args.save_dir:
        raise ValueError("Must provide a value for --save-dir if training.")

    try:
        if os.path.exists(args.save_dir):
            # save directory already exists, do we really want to overwrite?
            input("Save directory {} already exists. Press <Enter> "
                  "to clear, overwrite and continue , or "
                  "<Ctrl-c> to abort.".format(args.save_dir))
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    except KeyboardInterrupt:
        print()
        sys.exit(0)

    # Write tensorboard logs to save_dir/logs.
    log_dir = os.path.join(args.save_dir, "logs")
    os.makedirs(log_dir)

    # Read the training and validaton dataset, and get a vocabulary
    # from the train set.
    train_dataset, train_vocab, validation_dataset = read_data(
        args.squad_train_path, args.squad_dev_path, args.max_passage_length,
        args.max_question_length, args.min_token_count)

    train_dataset = SQuADDataset(train_dataset, args.max_passage_length, args.max_question_length, train_vocab)
    validation_dataset = SQuADDataset(validation_dataset, args.max_passage_length, args.max_question_length, train_vocab) 

    # Save the train_vocab to a file.
    vocab_dir = os.path.join(args.save_dir, "train_vocab")
    logger.info("Saving train vocabulary to {}".format(vocab_dir))
    train_vocab.save_to_files(vocab_dir)

    # Read GloVe embeddings.
    embedding_matrix = load_embeddings(args.glove_path, train_vocab)

    # Create model of the correct type.
    if args.model_type == "cbow":
        logger.info("Building CBOW model")
        model = CBOW(embedding_matrix)
    if args.model_type == "rnn":
        logger.info("Building RNN model")
        model = RNN(embedding_matrix, args.hidden_size, args.dropout)
    if args.model_type == "attention":
        logger.info("Building attention RNN model")
        model = AttentionRNN(embedding_matrix, args.hidden_size,
                             args.dropout)

    logger.info(model)

    # Move model to GPU if running with CUDA.
    if args.cuda:
        model = model.cuda()
    # Create the optimizer, and only update parameters where requires_grad=True
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=args.lr)
    # Train for the specified number of epochs.
    for i in tqdm(range(args.num_epochs), unit="epoch"):
        train_epoch(model, train_dataset, validation_dataset, train_vocab,
                    args.batch_size, optimizer, args.log_period,
                    args.validation_period, args.save_dir, log_dir,
                    args.cuda)


def train_epoch(model, train_dataset, validation_dataset, vocab,
                batch_size, optimizer, log_period, validation_period,
                save_dir, log_dir, cuda):
    """
    Train the model for one epoch.
    """
    # Set model to train mode (turns on dropout and such).
    model.train()
    # Create objects for calculating metrics.
    span_start_accuracy_metric = load_metric('accuracy')
    span_end_accuracy_metric = load_metric('accuracy')
    squad_metrics = load_metric('squad')
    # Create Tensorboard logger.
    writer = SummaryWriter(log_dir)

    # Build iterater, and have it bucket batches by passage / question length.
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    log_period_losses = 0
    span_accs = []
    train_generator = tqdm(train_loader)
    for batch in train_generator:
        # move the data to cuda if available
        if cuda:
            batch = move_to_cuda(batch, cuda_device=0)
        # Extract the relevant data from the batch.
        passage = batch["passage_tokens"]
        question = batch["question_tokens"]
        span_start = batch["span_start"]
        span_end = batch["span_end"]
        metadata = batch.get("metadata", {})

        # Run data through model to get start and end logits.
        output_dict = model(passage, question)
        start_logits = output_dict["start_logits"]
        end_logits = output_dict["end_logits"]
        softmax_start_logits = output_dict["softmax_start_logits"]
        softmax_end_logits = output_dict["softmax_end_logits"]

        # Calculate loss for start and end indices.
        loss = nll_loss(softmax_start_logits, span_start.view(-1))
        loss += nll_loss(softmax_end_logits, span_end.view(-1))
        log_period_losses += loss.item()

        # Backprop and take a gradient step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.global_step += 1

        # Calculate categorical span start and end accuracy.
        span_start_accuracy_metric.add_batch(predictions=start_logits.argmax(dim=-1), references=span_start.view(-1))
        span_end_accuracy_metric.add_batch(predictions=end_logits.argmax(dim=-1), references=span_end.view(-1))

        # Compute the best span, and calculate overall span accuracy.
        best_span = get_best_span(start_logits, end_logits)

        span_accs.append(compute_span_accuracy(
            best_span, torch.cat([span_start, span_end], -1)))
        # Calculate EM and F1 scores
        calculate_em_f1(best_span, metadata, passage.size(0),
                        squad_metrics)

        if model.global_step % log_period == 0:
            # Calculate metrics on train set.
            loss = log_period_losses / log_period
            span_start_accuracy = span_start_accuracy_metric.compute()['accuracy']
            span_end_accuracy = span_end_accuracy_metric.compute()['accuracy']
            span_accuracy = sum(span_accs)/ len(span_accs)
            eval_result = squad_metrics.compute()
            em, f1 = eval_result['exact_match'], eval_result['f1']
            tqdm_description = _make_tqdm_description(
                loss, em, f1, 'Train')
            # Log training statistics to progress bar
            train_generator.set_description(tqdm_description)
            # Log training statistics to Tensorboard
            log_to_tensorboard(writer, model.global_step, "train",
                               loss, span_start_accuracy, span_end_accuracy,
                               span_accuracy, em, f1)
            log_period_losses = 0
            span_accs = []

        if model.global_step % validation_period == 0:
            # Calculate metrics on validation set.
            (loss, span_start_accuracy, span_end_accuracy,
             span_accuracy, em, f1) = evaluate(
                 model, validation_dataset, batch_size, vocab, cuda)
            # Save a checkpoint.
            save_name = ("{}_step_{}_loss_{:.3f}_"
                         "em_{:.3f}_f1_{:.3f}.pth".format(
                             model.__class__.__name__, model.global_step,
                             loss, em, f1))
            save_model(model, save_dir, save_name)
            # Log validation statistics to Tensorboard.
            log_to_tensorboard(writer, model.global_step, "validation",
                               loss, span_start_accuracy, span_end_accuracy,
                               span_accuracy, em, f1)


def evaluate(model, evaluation_dataset, batch_size, vocab, cuda):
    """
    Evaluate a model on an evaluation dataset.
    """
    # Set model to evaluation mode (turns off dropout and such)
    model.eval()
    # Create objects for calculating metrics.
    span_start_accuracy = load_metric('accuracy')
    span_end_accuracy = load_metric('accuracy')
    squad_metrics = load_metric('squad')

    # Build iterater, and have it bucket batches by passage / question length.
    eval_loader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Get a generator of train batches.
    num_evaluation_batches = 0

    span_accs = []
    batch_losses = 0
    for batch in tqdm(eval_loader):
        # move the data to cuda if available
        if cuda:
            batch = move_to_cuda(batch, cuda_device=0)
        # Extract the relevant data from the batch.
        passage = batch["passage_tokens"]
        question = batch["question_tokens"]
        span_start = batch["span_start"]
        span_end = batch["span_end"]
        metadata = batch.get("metadata", {})

        # Run data through model to get start and end logits.
        output_dict = model(passage, question)
        start_logits = output_dict["start_logits"]
        end_logits = output_dict["end_logits"]
        softmax_start_logits = output_dict["softmax_start_logits"]
        softmax_end_logits = output_dict["softmax_end_logits"]

        # Calculate loss for start and end indices.
        loss = nll_loss(softmax_start_logits, span_start.view(-1))
        loss += nll_loss(softmax_end_logits, span_end.view(-1))
        batch_losses += loss.item()

        # Calculate categorical span start and end accuracy.
        span_start_accuracy.add_batch(predictions=start_logits.argmax(dim=-1), references=span_start.view(-1))
        span_end_accuracy.add_batch(predictions=end_logits.argmax(dim=-1), references=span_end.view(-1))
        # Compute the best span, and calculate overall span accuracy.
        best_span = get_best_span(start_logits, end_logits)
        span_accs.append(compute_span_accuracy(best_span, torch.cat([span_start, span_end], -1)))
        # Calculate EM and F1 scores
        calculate_em_f1(best_span, metadata, passage.size(0),
                        squad_metrics)
        num_evaluation_batches += 1
    # Set the model back to train mode.
    model.train()
    
    loss = batch_losses / num_evaluation_batches
    eval_result = squad_metrics.compute()
    average_em, average_f1 = eval_result['exact_match'], eval_result['f1']
    tqdm_description = _make_tqdm_description(
        loss, average_em, average_f1, 'Valid')
    # Log training statistics to progress bar
    # evaluation_generator.set_description(tqdm_description)

    # Extract the values from the metrics objects
    average_span_start_accuracy = span_start_accuracy.compute()['accuracy']
    average_span_end_accuracy = span_end_accuracy.compute()['accuracy']
    average_span_accuracy = sum(span_accs)/ len(span_accs)
    return (batch_losses / num_evaluation_batches,
            average_span_start_accuracy,
            average_span_end_accuracy,
            average_span_accuracy,
            average_em,
            average_f1)


def calculate_em_f1(best_span, metadata, batch_size,
                    squad_metrics):
    """
    Calculates EM and F1 scores.
    """
    if metadata is not None:
        best_span_str = []
        for i in range(batch_size):
            passage_str = metadata[i]['original_passage']
            offsets = metadata[i]['token_offsets']
            predicted_span = tuple(best_span[i].data.cpu().numpy())
            start_offset = offsets[predicted_span[0]][0]
            end_offset = offsets[predicted_span[1]][1]
            best_span_string = passage_str[start_offset:end_offset]
            best_span_str.append(best_span_string)
            answers = metadata[i].get('answers', [])
            if answers:
                squad_metrics.add(prediction={'prediction_text':best_span_string, 'id':metadata[i]['id']}, reference={'answers':answers, 'id':metadata[i]['id']})


def compute_span_accuracy(preds, gold_spans):
    """
    Compare the predicted span and gold answer span, and return the accuracy.
    """
    acc = 0
    assert preds.size(0) == gold_spans.size(0)
    for i in range(preds.size(0)):
        if preds[i][0] == gold_spans[i][0] and preds[i][1] == gold_spans[i][1]:
            acc += 1
    return acc / float(preds.size(0))


def log_to_tensorboard(writer, step, prefix, loss, span_start_accuracy,
                       span_end_accuracy, span_accuracy,
                       em, f1):
    """
    Log metrics to Tensorboard.
    """
    writer.add_scalar("{}/loss".format(prefix), loss, step)
    writer.add_scalar("{}/span_start_accuracy".format(prefix),
                      span_start_accuracy, step)
    writer.add_scalar("{}/span_end_accuracy".format(prefix),
                      span_end_accuracy, step)
    writer.add_scalar("{}/span_accuracy".format(prefix),
                      span_accuracy, step)
    writer.add_scalar("{}/EM".format(prefix), em, step)
    writer.add_scalar("{}/F1".format(prefix), f1, step)


def save_model(model, save_dir, save_name):
    """
    Save a model to the disk.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_weights = model.state_dict()
    serialization_dictionary = {
        "model_type": model.__class__.__name__,
        "model_weights": model_weights,
        "init_arguments": model.init_arguments,
        "global_step": model.global_step
    }

    save_path = os.path.join(save_dir, save_name)
    torch.save(serialization_dictionary, save_path)


def _make_tqdm_description(average_loss, average_em, average_f1, split):
    """
    Build the string to use as the tqdm progress bar description.
    """
    metrics = {
        "%s Loss"%split: average_loss,
        "%s EM"%split: average_em,
        "%s F1"%split: average_f1
    }
    return ", ".join(["%s: %.3f" % (name, value) for name, value
                      in metrics.items()]) + " ||"


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()