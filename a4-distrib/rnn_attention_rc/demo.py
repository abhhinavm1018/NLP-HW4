import logging

from torch.utils.data import Dataset, DataLoader
from flask import Flask, jsonify, render_template, request

from .utils import get_best_span, move_to_cuda
from .data import SQuADDataset, text_to_instance, collate_fn

logger = logging.getLogger(__name__)


def run_demo(model, train_vocab, host, port, cuda):
    """
    Run the web demo application.
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/_get_answer")
    def get_answer():
        # Take user input and convert to Instance
        user_context = request.args.get("context", "", type=str)
        user_question = request.args.get("question", "", type=str)

        input_instance = text_to_instance(
                    question=user_question,
                    passage=user_context)


        dataset = SQuADDataset(input_instance, 150, 15, train_vocab)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

        batch = next(iter(loader))
        if cuda:
            batch = move_to_cuda(batch, cuda_device=0)

        # Extract relevant data from batch.
        passage = batch["passage_tokens"]
        question = batch["question_tokens"]
        metadata = batch.get("metadata", {})

        # Run data through model to get start and end logits.
        output_dict = model(passage, question)
        start_logits = output_dict["start_logits"]
        end_logits = output_dict["end_logits"]

        # Compute the best span
        best_span = get_best_span(start_logits, end_logits)

        # Get the string corresponding to the best span
        passage_str = metadata[0]['original_passage']
        offsets = metadata[0]['token_offsets']
        predicted_span = tuple(best_span[0].data.cpu().numpy())
        start_offset = offsets[predicted_span[0]][0]
        end_offset = offsets[predicted_span[1]][1]
        best_span_string = passage_str[start_offset:end_offset]

        # Return the best string back to the GUI
        return jsonify(answer=best_span_string)

    logger.info("Launching Demo...")
    app.run(port=port, host=host)
