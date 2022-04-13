import os
import logging
import random
from unittest import TestCase

import torch

logger = logging.getLogger(__name__)


class RNNAttnRCTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(format=("%(asctime)s - %(levelname)s - "
                                    "%(name)s - %(message)s"),
                            level=logging.INFO)

        self.project_root = os.path.abspath(os.path.realpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir)))
        self.squad_train = os.path.join(self.project_root, "squad",
                                        "train_small.json")
        self.squad_validation = os.path.join(self.project_root, "squad",
                                             "val_small.json")
        self.squad_test = os.path.join(self.project_root, "squad",
                                       "test_small.json")

        seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(0)
