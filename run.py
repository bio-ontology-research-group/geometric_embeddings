import mowl

mowl.init_jvm("100g", "1g", 8)

from models.elembeddings_ppi import ELEmPPI
from data_utils.data import PPIYeastDataset
from evaluation_utils import ModelRankBasedEvaluator
import torch as th
import random


random.seed(0)
th.manual_seed(0)

dataset = PPIYeastDataset(
    "ontology.owl",
    "valid.owl",
    "test.owl",
)

model = ELEmPPI(
    dataset,
    embed_dim=50,
    margin=0.1,
    loss_type="leaky_relu",
    reg_r=1,
    reg_mode="relaxed",
    neg_losses=["gci0", "gci1", "gci2", "gci3"]
    learning_rate=0.001,
    epochs=400,
    batch_size=4096 * 8,
    model_filepath="model.pt",
    device="cuda",
)

model.train(
    loss_weight=True,
    path_to_dc='ontology_dc.owl',
)

with th.no_grad():
    model.load_best_model()
    evaluator = ModelRankBasedEvaluator(
        model,
        device="cuda",
        eval_method=model.eval_method,
    )

    evaluator.evaluate(show=True)
