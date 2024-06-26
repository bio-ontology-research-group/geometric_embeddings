import mowl

mowl.init_jvm("8g", "1g", 8)

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELModule
from mowl.projection.factory import projector_factory
from elembeddings_losses import *
from evaluation_utils import elembeddings_sim
from data_utils.dataloader import OntologyDataLoader
import torch as th
from torch import nn
from torch.nn.functional import relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from mowl.datasets import PathDataset


class ELEmModule(ELModule):
    """
    ELEmbeddings module

    :param nb_ont_classes: total number of classes
    :type nb_ont_classes: int
    :param nb_go_classes: total number of GO classes
    :type nb_go_classes: int
    :param nb_rels: total number of relations
    :type nb_rels: int
    :param go_threshold: value from class_index_dict starting from which GO classes are encoded
    :type go_threshold: int
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param loss_type: name of the loss, `relu` or `leaky_relu`
    :type loss_type: str
    :param reg_r: the radius of regularization ball
    :type reg_r: float/int
    :param reg_mode: mode of regularization, `relaxed` for \|c\| \leq R, `original` for \|c\| = R
    :type reg_mode: float/int
    :param neg_losses: abbreviations of GCIs to use for negative sampling (`gci0`, `gci1`, `gci2`, `gci3`, `gci0_bot`, `gci1_bot`, `gci3_bot`)
    :type neg_losses: list(str)
    """

    def __init__(
        self,
        nb_ont_classes,
        nb_go_classes,
        nb_rels,
        go_threshold,
        embed_dim=50,
        margin=0.1,
        loss_type="leaky_relu",
        reg_r=1,
        reg_mode="relaxed",
        neg_losses=["gci0", "gci1", "gci2", "gci3"]
    ):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_go_classes = nb_go_classes
        self.nb_rels = nb_rels
        self.go_threshold = go_threshold
        self.reg_r = reg_r
        self.neg_losses = neg_losses
        self.embed_dim = embed_dim

        if reg_mode in ["relaxed", "original"]:
            self.reg_mode = reg_mode
        else:
            raise ValueError('"reg_mode" should be one of ["relaxed", "original"]')

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.class_embed.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_embed.weight.data /= weight_data_normalized

        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.class_rad.weight.data, axis=1
        ).reshape(-1, 1)
        self.class_rad.weight.data /= weight_data_normalized

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.rel_embed.weight.data, axis=1
        ).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data_normalized

        self.margin = margin
        if loss_type in ["relu", "leaky_relu"]:
            self.loss_type = loss_type
        else:
            raise ValueError('"loss_type" should be one of ["relu", "leaky_relu"]')

    def class_reg(self, x):
        """
        Regularization function

        :param x: point to regularize
        :type x: torch.Tensor(torch.float64)
        :return res: regularized point
        :type res: torch.Tensor(torch.float64)
        """

        if self.reg_r is None:
            res = th.zeros(x.size()[0], 1)
        else:
            if self.reg_mode == "original":
                res = th.abs(th.linalg.norm(x, axis=1) - self.reg_norm)
            else:
                res = relu(th.linalg.norm(x, axis=1) - self.reg_norm)
            res = th.reshape(res, [-1, 1])
        return res

    def gci0_loss(self, data, neg=False):
        """
        Compute GCI0 (`C \sqsubseteq D`) loss

        :param data: GCI0 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci0_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci0_bot_loss(self, data, neg=False):
        """
        Compute GCI0_BOT (`C \sqsubseteq \bot`) loss

        :param data: GCI0_BOT data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci0_bot_loss(data, self.class_rad)

    def gci1_loss(self, data, neg=False):
        """
        Compute GCI1 (`C \sqcap D \sqsubseteq E`) loss

        :param data: GCI1 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci1_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci1_bot_loss(self, data, neg=False):
        """
        Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) loss

        :param data: GCI1_BOT data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci1_bot_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci2_loss(self, data, neg=False):
        """
        Compute GCI2 (`C \sqsubseteq \exists R.D`) loss

        :param data: GCI2 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci2_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci3_loss(self, data, neg=False):
        """
        Compute GCI3 (`\exists R.C \sqsubseteq D`) loss

        :param data: GCI3 data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci3_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci3_bot_loss(self, data, neg=False):
        """
        Compute GCI3_BOT (`\exists R.C \sqsubseteq \bot`) loss

        :param data: GCI3_BOT data
        :type data: torch.Tensor(torch.int64)
        :param neg: whether to compute negative or positive loss
        :type neg: bool
        :return: loss value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return gci3_bot_loss(data, self.class_rad)

    def eval_method(self, data):
        """
        Compute evaluation score (for GCI2 `C \sqsubseteq \exists R.D` axioms)

        :param data: evaluation data
        :type data: torch.Tensor(torch.int64)
        :return: evaluation score value for each data sample
        :return type: torch.Tensor(torch.float64)
        """

        return elembeddings_sim(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.margin,
            self.loss_type,
        )


class ELEmbeddings(EmbeddingELModel):
    """
    ELEmbeddings model

    :param dataset: dataset to use
    :type dataset: data_utils.data.PPIYeastDataset/data_utils.data.PPIHumanDataset/data_utils.data.AFPYeastDataset/data_utils.data.AFPHumanDataset
    :param embed_dim: embedding dimension
    :type embed_dim: int
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param reg_r: the radius of regularization ball
    :type reg_r: float/int
    :param reg_mode: mode of regularization, `relaxed` for \|c\| \leq R, `original` for \|c\| = R
    :type reg_mode: float/int
    :param neg_losses: abbreviations of GCIs to use for negative sampling (`gci0`, `gci1`, `gci2`, `gci3`, `gci0_bot`, `gci1_bot`, `gci3_bot`)
    :type neg_losses: list(str)
    :param loss_type: name of the loss, `relu` or `leaky_relu`
    :type loss_type: str
    :param learning_rate: learning rate
    :type learning_rate: float
    :param epochs: number of training epochs 
    :type epochs: int
    :param batch_size: batch size
    :type batch_size: int
    :param model_filepath: path to model checkpoint
    :type model_filepath: str
    :param device: device to use, `cpu` or `cuda`
    :type device: str
    """

    def __init__(
        self,
        dataset,
        embed_dim=50,
        margin=0,
        reg_r=1,
        reg_mode="relaxed",
        neg_losses=["gci0", "gci1", "gci2", "gci3"]
        loss_type="leaky_relu",
        learning_rate=0.001,
        epochs=1000,
        batch_size=4096 * 8,
        model_filepath=None,
        device="cpu",
    ):
        super().__init__(
            dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath,
        )

        self.embed_dim = embed_dim
        self.margin = margin
        self.reg_r = reg_r
        self.reg_mode = reg_mode
        self.neg_losses = neg_losses
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()

    def init_model(self):
        """
        Load ELEmbeddings module
        """

        self.nb_go_classes = len(
            [v for k, v in self.class_index_dict.items() if "GO" in k]
        )
        self.go_threshold = min(
            [v for k, v in self.class_index_dict.items() if "GO" in k]
        )
        self.module = ELEmModule(
            len(self.class_index_dict),
            self.nb_go_classes,
            len(self.object_property_index_dict),
            self.go_threshold,
            embed_dim=self.embed_dim,
            margin=self.margin,
            loss_type=self.loss_type,
            reg_r=self.reg_r,
            reg_mode=self.reg_mode,
            neg_losses=self.neg_losses,
        ).to(self.device)
        self.eval_method = self.module.eval_method

    def load_eval_data(self):
        """
        Load evaluation data
        """

        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.evaluation_classes

        self._head_entities = eval_classes[0].as_str
        self._tail_entities = eval_classes[1].as_str

        eval_projector = projector_factory(
            "taxonomy_rels", taxonomy=False, relations=[eval_property]
        )

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True

    def get_embeddings(self):
        """
        Get embeddings of relations and classes from the model checkpoint

        :return ent_embeds: dictionary class_name: its embedding
        :type ent_embeds: dict(str, numpy.array(numpy.float64))
        :return rel_embeds: dictionary relation_name: its embedding
        :type rel_embeds: dict(str, numpy.array(numpy.float64))
        """

        self.init_model()

        print("Load the best model", self.model_filepath)
        self.load_best_model()

        ent_embeds = {
            k: v
            for k, v in zip(
                self.class_index_dict.keys(),
                self.module.class_embed.weight.cpu().detach().numpy(),
            )
        }
        rel_embeds = {
            k: v
            for k, v in zip(
                self.object_property_index_dict.keys(),
                self.module.rel_embed.weight.cpu().detach().numpy(),
            )
        }
        return ent_embeds, rel_embeds

    def load_best_model(self):
        """
        Load the model from the checkpoint
        """

        self.init_model()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

    @property
    def training_set(self):
        """
        Get a set of triples that are true positives from the train dataset
        """

        self.load_eval_data()
        return self._training_set

    @property
    def testing_set(self):
        """
        Get a set of triples that are true positives from the test dataset
        """

        self.load_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        """
        Get a set of head entities `h` from triples `(h, r, t)`
        """

        self.load_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        """
        Get a set of tail entities `t` from triples `(h, r, t)`
        """

        self.load_eval_data()
        return self._tail_entities

    def train(self):
        raise NotImplementedError


class ELEmPPI(ELEmbeddings):
    """
    Final ELEmbeddings model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(
        self,
        patience=10,
        epochs_no_improve=20,
        loss_weight=True,
        path_to_dc=None,
        random_neg_fraction=1,
    ):
        """
        Model training

        :param patience: patience parameter for the scheduler
        :type patience: int
        :param epochs_no_improve: for how many epochs validation loss doesn't improve
        :type epochs_no_improve: int
        :param loss_weight: whether to use loss weights or not
        :type loss_weight: bool
        :param path_to_dc: absolute path to deductive closure ontology, need to provide if filtered negative sampling strategy is chosen or random_neg_fraction is less than 1
        :type path_to_dc: str
        :param random_neg_fraction: the fraction of random negatives (the rest negatives are sampled from the deductive closure), should be between 0 and 1
        :type random_neg_fraction: float/int
        """

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        no_improve = 0
        best_loss = float("inf")

        go_classes = np.arange(
            self.go_threshold, max(list(self.class_index_dict.values())) + 1
        )[:-2]
        protein_classes = np.arange(0, self.go_threshold)

        if path_to_dc is not None:
            train_dataloader = OntologyDataLoader(
                self.training_datasets["gci0"][:],
                self.training_datasets["gci1"][:],
                self.training_datasets["gci2"][:],
                self.training_datasets["gci3"][:],
                self.training_datasets["gci0_bot"][:],
                self.training_datasets["gci1_bot"][:],
                self.training_datasets["gci3_bot"][:],
                self.batch_size,
                go_classes,
                protein_classes,
                self.go_threshold,
                self.device,
                negative_mode="filtered",
                path_to_dc=path_to_dc,
                class_index_dict=self.class_index_dict,
                object_property_index_dict=self.object_property_index_dict,
                random_neg_fraction=random_neg_fraction,
            )
        else:
            train_dataloader = OntologyDataLoader(
                self.training_datasets["gci0"][:],
                self.training_datasets["gci1"][:],
                self.training_datasets["gci2"][:],
                self.training_datasets["gci3"][:],
                self.training_datasets["gci0_bot"][:],
                self.training_datasets["gci1_bot"][:],
                self.training_datasets["gci3_bot"][:],
                self.batch_size,
                go_classes,
                protein_classes,
                self.go_threshold,
                self.device,
                negative_mode="random",
                class_index_dict=self.class_index_dict,
                object_property_index_dict=self.object_property_index_dict,
                random_neg_fraction=random_neg_fraction,
            )
        num_steps = train_dataloader.num_steps
        steps = train_dataloader.steps
        all_steps = sum(list(steps.values()))
        if loss_weight:
            weights = [steps[k] / all_steps for k in steps.keys()]
        else:
            weights = [1] * 7

        for epoch in trange(self.epochs):
            self.module.train()

            train_loss = 0

            for batch in train_dataloader:
                cur_loss = 0
                (
                    gci0,
                    gci0_neg,
                    gci1,
                    gci1_neg,
                    gci2,
                    gci2_neg,
                    gci3,
                    gci3_neg,
                    gci0_bot,
                    gci0_bot_neg,
                    gci1_bot,
                    gci1_bot_neg,
                    gci3_bot,
                    gci3_bot_neg,
                ) = batch
                if len(gci0) > 0:
                    pos_loss = self.module(gci0, "gci0")
                    if "gci0" not in self.neg_losses:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci0_neg, "gci0", neg=True)
                        )
                    cur_loss += weights[0] * l
                if len(gci1) > 0:
                    pos_loss = self.module(gci1, "gci1")
                    if "gci1" not in self.neg_losses:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci1_neg, "gci1", neg=True)
                        )
                    cur_loss += weights[1] * l
                if len(gci2) > 0:
                    pos_loss = self.module(gci2, "gci2")
                    if "gci2" not in self.neg_losses:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci2_neg, "gci2", neg=True)
                        )
                    cur_loss += weights[2] * l
                if len(gci3) > 0:
                    pos_loss = self.module(gci3, "gci3")
                    if "gci3" not in self.neg_losses:
                        l = th.mean(pos_loss)
                    else:
                        l = th.mean(pos_loss) + th.mean(
                            self.module(gci3_neg, "gci3", neg=True)
                        )
                    cur_loss += weights[3] * l
                if len(gci0_bot) > 0:
                    pos_loss = self.module(gci0_bot, "gci0_bot")
                    l = th.mean(pos_loss)
                    cur_loss += weights[4] * l
                if len(gci1_bot) > 0:
                    pos_loss = self.module(gci1_bot, "gci1_bot")
                    l = th.mean(pos_loss)
                    cur_loss += weights[5] * l
                if len(gci3_bot) > 0:
                    pos_loss = self.module(gci3_bot, "gci3_bot")
                    l = th.mean(pos_loss)
                    cur_loss += weights[6] * l
                train_loss += cur_loss.detach().item()
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

            train_loss /= num_steps

            loss = 0
            with th.no_grad():
                self.module.eval()
                valid_loss = 0
                gci2_data = self.validation_datasets["gci2"][:]
                loss = th.mean(self.module(gci2_data, "gci2"))
                valid_loss += loss.detach().item()
                scheduler.step(valid_loss)

            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
                print(f"Best loss: {best_loss}, epoch: {epoch}")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve == epochs_no_improve:
                print(f"Stopped at epoch {epoch}")
                break

    def eval_method(self, data):
        """
        Evaluation method

        :param data: data for evaluation
        :type data: torch.Tensor(torch.int64)
        """

        return self.module.eval_method(data)
