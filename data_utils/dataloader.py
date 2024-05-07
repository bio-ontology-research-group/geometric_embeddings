import mowl

mowl.init_jvm("8g", "1g", 8)

import math
import numpy as np
import torch as th
from mowl.datasets import PathDataset
from mowl.datasets.el import ELDataset


class OntologyDataLoader:
    """
    Custom dataloader

    :param gci0: a set of axioms of type `C \sqsubseteq D`
    :type gci0: torch.Tensor(torch.int64)
    :param gci1: a set of axioms of type `C \sqcap D \sqsubseteq E`
    :type gci1: torch.Tensor(torch.int64)
    :param gci2: a set of axioms of type `C \sqsubseteq \exists R.D`
    :type gci2: torch.Tensor(torch.int64)
    :param gci3: a set of axioms of type `\exists R.C \sqsubseteq D`
    :type gci3: torch.Tensor(torch.int64)
    :param gci0_bot: a set of axioms of type `C \sqsubseteq \bot`
    :type gci0_bot: torch.Tensor(torch.int64)
    :param gci1_bot: a set of axioms of type `C \sqcap D \sqsubseteq \bot`
    :type gci1_bot: torch.Tensor(torch.int64)
    :param gci3_bot: a set of axioms of type `\exists R.C \sqsubseteq \bot`
    :type gci3_bot: torch.Tensor(torch.int64)
    :param batch_size: batch size
    :type batch_size: int
    :param go_classes: GO classes embeddings
    :type go_classes: numpy.array(numpy.int64)
    :param prot_classes: protein classes embeddings
    :type prot_classes: numpy.array(numpy.int64)
    :param go_threshold: value from class_index_dict starting from which GO classes are encoded
    :type go_threshold: int
    :param device: device name, `cuda` or `cpu`
    :type device: str
    :param negative_mode: negative sampling strategy, `random` for random sampling, `filtered` for filtering using deductive closure
    :type negative_mode: str
    :param path_to_dc: absolute filepath to deductive closure ontology
    :type path_to_dc: str
    :param class_index_dict: dictionary of classes and their embeddings
    :type class_index_dict: dict(str, numpy.array)
    :param object_property_index_dict: dictionary of relations and their embeddings
    :type object_property_index_dict: dict(str, numpy.array)
    :param random_negative_fraction: the fraction of random negatives (the rest negatives are sampled from the deductive closure), should be between 0 and 1
    :type random_negative_fraction: float/int
    """

    def __init__(
        self,
        gci0,
        gci1,
        gci2,
        gci3,
        gci0_bot,
        gci1_bot,
        gci3_bot,
        batch_size,
        go_classes,
        prot_classes,
        go_threshold,
        device,
        negative_mode="random",
        path_to_dc=None,
        class_index_dict=None,
        object_property_index_dict=None,
        random_neg_fraction=1,
    ):
        self.gci0 = gci0
        self.gci1 = gci1
        self.gci2 = gci2
        self.gci3 = gci3
        self.gci0_bot = gci0_bot
        self.gci1_bot = gci1_bot
        self.gci3_bot = gci3_bot
        self.batch_size = batch_size
        self.go_classes = go_classes
        self.prot_classes = prot_classes
        self.go_threshold = go_threshold
        self.device = device
        self.class_index_dict = class_index_dict
        self.random_neg_fraction = random_neg_fraction
        if negative_mode in ["random", "filtered"]:
            self.negative_mode = negative_mode
        else:
            raise ValueError('"negative_mode" should be one of ["random", "filtered"]')
        self.steps = {
            "gci0": int(math.ceil(len(gci0) / batch_size)),
            "gci1": int(math.ceil(len(gci1) / batch_size)),
            "gci2": int(math.ceil(len(gci2) / batch_size)),
            "gci3": int(math.ceil(len(gci3) / batch_size)),
            "gci0_bot": int(math.ceil(len(gci0_bot) / batch_size)),
            "gci1_bot": int(math.ceil(len(gci1_bot) / batch_size)),
            "gci3_bot": int(math.ceil(len(gci3_bot) / batch_size)),
        }
        self.remainders = {
            "gci0": 0,
            "gci1": 0,
            "gci2": 0,
            "gci3": 0,
            "gci0_bot": 0,
            "gci1_bot": 0,
            "gci3_bot": 0,
        }
        self.num_steps = max(self.steps.values())
        self.path_to_dc = path_to_dc
        if self.negative_mode == "filtered":
            self.deductive_closure = ELDataset(
                PathDataset(self.path_to_dc).ontology,
                class_index_dict,
                object_property_index_dict,
                device=self.device,
            ).get_gci_datasets()

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        return self.next()

    def generate_negatives(self, gci_name, pos_batch):
        """
        Function for negative batch sampling

        :param gci_name: GCI abbreviation, one of [`gci0`, `gci1`, `gci2`, `gci3`, `gci0_bot`, `gci1_bot`, `gci3_bot`]
        :type gci_name: str
        :param pos_batch: batch of positives
        :type pos_batch: torch.Tensor(torch.int64)
        :return neg_batch: batch of negatives
        :type neg_batch: torch.Tensor(torch.int64)
        """

        np.random.seed(0)
        if gci_name == "gci0":
            if self.random_neg_fraction >= 1:
                data = pos_batch[:, 1]
            else:
                data = pos_batch[
                    : math.floor(self.random_neg_fraction * pos_batch.size()[0]), 1
                ]
                dc_gci0 = self.deductive_closure["gci0"][:].cpu().numpy()
                pos_neg_data = (
                    pos_batch[
                        math.floor(self.random_neg_fraction * pos_batch.size()[0]) :, :1
                    ]
                    .cpu()
                    .numpy()
                )
                to_sample_from = dc_gci0[
                    np.isin(dc_gci0[:, :1], pos_neg_data).all(axis=1)
                ]
                num_to_sample = pos_neg_data.shape[0]
                part2 = to_sample_from[
                    np.random.choice(
                        to_sample_from.shape[0], num_to_sample, replace=False
                    )
                ]
            go_mask = data >= self.go_threshold
            prot_mask = data < self.go_threshold
            go_part = data * go_mask
            prot_part = data * prot_mask
            go_ixs = go_mask.nonzero().squeeze(1)
            prot_ixs = prot_mask.nonzero().squeeze(1)
            prot_corrupted = np.random.choice(
                self.prot_classes, size=len(prot_ixs), replace=True
            )
            prot_part[prot_ixs] = th.tensor(prot_corrupted).to(self.device)
            go_ixs = go_mask.nonzero().squeeze(1)
            go_corrupted = np.random.choice(
                self.go_classes, size=len(go_ixs), replace=True
            )
            go_part[go_ixs] = th.tensor(go_corrupted).to(self.device)
            corrupted = go_part + prot_part
            if self.random_neg_fraction >= 1:
                neg_batch = th.cat([pos_batch[:, :1], corrupted.unsqueeze(1)], dim=1)
            else:
                part1 = th.cat(
                    [
                        pos_batch[
                            : math.floor(
                                self.random_neg_fraction * pos_batch.size()[0]
                            ),
                            :1,
                        ],
                        corrupted.unsqueeze(1),
                    ],
                    dim=1,
                )
                neg_batch = th.cat([part1, th.tensor(part2).to(self.device)], dim=0)
            if self.negative_mode == "filtered" and self.random_neg_fraction >= 1:
                arr_max = np.maximum(
                    neg_batch.cpu().numpy().max(0) + 1,
                    self.deductive_closure["gci0"][:].cpu().numpy().max(0) + 1,
                )
                neg_batch = neg_batch[
                    ~np.isin(
                        neg_batch.cpu().numpy().dot(arr_max),
                        self.deductive_closure["gci0"][:].cpu().numpy().dot(arr_max),
                    )
                ]
        elif gci_name in ["gci1", "gci2", "gci3"]:
            if self.random_neg_fraction >= 1:
                data = pos_batch[:, 2]
            else:
                data = pos_batch[
                    : math.floor(self.random_neg_fraction * pos_batch.size()[0]), 2
                ]
                if gci_name == "gci1":
                    dc_gci = self.deductive_closure["gci1"][:].cpu().numpy()
                elif gci_name == "gci2":
                    dc_gci = self.deductive_closure["gci2"][:].cpu().numpy()
                elif gci_name == "gci3":
                    dc_gci = self.deductive_closure["gci3"][:].cpu().numpy()
                pos_neg_data = (
                    pos_batch[
                        math.floor(self.random_neg_fraction * pos_batch.size()[0]) :, :2
                    ]
                    .cpu()
                    .numpy()
                )
                to_sample_from = dc_gci[
                    np.isin(dc_gci[:, :2], pos_neg_data).all(axis=1)
                ]
                num_to_sample = pos_neg_data.shape[0]
                try:
                    part2 = to_sample_from[
                        np.random.choice(
                            to_sample_from.shape[0], num_to_sample, replace=False
                        )
                    ]
                except:
                    part2 = to_sample_from[
                        np.random.choice(
                            to_sample_from.shape[0], num_to_sample, replace=True
                        )
                    ]
            go_mask = data >= self.go_threshold
            prot_mask = data < self.go_threshold
            go_part = data * go_mask
            prot_part = data * prot_mask
            prot_ixs = prot_mask.nonzero().squeeze(1)
            prot_corrupted = np.random.choice(
                self.prot_classes, size=len(prot_ixs), replace=True
            )
            prot_part[prot_ixs] = th.tensor(prot_corrupted).to(self.device)
            go_ixs = go_mask.nonzero().squeeze(1)
            go_corrupted = np.random.choice(
                self.go_classes, size=len(go_ixs), replace=True
            )
            go_part[go_ixs] = th.tensor(go_corrupted).to(self.device)
            corrupted = go_part + prot_part
            if self.random_neg_fraction >= 1:
                neg_batch = th.cat([pos_batch[:, :2], corrupted.unsqueeze(1)], dim=1)
            else:
                part1 = th.cat(
                    [
                        pos_batch[
                            : math.floor(
                                self.random_neg_fraction * pos_batch.size()[0]
                            ),
                            :2,
                        ],
                        corrupted.unsqueeze(1),
                    ],
                    dim=1,
                )
                neg_batch = th.cat([part1, th.tensor(part2).to(self.device)], dim=0)
            if self.negative_mode == "filtered" and self.random_neg_fraction >= 1:
                if gci_name == "gci1":
                    arr_max = np.maximum(
                        neg_batch.cpu().numpy().max(0) + 1,
                        self.deductive_closure["gci1"][:].cpu().numpy().max(0) + 1,
                    )
                    neg_batch = neg_batch[
                        ~np.isin(
                            neg_batch.cpu().numpy().dot(arr_max),
                            self.deductive_closure["gci1"][:]
                            .cpu()
                            .numpy()
                            .dot(arr_max),
                        )
                    ]
                elif gci_name == "gci2":
                    arr_max = np.maximum(
                        neg_batch.cpu().numpy().max(0) + 1,
                        self.deductive_closure["gci2"][:].cpu().numpy().max(0) + 1,
                    )
                    neg_batch = neg_batch[
                        ~np.isin(
                            neg_batch.cpu().numpy().dot(arr_max),
                            self.deductive_closure["gci2"][:]
                            .cpu()
                            .numpy()
                            .dot(arr_max),
                        )
                    ]
                elif gci_name == "gci3":
                    arr_max = np.maximum(
                        neg_batch.cpu().numpy().max(0) + 1,
                        self.deductive_closure["gci3"][:].cpu().numpy().max(0) + 1,
                    )
                    neg_batch = neg_batch[
                        ~np.isin(
                            neg_batch.cpu().numpy().dot(arr_max),
                            self.deductive_closure["gci3"][:]
                            .cpu()
                            .numpy()
                            .dot(arr_max),
                        )
                    ]
        elif gci_name in ["gci0_bot", "gci1_bot"]:
            if self.random_neg_fraction >= 1:
                data = pos_batch[:, 0]
            else:
                data = pos_batch[
                    : math.floor(self.random_neg_fraction * pos_batch.size()[0]), 0
                ]
                if gci_name == "gci0_bot":
                    dc_gci = self.deductive_closure["gci0_bot"][:].cpu().numpy()
                elif gci_name == "gci1_bot":
                    dc_gci = self.deductive_closure["gci1_bot"][:].cpu().numpy()
                pos_neg_data = (
                    pos_batch[
                        math.floor(self.random_neg_fraction * pos_batch.size()[0]) :, 1:
                    ]
                    .cpu()
                    .numpy()
                )
                to_sample_from = dc_gci[
                    np.isin(dc_gci[:, 1:], pos_neg_data).all(axis=1)
                ]
                num_to_sample = pos_neg_data.shape[0]
                part2 = to_sample_from[
                    np.random.choice(
                        to_sample_from.shape[0], num_to_sample, replace=False
                    )
                ]
            go_mask = data >= self.go_threshold
            prot_mask = data < self.go_threshold
            go_part = data * go_mask
            prot_part = data * prot_mask
            prot_ixs = prot_mask.nonzero().squeeze(1)
            prot_corrupted = np.random.choice(
                self.prot_classes, size=len(prot_ixs), replace=True
            )
            prot_part[prot_ixs] = th.tensor(prot_corrupted).to(self.device)
            go_ixs = go_mask.nonzero().squeeze(1)
            go_corrupted = np.random.choice(
                self.go_classes, size=len(go_ixs), replace=True
            )
            go_part[go_ixs] = th.tensor(go_corrupted).to(self.device)
            corrupted = go_part + prot_part
            if self.random_neg_fraction >= 1:
                neg_batch = th.cat([corrupted.unsqueeze(1), pos_batch[:, 1:]], dim=1)
            else:
                part1 = th.cat(
                    [
                        pos_batch[
                            : math.floor(
                                self.random_neg_fraction * pos_batch.size()[0]
                            ),
                            1:,
                        ],
                        corrupted.unsqueeze(1),
                    ],
                    dim=1,
                )
                neg_batch = th.cat([part1, th.tensor(part2).to(self.device)], dim=0)
            if self.negative_mode == "filtered" and self.random_neg_fraction >= 1:
                if gci_name == "gci0_bot":
                    neg_batch = None
                if gci_name == "gci1_bot":
                    arr_max = np.maximum(
                        neg_batch.cpu().numpy().max(0) + 1,
                        self.deductive_closure["gci1_bot"][:].cpu().numpy().max(0) + 1,
                    )
                    neg_batch = neg_batch[
                        ~np.isin(
                            neg_batch.cpu().numpy().dot(arr_max),
                            self.deductive_closure["gci1_bot"][:]
                            .cpu()
                            .numpy()
                            .dot(arr_max),
                        )
                    ]
        elif gci_name == "gci3_bot":
            data = pos_batch[:, 1]
            go_mask = data >= self.go_threshold
            prot_mask = data < self.go_threshold
            go_part = data * go_mask
            prot_part = data * prot_mask
            go_ixs = go_mask.nonzero().squeeze(1)
            prot_ixs = prot_mask.nonzero().squeeze(1)
            prot_corrupted = np.random.choice(
                self.prot_classes, size=len(prot_ixs), replace=True
            )
            prot_part[prot_ixs] = th.tensor(prot_corrupted).to(self.device)
            go_ixs = go_mask.nonzero().squeeze(1)
            go_corrupted = np.random.choice(
                self.go_classes, size=len(go_ixs), replace=True
            )
            go_part[go_ixs] = th.tensor(go_corrupted).to(self.device)
            corrupted = go_part + prot_part
            neg_batch = th.cat(
                [
                    pos_batch[:, :1],
                    corrupted.unsqueeze(1),
                    pos_batch[:, 2:],
                ],
                dim=1,
            )
            if self.negative_mode == "filtered":
                neg_batch = None
        return neg_batch

    def next(self):
        """
        Batch iterator

        :return gci0_batch: a batch of positives for axioms of type `C \sqsubseteq D`
        :type gci0_batch: torch.Tensor(torch.int64)
        :return gci0_neg_batch: a batch of negatives for axioms of type `C \sqsubseteq D`
        :type gci0_neg_batch: torch.Tensor(torch.int64)
        :param gci1: a set of axioms of type `C \sqcap D \sqsubseteq E`
        :type gci1: torch.Tensor(torch.int64)
        :param gci2: a set of axioms of type `C \sqsubseteq \exists R.D`
        :type gci2: torch.Tensor(torch.int64)
        :param gci3: a set of axioms of type `\exists R.C \sqsubseteq D`
        :type gci3: torch.Tensor(torch.int64)
        :param gci0_bot: a set of axioms of type `C \sqsubseteq \bot`
        :type gci0_bot: torch.Tensor(torch.int64)
        :param gci1_bot: a set of axioms of type `C \sqcap D \sqsubseteq \bot`
        :type gci1_bot: torch.Tensor(torch.int64)
        :param gci3_bot: a set of axioms of type `\exists R.C \sqsubseteq \bot`
        :type gci3_bot: torch.Tensor(torch.int64)
        """
        if self.i < self.num_steps:
            if self.gci0.size()[0] > 0:
                j = self.remainders["gci0"]
                gci0_batch = self.gci0[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci0"] = (j + 1) % self.steps["gci0"]
                gci0_neg_batch = self.generate_negatives("gci0", gci0_batch)
            else:
                gci0_batch = self.gci0
                gci0_neg_batch = self.gci0
            if self.gci1.size()[0] > 0:
                j = self.remainders["gci1"]
                gci1_batch = self.gci1[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci1"] = (j + 1) % self.steps["gci1"]
                gci1_neg_batch = self.generate_negatives("gci1", gci1_batch)
            else:
                gci1_batch = self.gci1
                gci1_neg_batch = self.gci1
            if self.gci2.size()[0] > 0:
                j = self.remainders["gci2"]
                gci2_batch = self.gci2[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci2"] = (j + 1) % self.steps["gci2"]
                gci2_neg_batch = self.generate_negatives("gci2", gci2_batch)
            else:
                gci2_batch = self.gci2
                gci2_neg_batch = self.gci2
            if self.gci3.size()[0] > 0:
                j = self.remainders["gci3"]
                gci3_batch = self.gci3[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci3"] = (j + 1) % self.steps["gci3"]
                gci3_neg_batch = self.generate_negatives("gci3", gci3_batch)
            else:
                gci3_batch = self.gci3
                gci3_neg_batch = self.gci3
            if self.gci0_bot.size()[0] > 0:
                j = self.remainders["gci0_bot"]
                gci0_bot_batch = self.gci0_bot[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci0_bot"] = (j + 1) % self.steps["gci0_bot"]
                gci0_bot_neg_batch = self.generate_negatives("gci0_bot", gci0_bot_batch)
            else:
                gci0_bot_batch = self.gci0_bot
                gci0_bot_neg_batch = self.gci0_bot
            if self.gci1_bot.size()[0] > 0:
                j = self.remainders["gci1_bot"]
                gci1_bot_batch = self.gci1_bot[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci1_bot"] = (j + 1) % self.steps["gci1_bot"]
                gci1_bot_neg_batch = self.generate_negatives("gci1_bot", gci1_bot_batch)
            else:
                gci1_bot_batch = self.gci1_bot
                gci1_bot_neg_batch = self.gci1_bot
            if self.gci3_bot.size()[0] > 0:
                j = self.remainders["gci3_bot"]
                gci3_bot_batch = self.gci3_bot[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci3_bot"] = (j + 1) % self.steps["gci3_bot"]
                gci3_bot_neg_batch = self.generate_negatives("gci3_bot", gci3_bot_batch)
            else:
                gci3_bot_batch = self.gci3_bot
                gci3_bot_neg_batch = self.gci3_bot
            self.i += 1
            return (
                gci0_batch,
                gci0_neg_batch,
                gci1_batch,
                gci1_neg_batch,
                gci2_batch,
                gci2_neg_batch,
                gci3_batch,
                gci3_neg_batch,
                gci0_bot_batch,
                gci0_bot_neg_batch,
                gci1_bot_batch,
                gci1_bot_neg_batch,
                gci3_bot_batch,
                gci3_bot_neg_batch,
            )
        else:
            raise StopIteration
