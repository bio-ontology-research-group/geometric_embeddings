import mowl

mowl.init_jvm("100g", "1g", 8)

import numpy as np


class NaiveClassifier:
    """
    Naive classifier 

    :param training_set: a set of triples that are true positives from the train dataset
    :type training_set: list(mowl.projection.edge.Edge)
    :param head_entities: list of entities that are used as head entities in the testing set.
    :type head_entities: list(str)
    :param tail_entities: list of entities that are used as tail entities in the testing set.
    :type tail_entities: list(str)
    :param class_ix_dict: dictionary of classes and their embeddings
    :type class_ix_dict: dict(str, numpy.array)
    :param go_threshold: value from class_index_dict starting from which GO classes are encoded, need to provide for function prediction datasets
    :type go_threshold: int
    """

    def __init__(
        self,
        training_set,
        head_entities,
        tail_entities,
        class_ix_dict,
        go_threshold=None,
    ):
        self.training_set = training_set
        self.head_entities = head_entities
        self.tail_entities = tail_entities
        self.class_ix_dict = class_ix_dict
        self.go_threshold = go_threshold
        self.matrix = None
        self.training_set = [x.astuple() for x in training_set]

    def create_matrix(self, symmetric=False):
        """
        Create M_r(c, d) matrix

        :param symmetric: whether the matrix should be symmetric or not, used for protein-protein interaction datasets
        :type symmetric: bool
        """

        self.matrix = np.zeros(
            (len(self.head_entities), len(self.tail_entities)), dtype=np.int32
        )

        # interacts_with(P1, P2)
        if self.go_threshold is None:
            for c, _, d in self.training_set:
                c_emb = self.class_ix_dict[c]
                d_emb = self.class_ix_dict[d]
                self.matrix[c_emb, d_emb] = 1
                if symmetric:
                    self.matrix[d_emb, c_emb] = 1

        # has_function(P, GO)
        else:
            for c, _, d in self.training_set:
                c_emb = self.class_ix_dict[c]
                d_emb = self.class_ix_dict[d] - self.go_threshold
                self.matrix[c_emb, d_emb] = 1

    def predict(self):
        """
        Get naive predictions from the matrix M_r(c, d); need to compute the matrix first

        :return evaluation_properties: naive predictions
        :type evaluation_properties: numpy.array(numpy.int32)
        """

        if self.matrix is None:
            raise ValueError("Compute prediction matrix first using create_matrix()")

        evaluation_properties = np.sum(self.matrix, axis=0)
        return evaluation_properties