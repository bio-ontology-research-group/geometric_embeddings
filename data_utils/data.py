import mowl

mowl.init_jvm("8g", "1g", 8)

from mowl.datasets import PathDataset
from mowl.datasets.base import OWLClasses


class PPIYeastDataset(PathDataset):
    """
    Yeast iw (interacts_with) dataset class
    """

    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """
        This function returns a tuple of head `h` and tail `t` entities for evaluation triples (h, r, t); in this particular case both
        head and tail entities are the whole set of protein classes

        :return self._evaluation_entities: head and tail entities for evaluation triples
        :type self._evaluation_entities: tuple(mowl.datasets.base.OWLClasses)
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "4932." in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        """
        This function returns relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set;
        in case of Yeast iw dataset it returns `http://interacts_with`

        :return: relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set
        :type: str
        """

        return "http://interacts_with"


class PPIHumanDataset(PathDataset):
    """
    Human iw (interacts_with) dataset class
    """

    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """
        This function returns a tuple of head `h` and tail `t` entities for evaluation triples (h, r, t); in this particular case both 
        head and tail entities are the whole set of protein classes

        :return self._evaluation_entities: head and tail entities for evaluation triples
        :type self._evaluation_entities: tuple(mowl.datasets.base.OWLClasses)
        """

        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "9606." in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        """
        This function returns relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set; 
        in case of Human iw dataset it returns `http://interacts_with`

        :return: relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set
        :type: str
        """

        return "http://interacts_with"


class AFPYeastDataset(PathDataset):
    """
    Yeast hf (has_function) dataset class
    """

    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """
        This function returns a tuple of head `h` and tail `t` entities for evaluation triples (h, r, t); in this particular case
        head entities are the whole set of protein classes and tail entities are the whole set of GO functions

        :return self._evaluation_entities: head and tail entities for evaluation triples
        :type self._evaluation_entities: tuple(mowl.datasets.base.OWLClasses)
        """

        if self._evaluation_classes is None:
            proteins = set()
            gos = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "4932." in owl_name:
                    proteins.add(owl_cls)
                elif "GO_" in owl_name:
                    gos.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(gos)

        return self._evaluation_classes

    def get_evaluation_property(self):
        """
        This function returns relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set; 
        in case of Yeast hf dataset it returns `http://has_function`

        :return: relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set
        :type: str
        """

        return "http://has_function"


class AFPHumanDataset(PathDataset):
    """
    Human hf (has_function) dataset class
    """

    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """
        This function returns a tuple of head `h` and tail `t` entities for evaluation triples (h, r, t); in this particular case
        head entities are the whole set of protein classes and tail entities are the whole set of GO functions

        :return self._evaluation_entities: head and tail entities for evaluation triples
        :type self._evaluation_entities: tuple(mowl.datasets.base.OWLClasses)
        """

        if self._evaluation_classes is None:
            proteins = set()
            gos = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "9606." in owl_name:
                    proteins.add(owl_cls)
                elif "GO_" in owl_name:
                    gos.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(gos)

        return self._evaluation_classes

    def get_evaluation_property(self):
        """
        This function returns relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set; 
        in case of Human hf dataset it returns `http://has_function`

        :return: relation name `R` for axioms of the form `C \sqsubseteq \exists R.D` from validation and test set
        :type: str
        """

        return "http://has_function"
