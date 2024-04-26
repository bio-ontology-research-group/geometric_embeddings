import mowl

mowl.init_jvm("8g", "1g", 8)

from mowl.datasets import PathDataset
from mowl.datasets.base import OWLClasses


class PPIYeastDataset(PathDataset):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "4932." in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://interacts_with"


class PPIHumanDataset(PathDataset):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "9606." in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

    def get_evaluation_property(self):
        return "http://interacts_with"


class AFPYeastDataset(PathDataset):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
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
        return "http://has_function"


class AFPHumanDataset(PathDataset):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation"""

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
        return "http://has_function"
