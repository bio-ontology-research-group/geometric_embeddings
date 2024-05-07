import mowl

mowl.init_jvm("150g", "1g", 100)

from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from uk.ac.manchester.cs.owl.owlapi import OWLSubClassOfAxiomImpl
from java.util import HashSet
from mowl.ontology.normalize import ELNormalizer
from org.semanticweb.owlapi.model import IRI
import re
import numpy as np
import os


def precompute_gci0_dc(data_path):
    """
    Compute GCI0 (`C \sqsubseteq D`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, "ontology.owl"))

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager

    reasoner_factory = ElkReasonerFactory()
    reasoner = reasoner_factory.createReasoner(dataset.ontology)
    new_gci0_axioms = HashSet()

    gci0_dict = {k: [] for k in list(dataset.ontology.getClassesInSignature())}
    for cl in list(dataset.ontology.getClassesInSignature()):
        superclasses = list(reasoner.getSuperClasses(cl, False).getFlattened())
        subclasses = list(reasoner.getSubClasses(cl, False).getFlattened())
        for elem in superclasses:
            if elem not in gci0_dict[cl]:
                gci0_dict[cl].append(elem)
        for elem in subclasses:
            try:
                if cl not in gci0_dict[elem]:
                    gci0_dict[elem].append(cl)
            except:
                if elem not in gci0_dict.keys():
                    gci0_dict[elem] = [cl]
                else:
                    gci0_dict[elem].append(cl)

    for cl in gci0_dict.keys():
        for super_cl in gci0_dict[cl]:
            new_gci0_axioms.add(OWLSubClassOfAxiomImpl(cl, super_cl, []))

    for ax in list(train_norm["gci0"]):
        new_gci0_axioms.add(ax.owl_axiom)

    ppi_new_gci0_train = adapter.create_ontology("http://gci0_ontology")
    manager.addAxioms(ppi_new_gci0_train, new_gci0_axioms)
    adapter.owl_manager.saveOntology(
        ppi_new_gci0_train,
        IRI.create("file://" + os.path.join(data_path, "gci0_ontology.owl")),
    )
    return


def get_gci0_dict(data_path):
    """
    Create a dictionary `class: its superclasses`

    :param data_path: absolute filepath to the folder containing ontology with GCI0 deductive closure
    :type data_path: str
    :return gci0_dict: a dictionary `class: its superclasses`
    :type gci0_dict: dict(org.semanticweb.owlapi.model.OWLClass, list(org.semanticweb.owlapi.model.OWLClass))
    """

    gci0_dataset = PathDataset(os.path.join(data_path, "gci0_ontology.owl"))
    gci0_dict = {}
    for ax in list(gci0_dataset.ontology.getAxioms()):
        if "SubClassOf" not in str(ax):
            continue
        else:
            classes = list(ax.getClassesInSignature())
            str_ax = re.split("SubClassOf| ", str(ax))[1:]
            if str(classes[0]) == str_ax[0][1:]:
                if classes[0] not in gci0_dict.keys():
                    gci0_dict[classes[0]] = [classes[1]]
                else:
                    gci0_dict[classes[0]].append(classes[1])
            elif str(classes[1]) == str_ax[0][1:]:
                if classes[1] not in gci0_dict.keys():
                    gci0_dict[classes[1]] = [classes[0]]
                else:
                    gci0_dict[classes[1]].append(classes[0])
    return gci0_dict


def get_inv_gci0_dict(gci0_dict):
    """
    Create a dictionary `class: its subclasses`

    :param gci0_dict: dictionary `class: its superclasses`
    :type gci0_dict: dict(org.semanticweb.owlapi.model.OWLClass, list(org.semanticweb.owlapi.model.OWLClass))
    :return inv_gci0_dict: a dictionary `class: its subclasses`
    :type inv_gci0_dict: dict(org.semanticweb.owlapi.model.OWLClass, list(org.semanticweb.owlapi.model.OWLClass))
    """

    inv_gci0_dict = {}
    for k in gci0_dict.keys():
        for v in gci0_dict[k]:
            if v not in inv_gci0_dict.keys():
                inv_gci0_dict[v] = [k]
            else:
                inv_gci0_dict[v].append(k)
    return inv_gci0_dict


def precompute_gci1_dc(data_path):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, "ontology.owl"))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci1_axioms = HashSet()
    gci1_extracted = [elem.owl_axiom for elem in train_norm["gci1"]]

    for ax in gci1_extracted:
        classes = np.array(list(ax.getClassesInSignature()))
        str_ax = re.split("SubClassOf|ObjectIntersectionOf|\(| |\)|\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ""]
        ixs = []
        for cl in classes:
            ixs.append(str_ax.index(str(cl)))
        classes = classes[ixs]
        C = classes[0]
        D = classes[1]
        E = classes[2]
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
            for c_subclass in C_subclasses:
                new_gci1_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(c_subclass, D), E
                    )
                )
        if D in inv_gci0_dict.keys():
            D_subclasses = inv_gci0_dict[D]
            for d_subclass in D_subclasses:
                new_gci1_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(C, d_subclass), E
                    )
                )
        if E in gci0_dict.keys():
            E_superclasses = gci0_dict[E]
            for e_superclass in E_superclasses:
                new_gci1_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(C, D), e_superclass
                    )
                )

    ppi_new_gci1_train = adapter.create_ontology("http://gci1_ontology")
    manager.addAxioms(ppi_new_gci1_train, new_gci1_axioms)
    adapter.owl_manager.saveOntology(
        ppi_new_gci1_train,
        IRI.create("file://" + os.path.join(data_path, "gci1_ontology.owl")),
    )
    return


def precompute_gci2_dc(data_path):
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, "ontology.owl"))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci2_axioms = HashSet()
    gci2_extracted = [elem.owl_axiom for elem in train_norm["gci2"]]

    for ax in gci2_extracted:
        classes = np.array(list(ax.getClassesInSignature()))
        R = list(ax.getObjectPropertiesInSignature())[0]
        str_ax = re.split("SubClassOf|ObjectSomeValuesFrom|\(| |\)|\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ""]
        ixs = []
        for cl in classes:
            ixs.append(str_ax.index(str(cl)))
        ixs = list(map(lambda x: x if x == 0 else x - 1, ixs))
        classes = classes[ixs]
        C = classes[0]
        D = classes[1]
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
            for c_subclass in C_subclasses:
                new_gci2_axioms.add(
                    adapter.create_subclass_of(
                        c_subclass, adapter.create_object_some_values_from(R, D)
                    )
                )
        if D in gci0_dict.keys():
            D_superclasses = gci0_dict[D]
            for d_superclass in D_superclasses:
                new_gci2_axioms.add(
                    adapter.create_subclass_of(
                        C, adapter.create_object_some_values_from(R, d_superclass)
                    )
                )

    ppi_new_gci2_train = adapter.create_ontology("http://gci2_ontology")
    manager.addAxioms(ppi_new_gci2_train, new_gci2_axioms)
    adapter.owl_manager.saveOntology(
        ppi_new_gci2_train,
        IRI.create("file://" + os.path.join(data_path, "gci2_ontology.owl")),
    )
    return


def precompute_gci3_dc(data_path):
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, "ontology.owl"))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci3_axioms = HashSet()
    gci3_extracted = [elem.owl_axiom for elem in train_norm["gci3"]]

    for ax in gci3_extracted:
        classes = np.array(list(ax.getClassesInSignature()))
        R = list(ax.getObjectPropertiesInSignature())[0]
        str_ax = re.split("SubClassOf|ObjectSomeValuesFrom|\(| |\)|\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ""]
        ixs = []
        for cl in classes:
            ixs.append(str_ax.index(str(cl)))
        ixs = list(map(lambda x: x if x == 0 else x - 1, ixs))
        classes = classes[ixs]
        C = classes[0]
        D = classes[1]
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
            for c_subclass in C_subclasses:
                new_gci3_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_some_values_from(R, c_subclass), D
                    )
                )
        if D in gci0_dict.keys():
            D_superclasses = gci0_dict[D]
            for d_superclass in D_superclasses:
                new_gci3_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_some_values_from(R, C), d_superclass
                    )
                )

    ppi_new_gci3_train = adapter.create_ontology("http://gci3_ontology")
    manager.addAxioms(ppi_new_gci3_train, new_gci3_axioms)
    adapter.owl_manager.saveOntology(
        ppi_new_gci3_train,
        IRI.create("file://" + os.path.join(data_path, "gci3_ontology.owl")),
    )
    return


def precompute_gci1_bot_dc(data_path):
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) deductive closure

    :param data_path: absolute filepath to the folder containing train ontology
    :type data_path: str
    """

    dataset = PathDataset(os.path.join(data_path, "ontology.owl"))

    gci0_dict = get_gci0_dict(data_path)
    inv_gci0_dict = get_inv_gci0_dict(gci0_dict)

    elnorm = ELNormalizer()
    train_norm = elnorm.normalize(dataset.ontology)

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    new_gci1_bot_axioms = HashSet()
    gci1_bot_extracted = [elem.owl_axiom for elem in train_norm["gci1_bot"]]

    for ax in gci1_bot_extracted:
        classes = np.array(list(ax.getClassesInSignature()))
        str_ax = re.split("SubClassOf|ObjectIntersectionOf|\(| |\)|\)", str(ax))
        str_ax = [elem for elem in str_ax if elem != ""]
        ixs = []
        for cl in classes:
            ixs.append(str_ax.index(str(cl)))
        classes = classes[ixs]
        C = classes[0]
        D = classes[1]
        E = classes[2]
        if C in inv_gci0_dict.keys():
            C_subclasses = inv_gci0_dict[C]
            for c_subclass in C_subclasses:
                new_gci1_bot_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(c_subclass, D), E
                    )
                )
        if D in inv_gci0_dict.keys():
            D_subclasses = inv_gci0_dict[D]
            for d_subclass in D_subclasses:
                new_gci1_bot_axioms.add(
                    adapter.create_subclass_of(
                        adapter.create_object_intersection_of(C, d_subclass), E
                    )
                )

    ppi_new_gci1_bot_train = adapter.create_ontology("http://gci1_bot_ontology")
    manager.addAxioms(ppi_new_gci1_bot_train, new_gci1_bot_axioms)
    adapter.owl_manager.saveOntology(
        ppi_new_gci1_bot_train,
        IRI.create("file://" + os.path.join(data_path, "gci1_bot_ontology.owl")),
    )
    return
