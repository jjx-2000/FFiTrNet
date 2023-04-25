# generate atom features

from typing import Callable, List, Union
import numpy as np

from rdkit import Chem
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot

Atom = Chem.rdchem.Atom
AtomFeaturesGenerator = Callable[[Union[Atom, str]], np.ndarray]

ATOM_FEATURES_GENERATOR_REGISTRY = {}


def one_hot_encoding(value: int, choices: List) -> List:
    """
    Apply one hot encoding
    :param value:
    :param choices:
    :return: A one-hot encoding for given index and length
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def register_atom_features_generator(features_generator_name: str) \
                                    -> Callable[[AtomFeaturesGenerator], AtomFeaturesGenerator]:
    """
    Creates a decorator which registers a atom feature generator in global dictionaries to enable access by nome.

    :param features_generator_name: The name to use to access the features generator
    :return: A decorator which will add a atom features generator to the registry using the specified name
    """
    def decorator(features_generator: AtomFeaturesGenerator) -> AtomFeaturesGenerator:
        ATOM_FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> AtomFeaturesGenerator:
    """
    Gets a registered features generator by name.

    :param features_generator_name: The name of the features generator.
    :return: The desired features generator.
    """
    if features_generator_name not in ATOM_FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found.')

    return ATOM_FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns a list of names of available features generators."""
    return list(ATOM_FEATURES_GENERATOR_REGISTRY.keys())

@register_atom_features_generator('atom_type')
def atom_type_features_generator(atom: Atom) -> List:
    atom_type_choices = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 28, 29, 
    30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53]

    atom_type_value = atom.GetAtomicNum()
    return one_hot_encoding(atom_type_value, atom_type_choices)

@register_atom_features_generator('degree')
def degree_features_generator(atom: Atom) -> List:
    degree_choices = list(range(5))
    degree = atom.GetTotalDegree()
    return one_hot_encoding(degree, degree_choices)


@register_atom_features_generator('chiral_tag')
def chiral_tag_features_generator(atom: Atom) -> List:
    chiral_tag_choices = list(range(len(Chem.ChiralType.names)-1))
    chiral_tag = atom.GetChiralTag()
    return one_hot_encoding(chiral_tag, chiral_tag_choices)


@register_atom_features_generator('num_Hs')
def num_Hs_features_generator(atom: Atom) -> List:
    num_Hs_choices = list(range(5))
    num_Hs = atom.GetTotalNumHs()
    return one_hot_encoding(num_Hs, num_Hs_choices)


@register_atom_features_generator('hybridization')
def hybridization_features_generator(atom: Atom) -> List:
    hybridization_choices = list(range(len(Chem.HybridizationType.names)-1))
    hybridization = int(atom.GetHybridization())
    return one_hot_encoding(hybridization, hybridization_choices)


@register_atom_features_generator('aromatic')
def aromatic_features_generator(atom: Atom) -> List:
    return [1 if atom.GetIsAromatic() else 0]


@register_atom_features_generator('mass')
def mass_features_generator(atom: Atom) -> List:
    return [atom.GetMass()]


@register_atom_features_generator('hydrogen_bond')
def hydrogen_bond_features_generator(mol: Chem.Mol, atom: Atom) -> List:
    h_bond_infos = construct_hydrogen_bonding_info(mol)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    return acceptor_donor



def atom_features_union(mol: Chem.Mol, atom: Atom, generator_name_list: List) -> np.ndarray:
    """
    Concatenate the features generated by all generators in ATOM_FEATURES_GENERATOR_REGISTRY except position
    :param mol:
    :param atom: A RDKit atom
    :param generator_name_list: A list of generate name
    :return: a vector of atom features
    """
    atomFeatures = []
    for generator_name in generator_name_list:
        if generator_name in get_available_features_generators():
            if generator_name == 'hydrogen_bond':
                generator = get_features_generator(generator_name)
                atomFeatures += generator(mol, atom)
            else:
                generator = get_features_generator(generator_name)
                atomFeatures += generator(atom)
        else:
            raise KeyError(f'The generator {generator_name} is not in the generator list')
    return np.array(atomFeatures)