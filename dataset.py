# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import Random
from collections import defaultdict

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Scaffolds import MurckoScaffold

from util import get_task_names


# -------------------------------------
# attentive_fp fashion featurization
# -------------------------------------
def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_attr(mol, explicit_H=False, use_chirality=True, pharmaco=True, scaffold=True):
    if pharmaco:
        mol = tag_pharmacophore(mol)
    if scaffold:
        mol = tag_scaffold(mol)

    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding_unk(atom.GetDegree(),
                                      [0, 1, 2, 3, 4, 5, 'other']) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            # print(one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')])
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        if pharmaco:
            results = results + [int(atom.GetProp('Hbond_donor'))] + [int(atom.GetProp('Hbond_acceptor'))] + \
                      [int(atom.GetProp('Basic'))] + [int(atom.GetProp('Acid'))] + \
                      [int(atom.GetProp('Halogen'))]
        if scaffold:
            results = results + [int(atom.GetProp('Scaffold'))]
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


def bond_break(mol):
    results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.long)

    if results.size == 0:
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(mol, asMols=True, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(mol)

    else:
        bond_to_break = results[:, 0, :]
        bond_to_break = bond_to_break.tolist()

        rwmol = Chem.RWMol(mol)
        for i in bond_to_break:
            rwmol.RemoveBond(*i)
        # with Chem.RWMol(mol) as rwmol:
        #     for i in bond_to_break:
        #         rwmol.RemoveBond(*i)
        rwmol = rwmol.GetMol()
        cluster_idx = []
        Chem.rdmolops.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx)
        fra_edge_index, fra_edge_attr = bond_attr(rwmol)
        cluster_idx = torch.LongTensor(cluster_idx)

    return fra_edge_index, fra_edge_attr, cluster_idx


# ---------------------------------------------
# Scaffold and pharmacophore information utils
# ---------------------------------------------
# tag pharmoco features to each atom
fun_smarts = {
    'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
    'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
    'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
    'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
    'Halogen': '[F,Cl,Br,I]'
}
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def tag_pharmacophore(mol):
    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches:
            match_idxes.extend(mat)
        for i, atom in enumerate(mol.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return mol


# tag scaffold information to each atom
def tag_scaffold(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    match_idxes = mol.GetSubstructMatch(core)
    for i, atom in enumerate(mol.GetAtoms()):
        tag = '1' if i in match_idxes else '0'
        atom.SetProp('Scaffold', tag)
    return mol


# ---------------------------------
# data and dataset
# ---------------------------------
class MolData(Data):
    def __init__(self, fra_edge_index=None, fra_edge_attr=None, cluster_index=None, **kwargs):
        super(MolData, self).__init__(**kwargs)
        self.cluster_index = cluster_index
        self.fra_edge_index = fra_edge_index
        self.fra_edge_attr = fra_edge_attr

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cluster_index':
            return int(self.cluster_index.max()) + 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


class MolDataset(InMemoryDataset):

    def __init__(self, root, dataset, task_type, tasks, logger=None,
                 transform=None, pre_transform=None, pre_filter=None):

        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger

        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        self.logger.info(f'number of all smiles: {len(smilesList)}')
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
                remained_smiles.append(smiles)
            except:
                self.logger.info(f'not successfully processed smiles: {smiles}')
                pass
        self.logger.info(f'number of successfully processed smiles: {len(remained_smiles)}')

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):

            mol = Chem.MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 666
                data.y = torch.LongTensor([label])
                if self.task_type == 'regression':
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        smiles = Chem.MolToSmiles(mol)
        if mol is None: return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        fra_edge_index, fra_edge_attr, cluster_index = bond_break(mol)
        data = MolData(
            x=torch.FloatTensor(node_attr),
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            fra_edge_index=torch.LongTensor(fra_edge_index).t(),
            fra_edge_attr=torch.FloatTensor(fra_edge_attr),
            cluster_index=torch.LongTensor(cluster_index),
            y=None,
            smiles=smiles,
        )
        return data


# ---------------------------------
# load dataset
# ---------------------------------
def load_dataset_all(path, dataset, seed, task_type=None, tasks=None, logger=None):
    pyg_dataset = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks, logger=logger)
    # del pyg_dataset.data.smiles

    logger.info(f'Total smiles = {len(pyg_dataset):,}')

    return pyg_dataset


def load_dataset_random(path, dataset, seed, task_type, tasks=None, logger=None):
    pyg_dataset = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks, logger=logger)
    del pyg_dataset.data.smiles

    # Seed randomness
    random = Random(seed)
    indices = list(range(len(pyg_dataset)))
    random.seed(seed)
    random.shuffle(indices)

    train_size = int(0.8 * len(pyg_dataset))
    val_size = int(0.1 * len(pyg_dataset))
    test_size = len(pyg_dataset) - train_size - val_size

    trn_id, val_id, test_id = indices[:train_size], \
                              indices[train_size:(train_size + val_size)], \
                              indices[(train_size + val_size):]

    trn, val, test = pyg_dataset[torch.LongTensor(trn_id)], \
                     pyg_dataset[torch.LongTensor(val_id)], \
                     pyg_dataset[torch.LongTensor(test_id)]

    logger.info(f'Total smiles = {len(pyg_dataset):,} | '
                f'train smiles = {train_size:,} | '
                f'val smiles = {val_size:,} | '
                f'test smiles = {test_size:,}')

    assert task_type == 'classification' or 'regression'
    if task_type == 'classification':
        weights = []
        for i in range(len(tasks)):
            validId = np.where((pyg_dataset.data.y[:, i] == 0) | (pyg_dataset.data.y[:, i] == 1))[0]
            pos_len = (pyg_dataset.data.y[:, i][validId].sum()).item()
            neg_len = len(pyg_dataset.data.y[:, i][validId]) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        trn.weights = weights
    else:
        trn.weights = None

    return trn, val, test


# anti-noise experiments for hiv dataset
def load_dataset_noise(path, dataset, seed, task_type, tasks, rate, logger=None):
    pyg_dataset = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks, logger=logger)
    del pyg_dataset.data.smiles

    train_size = int(0.8 * len(pyg_dataset))
    val_size = int(0.1 * len(pyg_dataset))
    test_size = len(pyg_dataset) - train_size - val_size

    pyg_dataset, perm = pyg_dataset.shuffle(return_perm=True)
    trn_perm, val_perm = perm[:train_size], perm[train_size:(train_size + val_size)]
    trn_cutoff, val_cutoff = int(train_size * rate), int(val_size * rate)
    trn_noise_perm, val_noise_perm = trn_perm[:trn_cutoff], val_perm[:val_cutoff]
    noise_perm = torch.cat([trn_noise_perm, val_noise_perm])

    # add same rate noise to train set and val set(simply change the label)
    pyg_dataset.data.y[noise_perm] = 1 - pyg_dataset.data.y[noise_perm]

    trn, val, test = pyg_dataset[:train_size], \
                     pyg_dataset[train_size:(train_size + val_size)], \
                     pyg_dataset[(train_size + val_size):]

    logger.info(f'Total smiles = {len(pyg_dataset):,} | '
                f'train smiles = {train_size:,} | '
                f'val smiles = {val_size:,} | '
                f'test smiles = {test_size:,}')

    weights = []
    pos_len = (pyg_dataset.data.y.sum()).item()
    neg_len = len(pyg_dataset) - pos_len
    weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    trn.weights = weights
    logger.info(weights)

    return trn, val, test


def load_dataset_scaffold(path, dataset, seed, task_type, tasks=None, logger=None):
    pyg_dataset = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks, logger=logger)

    trn_id, val_id, test_id, weights = scaffold_split(pyg_dataset, task_type=task_type, tasks=tasks,
                                                      seed=seed, logger=logger)
    del pyg_dataset.data.smiles
    trn, val, test = pyg_dataset[torch.LongTensor(trn_id)], \
                     pyg_dataset[torch.LongTensor(val_id)], \
                     pyg_dataset[torch.LongTensor(test_id)]
    trn.weights = weights

    return trn, val, test


# ---------------------------------------------
# Scaffold utils, copy from chemprop.
# ---------------------------------------------
def generate_scaffold(mol, include_chirality=False):
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold.
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(smiles, use_indices=False):
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).
    :param smiles: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, smi in enumerate(smiles):
        scaffold = generate_scaffold(smi)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smi)

    return scaffolds


def scaffold_split(pyg_dataset, task_type, tasks, sizes=(0.8, 0.1, 0.1), balanced=True, seed=1, logger=None):
    assert sum(sizes) == 1

    # Split
    logger.info('generating scaffold......')
    num = len(pyg_dataset)
    train_size, val_size, test_size = sizes[0] * num, sizes[1] * num, sizes[2] * num
    train_ids, val_ids, test_ids = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(pyg_dataset.data.smiles, use_indices=True)

    # Seed randomness
    random = Random(seed)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train_ids) + len(index_set) <= train_size:
            train_ids += index_set
            train_scaffold_count += 1
        elif len(val_ids) + len(index_set) <= val_size:
            val_ids += index_set
            val_scaffold_count += 1
        else:
            test_ids += index_set
            test_scaffold_count += 1

    logger.info(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                f'train scaffolds = {train_scaffold_count:,} | '
                f'val scaffolds = {val_scaffold_count:,} | '
                f'test scaffolds = {test_scaffold_count:,}')

    logger.info(f'Total smiles = {num:,} | '
                f'train smiles = {len(train_ids):,} | '
                f'val smiles = {len(val_ids):,} | '
                f'test smiles = {len(test_ids):,}')

    assert len(train_ids) + len(val_ids) + len(test_ids) == len(pyg_dataset)

    # Compute weights
    assert task_type == 'classification' or 'regression'
    if task_type == 'classification':
        weights = []
        for i in range(len(tasks)):
            pos_len = (pyg_dataset.data.y[:, i].sum()).item()
            neg_len = len(pyg_dataset) - pos_len
            weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])

    else:
        weights = None

    return train_ids, val_ids, test_ids, weights


# ---------------------------------
# build dataset and dataloader
# ---------------------------------
def build_dataset(args, logger):
    data_path = os.path.join(args.data_dir, args.dataset)
    task_name = get_task_names(os.path.join(data_path, 'raw/{}.csv'.format(args.dataset)))
    if args.task_type == 'classification':
        out_dim = 2 * len(task_name)
    elif args.task_type == 'regression':
        out_dim = len(task_name)
    else:
        raise Exception('Unknown task type')
    args.task_name = task_name
    args.out_dim = out_dim

    if args.split_type == 'random':
        train_dataset, valid_dataset, test_dataset = load_dataset_random(data_path,
                                                                         args.dataset,
                                                                         args.seed,
                                                                         args.task_type,
                                                                         task_name,
                                                                         logger)

    elif args.split_type == 'scaffold':
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(data_path,
                                                                           args.dataset,
                                                                           args.seed,
                                                                           args.task_type,
                                                                           task_name,
                                                                           logger)

    elif args.split_type == 'noise':
        train_dataset, valid_dataset, test_dataset = load_dataset_noise(data_path,
                                                                        args.dataset,
                                                                        args.seed,
                                                                        args.task_type,
                                                                        task_name,
                                                                        args.rate,
                                                                        logger)
    elif args.split_type == 'all':
        dataset = load_dataset_all(data_path,
                                   args.dataset,
                                   args.seed,
                                   args.task_type,
                                   task_name,
                                   logger)
        return dataset

    else:
        raise Exception('Unknown dataset split type')

    return train_dataset, valid_dataset, test_dataset


def build_loader(args, logger):
    if args.split_type == 'all':
        dataset = build_dataset(args, logger)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train_dataset, valid_dataset, test_dataset = build_dataset(args, logger)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    weights = train_dataset.weights

    return train_dataloader, valid_dataloader, test_dataloader, weights
