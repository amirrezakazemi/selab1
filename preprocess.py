import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from sklearn.utils import resample

def rawdata_to_csv():
  all_prots = []
  datasets = ['kiba','davis']
  for dataset in datasets:
      fpath = 'data/' + dataset + '/'
      train_val_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
      train_fold = [ee for i in range(len(train_val_fold)-1) for ee in train_val_fold[i]]
      val_fold = [ee for ee in train_val_fold[-1]]
      test_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
      ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
      proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
      affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
      drugs = []
      prots = []
      for d in ligands.keys():
          lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
          drugs.append(lg)
      for t in proteins.keys():
          prots.append(proteins[t])
      if dataset == 'davis':
          affinity = [-np.log10(y/1e9) for y in affinity]
      affinity = np.asarray(affinity)
      opts = ['train', 'val', 'test']
      for opt in opts:
          rows, cols = np.where(np.isnan(affinity)==False)
          if opt=='train':
              rows,cols = rows[train_fold], cols[train_fold]
          elif opt=='test':
              rows,cols = rows[test_fold], cols[test_fold]
          elif opt=='val':
              rows,cols = rows[val_fold], cols[val_fold]
          with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
              f.write('compound_iso_smiles,target_sequence,affinity\n')
              for pair_ind in range(len(rows)):
                  ls = []
                  ls += [ drugs[rows[pair_ind]]  ]
                  ls += [ prots[cols[pair_ind]]  ]
                  ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                  f.write(','.join(map(str,ls)) + '\n')
      all_prots += list(set(prots))
      X = pd.read_csv("data/" + dataset + '_train.csv')
      if dataset == "davis":
        threshold = 7
      else:
        threshold = 12.1
      major = X[X.affinity<=threshold]
      minor = X[X.affinity>threshold]
      minor_upsampled = resample(minor,
                          replace=True, # sample with replacement
                          n_samples=len(major), # match number in majority class
                          random_state=2)
      X_upsampled = pd.concat([major, minor_upsampled])
      X_upsampled.to_csv("data/"+dataset +'_resampled_train.csv')



def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def seq_cat(prot, max_seq_len, seq_dict):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


def csv_to_PytorchData():
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)
    max_seq_len = 1000

    compound_iso_smiles = []
    for dt_name in ['kiba', 'davis']:
        opts = ['train', 'val', 'test']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
            compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    datasets = ['davis', 'kiba']
    for dataset in datasets:
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_resampled_train = 'data/processed/' + dataset + '_resampled_train.pt'
        processed_data_file_val = 'data/processed/' + dataset + '_val.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)) or (
        not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_resampled_train))):
            df = pd.read_csv('data/' + dataset + '_train.csv')
            train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                df['affinity'])
            XT = [seq_cat(t, max_seq_len, seq_dict) for t in train_prots]
            train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

            df = pd.read_csv('data/' + dataset + '_resampled_train.csv')
            train_resampled_drugs, train_resampled_prots, train_resampled_Y = list(df['compound_iso_smiles']), list(
                df['target_sequence']), list(df['affinity'])
            XT = [seq_cat(t, max_seq_len, seq_dict) for t in train_resampled_prots]
            train_resampled_drugs, train_resampled_prots, train_resampled_Y = np.asarray(
                train_resampled_drugs), np.asarray(XT), np.asarray(train_resampled_Y)

            df = pd.read_csv('data/' + dataset + '_val.csv')
            val_drugs, val_prots, val_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                df['affinity'])
            XT = [seq_cat(t, max_seq_len, seq_dict) for t in val_prots]
            val_drugs, val_prots, val_Y = np.asarray(val_drugs), np.asarray(XT), np.asarray(val_Y)

            df = pd.read_csv('data/' + dataset + '_test.csv')
            test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                df['affinity'])
            XT = [seq_cat(t, max_seq_len, seq_dict) for t in test_prots]
            test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)

            train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots,
                                        y=train_Y, smile_graph=smile_graph)

            train_resampled_data = TestbedDataset(root='data', dataset=dataset + '_resampled_train',
                                                  xd=train_resampled_drugs, xt=train_resampled_prots,
                                                  y=train_resampled_Y, smile_graph=smile_graph)

            val_data = TestbedDataset(root='data', dataset=dataset + '_val', xd=val_drugs, xt=val_prots, y=val_Y,
                                      smile_graph=smile_graph)

            test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots, y=test_Y,
                                       smile_graph=smile_graph)
            print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
        else:
            print(processed_data_file_train, ' and ', processed_data_file_test, ' and ', processed_data_file_val,
                  ' are already created')

