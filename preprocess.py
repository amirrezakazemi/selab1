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
