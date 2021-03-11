from models import *
from training import predicting
import pandas as pd
import sys
import csv
from data_preprocessing import *
input_dir , output_dir = sys.argv[1], sys.argv[2]
df = pd.read_table(input_dir, header=None)

df.columns = ['target_sequence', 'compound_iso_smiles']
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000
train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']), [-1] * len(df['target_sequence'])
XT = [seq_cat(t, max_seq_len, seq_dict) for t in train_prots]
train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
compound_iso_smiles = set(list(df['compound_iso_smiles']))
smile_graph = {}
for smile in compound_iso_smiles:
  g = smile_to_graph(smile)
  smile_graph[smile] = g
train_data = TestbedDataset(root='data', dataset='main', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
model = GINConvNet()
model.load_state_dict(torch.load('model_davis.model'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
G, P = predicting(model, device, train_loader)
print(P)
writer=csv.writer(open(output_dir,'w'))
for word in P:
    writer.writerow([word])