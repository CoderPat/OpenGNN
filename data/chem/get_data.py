import os
from rdkit import Chem
import glob
import json
import numpy as np

if not os.path.exists('data'):
    os.mkdir('data')
    print('made directory ./data/')

download_path = os.path.join('data', 'dsgdb9nsd.xyz.tar.bz2')
if not os.path.exists(download_path):
    print('downloading data to %s ...' % download_path)
    source = 'https://ndownloader.figshare.com/files/3195389'
    os.system('wget -O %s %s' % (download_path, source))
    print('finished downloading')

unzip_path = os.path.join('data', 'qm9_raw')
if not os.path.exists(unzip_path):
    print('extracting data to %s ...' % unzip_path)
    os.mkdir(unzip_path)
    os.system('tar xvjf %s -C %s' % (download_path, unzip_path))
    print('finished extracting')


def preprocess():
    index_of_mu = 4

    def read_xyz(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            smiles = lines[-2].split('\t')[0]
            properties = lines[1].split('\t')
            mu = float(properties[index_of_mu])
        return {'smiles': smiles, 'mu': mu}

    print('loading train/validation split')
    with open('valid_idx.json', 'r') as f:
        valid_idx = json.load(f)['valid_idxs']
    valid_files = [os.path.join(unzip_path, 'dsgdb9nsd_%s.xyz' % i)
                   for i in valid_idx]

    print('reading data...')
    raw_data = {'train': [], 'valid': []}
    all_files = glob.glob(os.path.join(unzip_path, '*.xyz'))
    for file_idx, file_path in enumerate(all_files):
        if file_idx % 100 == 0:
            print('%.1f %%    \r' %
                  (file_idx / float(len(all_files)) * 100), end=""),
        if file_path not in valid_files:
            raw_data['train'].append(read_xyz(file_path))
        else:
            raw_data['valid'].append(read_xyz(file_path))
    all_mu = [mol['mu'] for mol in raw_data['train']]
    mean_mu = np.mean(all_mu)
    std_mu = np.std(all_mu)

    def normalize_mu(mu):
        return (mu - mean_mu) / std_mu

    def onehot(idx, len):
        z = [0 for _ in range(len)]
        z[idx] = 1
        return z

    bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}

    def to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        edges = []
        nodes = []
        for bond in mol.GetBonds():
            edges.append((str(bond.GetBondType()),
                          bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for atom in mol.GetAtoms():
            nodes.append(atom.GetSymbol())
        return nodes, edges

    print('parsing smiles as graphs...')
    processed_graphs = {'train': [], 'valid': []}
    processed_labels = {'train': [], 'valid': []}
    for section in ['train', 'valid']:
        for i, (smiles, mu) in enumerate([(mol['smiles'], mol['mu']) for mol in raw_data[section]]):
            if i % 100 == 0:
                print('%s: %.1f %%      \r' %
                      (section, 100*i/float(len(raw_data[section]))), end="")
            nodes, edges = to_graph(smiles)
            processed_graphs[section].append({
                'edges': edges,
                'node_labels': nodes
            })
            processed_labels[section].append([normalize_mu(mu)])

        print('%s: 100 %%      ' % (section))
        with open('molecules_graphs_%s.jsonl' % section, 'w') as f:
            for graph in processed_graphs[section]:
                f.write(json.dumps(graph) + "\n")
        with open('molecules_labels_%s.jsonl' % section, 'w') as f:
            for label in processed_labels[section]:
                f.write(json.dumps(label) + "\n")


preprocess()
