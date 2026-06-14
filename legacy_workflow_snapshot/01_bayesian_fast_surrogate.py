import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

def smiles_to_graph(smiles, target_val=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 🛠️ 核心升级：6维物理化学特征 (加入杂化、芳香性、环信息)
    node_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        charge = atom.GetFormalCharge()
        hybridization = int(atom.GetHybridization())
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        is_in_ring = 1.0 if atom.IsInRing() else 0.0
        node_features.append([atomic_num, degree, charge, hybridization, is_aromatic, is_in_ring])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i]) 
        
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    y = torch.tensor([target_val], dtype=torch.float) if target_val is not None else None
    return Data(x=x, edge_index=edge_index, y=y, smiles=smiles)

class MoleculeGraphDataset(Dataset):
    def __init__(self, df, has_target=True):
        super().__init__(None, None, None)
        self.graphs = []
        print("🕸️ 正在将分子序列折叠为高维图拓扑网络 (Graph)...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            target = row['Density_calc(g/cm3)'] if has_target else None
            data = smiles_to_graph(row['SMILES'], target)
            if data is not None: self.graphs.append(data)
    def len(self): return len(self.graphs)
    def get(self, idx): return self.graphs[idx]

class BayesianGNN(torch.nn.Module):
    def __init__(self, hidden_channels=128):
        super(BayesianGNN, self).__init__()
        # 🛠️ 匹配升级后的 6 维输入
        self.conv1 = GCNConv(6, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 64)
        self.lin2 = Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch) 
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.lin2(x)

def main():
    print("🧠 启动 Workflow 2.5: Bayesian GNN (强化拓扑特征版)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_old = pd.read_csv("../data/old_dataset.csv")
    if 'SMILES' not in df_old.columns:
        df_smiles = pd.read_csv("../data/smiles_mapping.csv")
        df_train = pd.merge(df_old, df_smiles, on='Molecule', how='inner')
    else:
        df_train = df_old.copy()
        
    train_dataset = MoleculeGraphDataset(df_train, has_target=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = BayesianGNN(hidden_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, 61):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0: print(f"Epoch {epoch:03d}/60 | Loss: {total_loss/len(train_loader):.4f}")

    df_gpt = pd.read_csv("../data/GPT_Generated_Candidates.csv")
    pool_dataset = MoleculeGraphDataset(df_gpt, has_target=False)
    pool_loader = DataLoader(pool_dataset, batch_size=128, shuffle=False)
    
    model.train() # MC-Dropout
    num_passes = 10
    all_preds_list = [[] for _ in range(num_passes)]
    
    with torch.no_grad():
        for i in range(num_passes):
            for data in pool_loader:
                data = data.to(device)
                preds = model(data.x, data.edge_index, data.batch).cpu().numpy().flatten()
                all_preds_list[i].extend(preds)
                
    all_preds = np.array(all_preds_list) 
    mean_preds = np.mean(all_preds, axis=0)
    var_preds = np.var(all_preds, axis=0) 

    K = 2.0 
    ucb_scores = mean_preds + K * np.sqrt(var_preds)
    
    df_results = pd.DataFrame({'Molecule': df_gpt['Molecule'], 'SMILES': df_gpt['SMILES'], 'UCB_Score': ucb_scores})
    top100 = df_results.sort_values(by='UCB_Score', ascending=False).head(100).copy()
    top100['Molecule'] = [f"GPT_AL_Target_{i+1:04d}" for i in range(len(top100))]
    
    output_path = "../data/active_learning_targets_100.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top100.to_csv(output_path, index=False)
    print(f"🎯 已锁定 100 个主动学习探索目标！保存至 {output_path}")

if __name__ == "__main__":
    main()
