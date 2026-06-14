import os
import torch
import torch.nn as nn
import pandas as pd
import random
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, radius_graph

OLD_CSV_PATH = "../data/old_dataset.csv"
XYZ_DIR = "../data/raw_2100_xyz"
MODEL_SAVE_PATH = "../results/best_multitask_egnn.pth"

TARGET_PROPS = [
    'Density_calc(g/cm3)', 'Heat_of_Formation(kcal/mol)', 'HOMO_LUMO_Gap(eV)', 
    'SAscore', 'VS_max', 'Sigma2_tot', 'Nu', 'Trigger_Bond_Rho', 'Molecular_Weight'
]

# ==============================================================================
# 核心物理引擎：原生 PyTorch 实现的 EGNN 层 (绝对免疫 PyG 版本冲突)
# 遵循等变更新方程: x_i^{(l+1)} = x_i^{(l)} + \sum (x_i - x_j) * \phi_x(m_{ij})
# ==============================================================================
def native_scatter_mean(src, index, dim_size):
    """原生并行聚合算子，代替第三方 scatter_mean"""
    out = torch.zeros(dim_size, src.size(-1), device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    count = torch.zeros(dim_size, src.size(-1), device=src.device)
    count.scatter_add_(0, index_expanded, torch.ones_like(src))
    return out / count.clamp(min=1)

class NativeEGNNLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, 1, bias=False)
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        
        # 提取空间物理特征: 相对位移与欧氏距离平方
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, dim=1).unsqueeze(1)
        
        # 1. 边缘信息流 (Message Passing)
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=-1))
        
        # 2. 等变坐标修正 (Equivariant Coordinate Update)
        # 此处彻底排除了人为的高斯噪声，让网络依靠物理梯度自动寻找弛豫极小值点
        coord_msg = coord_diff * self.coord_mlp(m_ij)
        pos_aggr = native_scatter_mean(coord_msg, row, dim_size=pos.size(0))
        pos_out = pos + pos_aggr
        
        # 3. 节点特征更新
        m_aggr = native_scatter_mean(m_ij, row, dim_size=h.size(0))
        h_out = h + self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        
        return h_out, pos_out

class TrueHybridEGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_targets=9, num_2d=9):
        super().__init__()
        self.node_emb = nn.Embedding(100, hidden_dim) 
        
        self.conv1 = NativeEGNNLayer(hidden_dim)
        self.conv2 = NativeEGNNLayer(hidden_dim)
        self.conv3 = NativeEGNNLayer(hidden_dim)
        self.conv4 = NativeEGNNLayer(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_2d, 256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, num_targets)
        )

    def forward(self, z, pos, batch, x_2d):
        h = self.node_emb(z)
        edge_index = radius_graph(pos, r=4.0, batch=batch, max_num_neighbors=32)
        
        h, pos = self.conv1(h, pos, edge_index)
        h, pos = self.conv2(h, pos, edge_index)
        h, pos = self.conv3(h, pos, edge_index)
        h, pos = self.conv4(h, pos, edge_index)
        
        h_graph = global_mean_pool(h, batch)
        return self.mlp(torch.cat([h_graph, x_2d], dim=1))
# ==============================================================================

def extract_2d_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [0.0]*9
    nitro_pat = Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])]')
    return [
        Descriptors.ExactMolWt(mol), rdMolDescriptors.CalcNumHeteroatoms(mol), rdMolDescriptors.CalcNumRings(mol),
        len(mol.GetSubstructMatches(nitro_pat)) if nitro_pat else 0, sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]-[$([NX3](=O)=O),$([NX3+](=O)[O-])]'))),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]-[$([NX3](=O)=O),$([NX3+](=O)[O-])]'))),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#7]'))),
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[N]=[N+]=[N-]')))
    ]

def load_xyz_graph(mol_name):
    xyz_path = os.path.join(XYZ_DIR, f"{mol_name}.xyz")
    if not os.path.exists(xyz_path): return None, None
    atomic_nums = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Cl': 17}
    z, pos = [], []
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 3: return None, None
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] in atomic_nums:
                z.append(atomic_nums[parts[0]])
                pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not z: return None, None
    return torch.tensor(z, dtype=torch.long), torch.tensor(pos, dtype=torch.float)

class EnergeticDataset(Dataset):
    def __init__(self, data_list): 
        super().__init__(None, None, None)
        self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def main():
    print("==================================================")
    print("🚀 启动 9D 物理引擎: Native EGNN (全节点兼容等变微调架构)")
    print("==================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_all = pd.read_csv(OLD_CSV_PATH).dropna(subset=TARGET_PROPS)
    df_all = df_all[(df_all['Heat_of_Formation(kcal/mol)'] > -2000) & (df_all['Heat_of_Formation(kcal/mol)'] < 3000)].reset_index(drop=True)

    targets_tensor = torch.tensor(df_all[TARGET_PROPS].values, dtype=torch.float)
    means, stds = targets_tensor.mean(dim=0), targets_tensor.std(dim=0)
    stds[stds == 0] = 1.0
    targets_norm = (targets_tensor - means) / stds

    data_list = []
    mol_col = 'Molecule' if 'Molecule' in df_all.columns else 'Moleule'
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="内存挂载 3D 拓扑"):
        mol_name = str(row[mol_col]).replace('.xyz', '').replace('.out', '')
        z, pos = load_xyz_graph(mol_name)
        if z is not None:
            x_2d = torch.tensor([extract_2d_features(row['SMILES'])], dtype=torch.float)
            data_list.append(Data(z=z, pos=pos, y=targets_norm[idx].unsqueeze(0), x_2d=x_2d))

    torch.manual_seed(42)
    indices = torch.randperm(len(data_list)).tolist()
    train_size = int(0.9 * len(data_list))
    
    train_loader = DataLoader(EnergeticDataset([data_list[i] for i in indices[:train_size]]), batch_size=64, shuffle=True)
    val_loader = DataLoader(EnergeticDataset([data_list[i] for i in indices[train_size:]]), batch_size=64, shuffle=False)

    model = TrueHybridEGNN(hidden_dim=128, num_targets=len(TARGET_PROPS), num_2d=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    epochs = 80
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.z.to(device), batch.pos.to(device), batch.batch.to(device), batch.x_2d.to(device))
            loss = F.huber_loss(out, batch.y.to(device), delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch.z.to(device), batch.pos.to(device), batch.batch.to(device), batch.x_2d.to(device))
                    val_loss += F.huber_loss(out, batch.y.to(device), delta=1.0).item() * batch.num_graphs
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            print(f"Epoch {epoch:03d}/{epochs} | Val Huber Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_loss:
                best_loss = val_loss
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save({'state_dict': model.state_dict(), 'means': means.numpy(), 'stds': stds.numpy(), 'targets': TARGET_PROPS}, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
