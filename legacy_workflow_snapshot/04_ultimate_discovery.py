import os
# ==============================================================================
# 🛡️ HPC 线程锁死阵列 (必须在导入 numpy/torch/rdkit 之前执行！)
# 防止 Multiprocessing 导致几千个 OpenMP 线程互相绞杀引发系统级卡死
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)  # 锁死 PyTorch CPU 线程

import torch.nn as nn
import tempfile
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, radius_graph
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

INPUT_CSV = "../data/GPT_Generated_Candidates.csv"
OUTPUT_CSV = "../results/Final_5D_Top_Candidates.csv"
MODEL_SAVE_PATH = "../results/best_multitask_egnn.pth"

# ==============================================================================
# 核心物理引擎：Native EGNN (与 03 完全对齐)
# ==============================================================================
def native_scatter_mean(src, index, dim_size):
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
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff**2, dim=1).unsqueeze(1)
        
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=-1))
        coord_msg = coord_diff * self.coord_mlp(m_ij)
        pos_aggr = native_scatter_mean(coord_msg, row, dim_size=pos.size(0))
        pos_out = pos + pos_aggr
        
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

def smiles_to_3d_xtb(smi):
    try:
        smi = str(smi).strip()
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0: return None, "ETKDG 坐标塌缩"

        # 🛡️ 强制在子进程中将环境变量传递给 xTB 二进制程序
        xtb_env = os.environ.copy()
        xtb_env['OMP_NUM_THREADS'] = '1'
        xtb_env['MKL_NUM_THREADS'] = '1'

        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "init.xyz")
            Chem.MolToXYZFile(mol, xyz_path)
            
            # 极速 GFN2-xTB 构象弛豫，传入单线程隔离的 env
            cmd = ["xtb", "init.xyz", "--opt", "normal", "--gfn", "2", "--chrg", "0"]
            subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45, env=xtb_env)
            
            opt_xyz = os.path.join(tmpdir, "xtbopt.xyz")
            if not os.path.exists(opt_xyz): 
                AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
                Chem.MolToXYZFile(mol, opt_xyz)
            
            atomic_nums = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
            z, pos = [], []
            with open(opt_xyz, 'r') as f:
                for line in f.readlines()[2:]:
                    parts = line.split()
                    if len(parts) >= 4 and parts[0] in atomic_nums:
                        z.append(atomic_nums[parts[0]])
                        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        
        x_2d = torch.tensor([extract_2d_features(smi)], dtype=torch.float)
        return Data(z=torch.tensor(z, dtype=torch.long), pos=torch.tensor(pos, dtype=torch.float), x_2d=x_2d, smiles=smi), "Success"
    except Exception as e:
        return None, str(e)

class InferenceDataset(Dataset):
    def __init__(self, data_list): 
        super().__init__(None, None, None)
        self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def main():
    print("==================================================")
    print("🔮 启动 04_Ultimate: xTB 物理降噪 + Native EGNN 联合推断引擎")
    print("==================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ 找不到权重文件: {MODEL_SAVE_PATH}。请确保 03 脚本已成功运行。")
        return

    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
    means, stds = torch.tensor(ckpt['means']).to(device), torch.tensor(ckpt['stds']).to(device)
    target_props = ckpt['targets']

    model = TrueHybridEGNN(num_targets=len(target_props)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入文件: {INPUT_CSV}")
        return

    df_gen = pd.read_csv(INPUT_CSV)
    
    # 限制最大并行度，留出冗余核心给系统调度
    num_cores = min(60, max(1, os.cpu_count() - 4))

    valid_graphs = []
    print(f"⚡ 分配 {num_cores} 核集群执行大通量 xTB 量子力学预弛豫 (强制单线程隔离)...")
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(smiles_to_3d_xtb, df_gen['SMILES']), total=len(df_gen)))
        for data, msg in results:
            if data is not None: valid_graphs.append(data)

    if not valid_graphs:
        print("⚠️ 所有分子物理拓扑均崩溃，无法进行推断。")
        return

    loader = DataLoader(InferenceDataset(valid_graphs), batch_size=256, shuffle=False)
    all_preds, all_smiles = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="GPU EGNN 高保真推断"):
            batch = batch.to(device)
            out_real = model(batch.z, batch.pos, batch.batch, batch.x_2d) * stds + means
            all_preds.extend(out_real.cpu().numpy())
            all_smiles.extend(batch.smiles)

    df_res = pd.DataFrame(all_preds, columns=[f"Pred_{c}" for c in target_props])
    df_res.insert(0, 'SMILES', all_smiles)
    df_res.insert(0, 'Molecule', [f"GPT_xTB_Candidate_{i:06d}" for i in range(len(df_res))])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"\n🎉 完美收官！高保真 9D 数据已写入: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
