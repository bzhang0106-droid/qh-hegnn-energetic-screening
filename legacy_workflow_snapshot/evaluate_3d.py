import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. 路径与配置
# ==========================================
OLD_CSV_PATH = "../data/old_dataset.csv"
NEW_CSV_PATH = "../data/oracle_results.csv"
XYZ_DIR = "../data/raw_2100_xyz"
MODEL_SAVE_PATH = "../results/best_multitask_egnn.pth"
PLOT_SAVE_PATH = "../results/Parity_Plot_5D.png"

TARGET_PROPS = [
    'Density_calc(g/cm3)', 
    'Heat_of_Formation(kcal/mol)', 
    'VS_max', 
    'Trigger_Bond_Rho', 
    'HOMO_LUMO_Gap(eV)'
]

# 🧬 与 03/04 绝对对齐的 2D 拓扑锚点提取器
def extract_2d_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [0.0]*5
    mw = Descriptors.ExactMolWt(mol)
    n_het = rdMolDescriptors.CalcNumHeteroatoms(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_NO2 = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])]')))
    n_N = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    return [mw, n_het, n_rings, n_NO2, n_N]

# 🧠 与 03/04 绝对对齐的混合网络架构
class HybridEGNN(torch.nn.Module):
    def __init__(self, num_targets=5, num_2d=5):
        super().__init__()
        self.schnet = SchNet(hidden_channels=128, num_filters=128, num_interactions=6, num_gaussians=50, cutoff=10.0, readout='mean')
        self.schnet.lin2 = torch.nn.Linear(64, 64) 
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(64 + num_2d, 128),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_targets)
        )

    def forward(self, z, pos, batch, x_2d):
        # 验证期间不注入噪声
        emb_3d = self.schnet(z, pos, batch)
        fused = torch.cat([emb_3d, x_2d], dim=1)
        return self.mlp(fused)

class EnergeticDataset(torch.utils.data.Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def load_xyz_coordinates(mol_name):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 启动 Hybrid EGNN 5D 量子特征验证模式！引擎: {device}")

    # ==========================================
    # 2. 严格的数据加载与清洗逻辑
    # ==========================================
    df_old = pd.read_csv(OLD_CSV_PATH) if os.path.exists(OLD_CSV_PATH) else pd.DataFrame()
    df_new = pd.read_csv(NEW_CSV_PATH) if os.path.exists(NEW_CSV_PATH) else pd.DataFrame()
    
    if not df_old.empty: df_old = df_old.drop_duplicates(subset=['SMILES'])
    if not df_new.empty: df_new = df_new.drop_duplicates(subset=['SMILES'])

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['SMILES'], keep='last')
    df_all = df_all.dropna(subset=TARGET_PROPS)

    targets_tensor = torch.tensor(df_all[TARGET_PROPS].values, dtype=torch.float)
    
    print("[INFO] 正在重构 3D 坐标空间并挂载 2D 锚点...")
    data_list = []
    mol_col = 'Molecule' if 'Molecule' in df_all.columns else 'Moleule'
    for idx, row in df_all.iterrows():
        mol_name = str(row[mol_col]).replace('.xyz', '').replace('.out', '')
        smi = str(row['SMILES'])
        z, pos = load_xyz_coordinates(mol_name)
        if z is not None:
            y_real = targets_tensor[idx].unsqueeze(0) 
            x_2d = torch.tensor([extract_2d_features(smi)], dtype=torch.float)
            data_list.append(Data(z=z, pos=pos, y=y_real, x_2d=x_2d, mol_name=mol_name))

    random.seed(42)
    random.shuffle(data_list)
    train_size = int(0.9 * len(data_list))
    val_dataset = EnergeticDataset(data_list[train_size:])
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"[INFO] 成功提取出 {len(val_dataset)} 个绝对未见的验证集样本。")

    # ==========================================
    # 3. 唤醒混合网络模型
    # ==========================================
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ 找不到模型权重: {MODEL_SAVE_PATH}")
        print("💡 请先运行 python 03_egnn_painn_train.py 训练新版 Hybrid 模型。")
        return
        
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    means = torch.tensor(checkpoint['means']).to(device)
    stds = torch.tensor(checkpoint['stds']).to(device)
    
    model = HybridEGNN(num_targets=len(TARGET_PROPS), num_2d=5).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # ==========================================
    # 4. 执行预测
    # ==========================================
    all_preds, all_targets = [], []
    print("⏳ 正在对验证集执行高精度 Hybrid 拓扑推断...")
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out_norm = model(batch.z, batch.pos, batch.batch, batch.x_2d)
            out_real = out_norm * stds + means
            all_preds.append(out_real.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # ==========================================
    # 5. 指标计算与雷达图生成
    # ==========================================
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    units = ["g/cm³", "kcal/mol", "kcal/mol", "e/Bohr³", "eV"]
    titles = ["Density", "Heat of Formation", "VS Max (ESP)", "Trigger Bond (Rho)", "HOMO-LUMO Gap"]

    print("\n📊 真实 Hybrid 验证集物理表现 (已免疫坐标坍塌):")
    for i in range(5):
        y_true, y_pred = all_targets[:, i], all_preds[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  {titles[i]:<20} | MAE: {mae:<8.4f} {units[i]:<10} | R2: {r2:.4f}")

        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.6, color=colors[i], edgecolor='w', s=50)
        min_val, max_val = np.min(y_true), np.max(y_true)
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', lw=1.5, alpha=0.7)
        ax.set_title(f"{titles[i]}\n$R^2$ = {r2:.3f} | MAE = {mae:.3f} {units[i]}", fontsize=12, pad=10)
        ax.set_xlabel("DFT True Value", fontsize=10)
        ax.set_ylabel("Hybrid Predicted Value", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✅ 全景物理雷达图已保存至: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()
