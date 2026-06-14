import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    class MockSAScorer:
        def calculateScore(self, mol): return 3.0
    sascorer = MockSAScorer()

INPUT_CSV = "../results/Final_5D_Top_Candidates.csv"
OUTPUT_CSV = "../results/Pareto_Optimal_Candidates.csv"

# [第一重过滤：科学红线绝对禁区]
MIN_DENSITY  = 1.60
MIN_HOF      = 0.0
MIN_RHO      = 0.165
MAX_VS_MAX   = 55.0
MIN_GAP      = 3.0
MAX_SA_SCORE = 5.5

def get_pareto_front(df, max_cols, min_cols):
    """
    【学术核武】快速非支配排序算法。
    寻找在多维物理量博弈中，不被任何其他分子绝对支配的最优解集群。
    """
    scores = np.zeros((len(df), len(max_cols) + len(min_cols)))
    # 全部映射为“极小化 (Minimize)”
    for i, col in enumerate(max_cols):
        scores[:, i] = -df[col].values
    for i, col in enumerate(min_cols):
        scores[:, len(max_cols) + i] = df[col].values
        
    n = len(scores)
    is_pareto = np.ones(n, dtype=bool)
    
    # 向量化帕累托支配判定
    for i in range(n):
        diff = scores - scores[i]
        # 判断分子 i 是否被其他分子 j 绝对支配：j 在所有维度均 <= i，且至少有一项 < i
        dominated_by = np.any(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1))
        if dominated_by:
            is_pareto[i] = False
            
    # 返回绝对帕累托前沿 (Rank 0 面) 上的无敌分子
    return df[is_pareto].copy()

def main():
    print("==================================================")
    print("⚖️ 启动多目标纯正帕累托裁决机 (Non-dominated Pareto Front)")
    print("==================================================")

    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV)
    
    dens_col = [c for c in df.columns if 'Density' in c][0]
    hof_col  = [c for c in df.columns if 'Heat_of_Formation' in c][0]
    vs_col   = [c for c in df.columns if 'VS_max' in c][0]
    rho_col  = [c for c in df.columns if 'Trigger_Bond_Rho' in c][0]
    gap_col  = [c for c in df.columns if 'HOMO_LUMO_Gap' in c][0]

    # 1. 结构与合成可行性安检
    valid_indices, sa_scores = [], []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row['SMILES']))
        if mol is not None and Chem.GetFormalCharge(mol) == 0:
            sa_scores.append(sascorer.calculateScore(mol))
            valid_indices.append(idx)

    df = df.loc[valid_indices].copy()
    df['SA_Score_RDKit'] = sa_scores

    # 2. 物理红线截断
    df_pareto_pool = df[
        (df[dens_col] >= MIN_DENSITY) & (df[vs_col] <= MAX_VS_MAX) &
        (df[rho_col] >= MIN_RHO) & (df[hof_col] >= MIN_HOF) &
        (df[gap_col] >= MIN_GAP) & (df['SA_Score_RDKit'] <= MAX_SA_SCORE)
    ].copy()

    print(f"\n[INFO] 突破红线生还者池: {len(df_pareto_pool)} 个分子。")
    if len(df_pareto_pool) == 0:
        pd.DataFrame(columns=df.columns).to_csv(OUTPUT_CSV, index=False)
        print("⚠️ 警告：物理漏斗全军覆没！飞轮强制转向下一纪元探索。")
        return

    # 3. 核心计算：提取多维空间中的 Pareto Rank-0 前沿面
    max_cols = [dens_col, hof_col, gap_col, rho_col]
    min_cols = ['SA_Score_RDKit', vs_col]
    
    df_pareto = get_pareto_front(df_pareto_pool, max_cols, min_cols)
    
    # 在等价的帕累托前沿内部，以密度作为次级优先级进行排位送入神谕
    df_final = df_pareto.sort_values(by=dens_col, ascending=False).reset_index(drop=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print("\n👑 终极预备役 TOP 5 (严格位于帕累托前沿 Rank-0):")
    cols_to_show = ['Molecule', dens_col, hof_col, rho_col, 'SA_Score_RDKit']
    print(df_final.head(5)[cols_to_show].to_string(index=False))
    print(f"\n🏆 突破漏斗进入神谕候选队列: {len(df_final)} 个！已存档。")

if __name__ == "__main__":
    main()
