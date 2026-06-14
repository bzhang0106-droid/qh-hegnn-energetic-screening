import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None

def main():
    print("==================================================")
    print("🕵️ 启动数据泄露与近亲繁殖检测器 (飞轮自适应版)")
    print("==================================================")
    
    df = pd.read_csv('../data/old_dataset.csv')
    total_mols = len(df)
    print(f"📦 当前总库容量: {total_mols}")

    duplicates = df[df.duplicated(subset=['SMILES'], keep=False)]
    if not duplicates.empty:
        print(f"🚨 发现 {len(duplicates)} 条绝对重复数据！")

    collapse_ratio = 0.0
    if total_mols > 254:
        print("\n🔬 正在进行化学空间坍塌分析 (计算 Tanimoto 相似度)...")
        old_df = df.iloc[:-254].copy()
        new_df = df.iloc[-254:].copy()
        
        old_fps = [fp for fp in [get_fingerprint(s) for s in old_df['SMILES']] if fp]
        new_fps = [fp for fp in [get_fingerprint(s) for s in new_df['SMILES']] if fp]

        high_sim_count = 0
        for new_fp in new_fps:
            sims = DataStructs.BulkTanimotoSimilarity(new_fp, old_fps)
            if sims and max(sims) > 0.90:
                high_sim_count += 1
                
        collapse_ratio = (high_sim_count / len(new_fps) * 100) if new_fps else 0.0
        print(f"📊 模式坍塌诊断结果: 同质化比例 {collapse_ratio:.1f}%")
        
        if collapse_ratio > 50:
            print("🚨 [警告] 发生严重近亲繁殖！系统已向飞轮调度中心发送干预请求。")
        else:
            print("✅ 空间多样性健康，排除模式坍塌。")
            
    # 🌟 发送神经递质信号
    os.makedirs('../results', exist_ok=True)
    with open('../results/collapse_ratio.txt', 'w') as f:
        f.write(str(collapse_ratio))

if __name__ == "__main__":
    main()
