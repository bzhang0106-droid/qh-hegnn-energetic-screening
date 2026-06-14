import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# ⚙️ 配置路径
# ==========================================
DATA_PATH = "../data/old_dataset.csv"
OUT_DIR = "../results/eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("==================================================")
    print("📊 启动高能分子数据集全景 EDA 分析 (9D 架构版)")
    print("==================================================")

    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到核心数据集: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # 🌟 核心升级：对齐 9D 特征列
    target_cols = [
        'Density_calc(g/cm3)', 'Heat_of_Formation(kcal/mol)',
        'HOMO_LUMO_Gap(eV)', 'SAscore', 'VS_max',
        'Sigma2_tot', 'Nu', 'Trigger_Bond_Rho',
        'Molecular_Weight'
    ]

    existing_cols = [col for col in target_cols if col in df.columns]

    print(f"\n📦 数据集规模: {df.shape[0]} 个分子, {df.shape[1]} 个特征维度")
    print(f"✅ 成功匹配到 {len(existing_cols)} 个有效物理特征。")
    print("\n🔍 核心特征统计学描述:")
    print(df[existing_cols].describe().round(4))

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # 1. 生成特征分布矩阵 (Histograms)
    print(f"\n🎨 正在绘制特征分布直方图...")
    # 🌟 核心修改：扩容为 3x3 绘图阵列以容纳 9 维数据
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, col in enumerate(existing_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
        axes[i].set_title(col, pad=10, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')

    # 清理多余的空白子图（如果有效特征不足 9 个）
    for j in range(len(existing_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    dist_path = os.path.join(OUT_DIR, "01_feature_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    print(f"  -> 已保存: {dist_path}")

    # 2. 绘制 5D 帕累托前沿散点图 (Density vs HOF, 颜色映射 SAscore)
    print(f"🎯 正在绘制 HELS 目标帕累托前沿...")
    plt.figure(figsize=(10, 8))
    if 'Density_calc(g/cm3)' in existing_cols and 'Heat_of_Formation(kcal/mol)' in existing_cols:
        scatter = plt.scatter(
            df['Density_calc(g/cm3)'],
            df['Heat_of_Formation(kcal/mol)'],
            c=df.get('SAscore', 0),
            cmap='viridis',
            alpha=0.7, edgecolors='w', s=50
        )
        plt.colorbar(scatter, label='SAscore (Lower is easier to synthesize)')
        plt.xlabel('Density (g/cm³)')
        plt.ylabel('Heat of Formation (kcal/mol)')
        plt.title('HELS Pareto Frontier: Density vs. HOF')
        # 标出高密度高能量象限红线
        plt.axvline(x=1.8, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)

        pareto_path = os.path.join(OUT_DIR, "02_pareto_frontier.png")
        plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
        print(f"  -> 已保存: {pareto_path}")

    # 3. 特征相关性热力图 (排查多重共线性)
    print(f"🔗 正在绘制特征相关性热力图...")
    plt.figure(figsize=(12, 10))
    corr = df[existing_cols].corr()
    # 使用 vmin=-1, vmax=1 保证色阶跨度正确
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Matrix (9D Architecture)', pad=15)
    
    corr_path = os.path.join(OUT_DIR, "03_correlation_matrix.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"  -> 已保存: {corr_path}")

    print("\n🎉 EDA 分析完成！请到 master1 节点的 ../results/eda_plots 目录下查看高精度图片。")

if __name__ == "__main__":
    main()
