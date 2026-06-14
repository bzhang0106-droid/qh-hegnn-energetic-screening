import os
import argparse
import torch
import random
import pandas as pd
from tqdm import tqdm
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

class SELFIESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.inputs = []
        print("🧬 正在编译纯净 SELFIES 序列...")
        for smi in tqdm(smiles_list):
            try:
                encoded_selfies = sf.encoder(smi)
                if encoded_selfies is None: continue
                text = f"[START]{encoded_selfies}[END]"
                tokens = tokenizer.encode(text, max_length=max_length)
                self.inputs.append(torch.tensor(tokens))
            except: pass
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx]

def is_valid_he_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or Chem.GetFormalCharge(mol) != 0 or Descriptors.NumRadicalElectrons(mol) > 0: return False
        if len(Chem.GetMolFrags(mol)) > 1: return False
        
        mw = Descriptors.ExactMolWt(mol)
        if mw < 120 or mw > 400: return False
        
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        c_count = symbols.count('C')
        n_count = symbols.count('N')
        if c_count == 0 or n_count == 0: return False

        # ⚖️ 【新增核心约束】：强行限制碳氮比，防止生成虚幻的全氮废料
        # 要求氮原子的数量不能超过碳原子的 3 倍 (N/C <= 3)
        if n_count / c_count > 3.0: return False 

        amino_pattern = Chem.MolFromSmarts('[NX3H2]')
        nitro_pattern = Chem.MolFromSmarts('[$([NX3](=O)=O),$([NX3+](=O)[O-])]')
        trinitro_carbon = Chem.MolFromSmarts('[C](-[NX3+](=O)[O-])(-[NX3+](=O)[O-])(-[NX3+](=O)[O-])')
        peroxide = Chem.MolFromSmarts('[OX2]-[OX2]')

        if not mol.HasSubstructMatch(amino_pattern) or not mol.HasSubstructMatch(nitro_pattern): return False
        if mol.HasSubstructMatch(trinitro_carbon) or mol.HasSubstructMatch(peroxide): return False
        return True
    except: return False

def check_smiles_worker(smi):
    return smi if smi and is_valid_he_molecule(smi) else None

SCAFFOLDS = [
    "c1ccccc1",           # 经典苯环 (TATB/TNT 骨架)
    "C12C3C4C1C5C2C3C45", # 立方烷 (八硝基立方烷骨架)
    "C1CC2CCC1C2",        # 降冰片烷 (高张力碳骨架)
    "C1CN2CCN1CC2",       # 类似 RDX/HMX 的多氮杂环底座
    "C1=NNC=N1",          # 三唑
    "C1=NON=C1"           # 呋咱环 (表现极好的高能骨架)
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--explore", action="store_true")
    args = parser.parse_args()

    print("==================================================")
    print(f"🌌 启动 Workflow: 大模型引擎 (Temp: {args.temperature})")
    print("==================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.explore:
        SCAFFOLDS.extend(["C1=NN=NC1", "C1(N=NC=N1)", "C1N2C3C4C1C5C2C435", "C1=NN(N=N1)", "N1=NN=NN1"])
        print("💥 [飞轮强干预] 已注入非常规高能骨架 (富氮桥环/四唑)，强制跳出局部空间！")

    df_old = pd.read_csv("../data/old_dataset.csv")
    df_train = df_old.drop_duplicates(subset=['SMILES']).copy()

    all_selfies = [sf.encoder(smi) for smi in df_train['SMILES'] if sf.encoder(smi) is not None]
    alphabet = sf.get_alphabet_from_selfies(all_selfies)
    alphabet.update(["[START]", "[END]", "[PAD]"])
    vocab = {token: idx for idx, token in enumerate(sorted(list(alphabet)))}
    idx_to_vocab = {idx: token for token, idx in vocab.items()}

    class SimpleTokenizer:
        def encode(self, text, max_length=128):
            tokens = list(sf.split_selfies(text.replace("[START]", "").replace("[END]", "")))
            tokens = ["[START]"] + tokens + ["[END]"]
            ids = [vocab.get(t, vocab.get("[PAD]")) for t in tokens]
            if len(ids) < max_length: ids += [vocab["[PAD]"]] * (max_length - len(ids))
            return ids[:max_length]
        def decode(self, ids):
            return "".join([idx_to_vocab.get(i, "") for i in ids if i not in [vocab["[PAD]"], vocab["[START]"], vocab["[END]"]]])

    tokenizer = SimpleTokenizer()
    dataset = SELFIESDataset(df_train['SMILES'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    config = GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=256, n_layer=6, n_head=8, bos_token_id=vocab["[START]"], eos_token_id=vocab["[END]"], pad_token_id=vocab["[PAD]"])
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print("🔥 正在从零训练生成引擎...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = model(batch, labels=batch).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d}/10 | Loss: {total_loss/len(dataloader):.4f}")

    num_to_generate = 50000 
    output_path = "../data/GPT_Generated_Candidates.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_cores = max(1, os.cpu_count() - 2)
    batch_size = 4096 if torch.cuda.is_available() else 64
    
    print(f"\n✨ 开启并发收割！CPU 护航核心数: {num_cores}")
    model.eval()

    generated_smiles = set()
    scaffold_ids = [[vocab["[START]"]] + [vocab.get(t, vocab["[PAD]"]) for t in list(sf.split_selfies(sf.encoder(scaf)))] for scaf in SCAFFOLDS if sf.encoder(scaf)]

    with torch.no_grad():
        pbar = tqdm(total=num_to_generate, desc="收割中")
        pool = mp.Pool(processes=num_cores)
        
        while len(generated_smiles) < num_to_generate:
            chosen_prompt = random.choice(scaffold_ids)
            input_ids = torch.tensor([chosen_prompt] * batch_size).to(device)

            # 🌡️ 动态接受来自飞轮中枢的温度指令
            generated_ids = model.generate(
                input_ids, max_length=80, do_sample=True,
                temperature=args.temperature, top_k=100, 
                pad_token_id=vocab["[PAD]"], eos_token_id=vocab["[END]"]
            )

            decoded_smiles = []
            for g_ids in generated_ids:
                try:
                    smi = sf.decoder(tokenizer.decode(g_ids.cpu().numpy()))
                    if smi and smi not in generated_smiles: decoded_smiles.append(smi)
                except: pass

            valid_results = pool.map(check_smiles_worker, decoded_smiles)
            for smi in valid_results:
                if smi is not None and len(generated_smiles) < num_to_generate:
                    generated_smiles.add(smi)
                    pbar.update(1)

            if len(generated_smiles) % 1000 == 0 and len(generated_smiles) > 0:
                pd.DataFrame([{'Molecule': f"GPT_Mutant_v3_{i:06d}", 'SMILES': s} for i, s in enumerate(generated_smiles)]).to_csv(output_path, index=False)

        pool.close()
        pool.join()
        pbar.close()

    print(f"🎉 基因筛选完成！已保存至 {output_path}")

if __name__ == "__main__":
    main()
