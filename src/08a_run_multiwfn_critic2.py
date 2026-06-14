import os
import argparse
import subprocess
import concurrent.futures
import glob

# ==========================================
# 路径与命令配置
# ==========================================
DEFAULT_WORK_DIR = "../temp_calc"
MULTIWFN_CMD = "/home/gma/bzhang/soft/Multiwfn_2026.2.2_bin_Linux_noGUI/Multiwfn_noGUI"
CRITIC2_CMD = "/home/gma/bzhang/critic2/build/src/critic2"
ORCA2MKL_CMD = "/home/soft/amd/codes/orca/v6.0.0-gcc13.2.0/orca_2mkl"


def ensure_molden(mol_dir_path):
    """
    Ensure a *.molden.input file exists.
    If not, try to generate it from *.gbw using orca_2mkl.
    """
    molden_files = glob.glob(os.path.join(mol_dir_path, "*.molden.input"))
    if molden_files:
        return molden_files[0], "existing"

    gbw_files = glob.glob(os.path.join(mol_dir_path, "*.gbw"))
    if not gbw_files:
        return None, "no_gbw"

    gbw_file = gbw_files[0]
    base = os.path.splitext(os.path.basename(gbw_file))[0]

    try:
        subprocess.run(
            [ORCA2MKL_CMD, base, "-molden"],
            cwd=mol_dir_path,
            timeout=600,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return None, f"orca_2mkl_failed: {e}"

    molden_files = glob.glob(os.path.join(mol_dir_path, "*.molden.input"))
    if molden_files:
        return molden_files[0], "generated"

    return None, "molden_generation_failed"


def process_molecule(args_tuple):
    mol_dir_path, force, run_esp, run_qtaim = args_tuple
    mol_name = os.path.basename(mol_dir_path)

    if not os.path.isdir(mol_dir_path):
        return f"❌ {mol_name}: 目录不存在，跳过。"

    molden_file, molden_status = ensure_molden(mol_dir_path)
    if not molden_file:
        return f"❌ {mol_name}: 找不到/无法生成 molden.input ({molden_status})，跳过。"

    molden_filename = os.path.basename(molden_file)
    msg = f"[{mol_name}] molden={molden_status} "

    # --------------------------------------------------
    # 1. Multiwfn ESP / molecular surface descriptors
    # --------------------------------------------------
    esp_out = os.path.join(mol_dir_path, "esp_output.txt")
    if run_esp:
        if force and os.path.exists(esp_out):
            os.remove(esp_out)

        if not os.path.exists(esp_out):
            inp = "12\n0\n\n\n"
            bash_command = (
                f"ulimit -s unlimited && "
                f"export OMP_STACKSIZE=2G && "
                f"{MULTIWFN_CMD} {molden_filename}"
            )
            try:
                res_m = subprocess.run(
                    bash_command,
                    input=inp,
                    cwd=mol_dir_path,
                    text=True,
                    capture_output=True,
                    timeout=1800,
                    shell=True,
                    executable="/bin/bash",
                )
                with open(esp_out, "w", encoding="utf-8") as f:
                    f.write(res_m.stdout or "")
                    if res_m.stderr:
                        f.write("\n\n=== STDERR ===\n" + res_m.stderr)
                msg += "✅ ESP完成 "
            except Exception as e:
                msg += f"❌ ESP失败({e}) "
        else:
            msg += "⚡ ESP已存在 "
    else:
        msg += "⏭ ESP跳过 "

    # --------------------------------------------------
    # 2. Critic2 QTAIM
    # --------------------------------------------------
    critic_out = os.path.join(mol_dir_path, "critic2_cpreport.out")
    if run_qtaim:
        if force and os.path.exists(critic_out):
            os.remove(critic_out)

        if not os.path.exists(critic_out):
            inp_content = f"molecule {molden_filename}\nauto\ncpreport\n"
            inp_file = os.path.join(mol_dir_path, "critic2_fast.inp")
            try:
                with open(inp_file, "w", encoding="utf-8") as f:
                    f.write(inp_content)

                res_c = subprocess.run(
                    [CRITIC2_CMD, "critic2_fast.inp"],
                    cwd=mol_dir_path,
                    timeout=300,
                    capture_output=True,
                    text=True,
                )
                with open(critic_out, "w", encoding="utf-8") as f:
                    f.write(res_c.stdout or "")
                    if res_c.stderr:
                        f.write("\n\n=== STDERR ===\n" + res_c.stderr)
                msg += "✅ QTAIM完成"
            except Exception as e:
                msg += f"❌ QTAIM失败({e})"
        else:
            msg += "⚡ QTAIM已存在"
    else:
        msg += "⏭ QTAIM跳过"

    return msg


def collect_dirs(work_dir, explicit_dirs):
    if explicit_dirs:
        dirs = []
        for d in explicit_dirs:
            d = os.path.abspath(os.path.expanduser(d))
            if os.path.isdir(d):
                dirs.append(d)
            else:
                print(f"[WARN] explicit dir not found: {d}")
        return dirs

    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    return [
        os.path.join(work_dir, d)
        for d in os.listdir(work_dir)
        if os.path.isdir(os.path.join(work_dir, d))
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run Multiwfn ESP and Critic2 QTAIM for selected molecule directories."
    )
    parser.add_argument("--work_dir", default=DEFAULT_WORK_DIR,
                        help="Directory containing molecule subfolders. Default: ../temp_calc")
    parser.add_argument("--dirs", nargs="*", default=None,
                        help="Explicit molecule directories to process. Overrides --work_dir scan.")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Recompute esp_output.txt and critic2_cpreport.out if they already exist.")
    parser.add_argument("--no_esp", action="store_true")
    parser.add_argument("--no_qtaim", action="store_true")
    args = parser.parse_args()

    print("==================================================")
    print("🚀 启动物理特征补全行动 (Multiwfn + Critic2)")
    print("==================================================")

    dirs = collect_dirs(args.work_dir, args.dirs)
    dirs = sorted(dirs)

    print(f"[INFO] 待处理目录数: {len(dirs)}")
    for d in dirs[:20]:
        print("  -", d)
    if len(dirs) > 20:
        print(f"  ... plus {len(dirs) - 20} more")

    if not dirs:
        print("[WARN] 没有可处理目录。")
        return

    payload = [
        (d, args.force, not args.no_esp, not args.no_qtaim)
        for d in dirs
    ]

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for result in executor.map(process_molecule, payload):
            completed += 1
            print(f"[{completed}/{len(dirs)}] {result}")

    print("\n🎉 物理特征补充完毕！")


if __name__ == "__main__":
    main()
