import re
import argparse
import os
import csv
import math

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Regex mềm cho từng dòng
RE_N    = re.compile(r"^N=\s*(\d+)\b.*2N", re.IGNORECASE)
RE_FIT  = re.compile(r"FIT\s+SVM-On-Tree:\s*([0-9.]+)s.*sklearn SVC:\s*([0-9.]+)s", re.IGNORECASE)
RE_PRED = re.compile(r"PRED\s+SVM-On-Tree:\s*([0-9.]+)s.*sklearn SVC:\s*([0-9.]+)s", re.IGNORECASE)
RE_ACC  = re.compile(r"ACC\s+SVM-On-Tree:\s*([0-9.]+)\s*.*sklearn SVC:\s*([0-9.]+)", re.IGNORECASE)

def parse_log_lines(lines):
    rows = []
    cur = {}

    def flush_if_complete():
        if all(k in cur for k in ("N","fit_tree","fit_svc","pred_tree","pred_svc","acc_tree","acc_svc")):
            rows.append({
                "N": cur["N"],
                "fit_tree_s": cur["fit_tree"],
                "fit_svc_s": cur["fit_svc"],
                "pred_tree_s": cur["pred_tree"],
                "pred_svc_s": cur["pred_svc"],
                "acc_tree": cur["acc_tree"],
                "acc_svc": cur["acc_svc"],
                "speedup_fit": (cur["fit_svc"]/cur["fit_tree"]) if cur["fit_tree"]>0 else math.inf,
                "speedup_pred": (cur["pred_svc"]/cur["pred_tree"]) if cur["pred_tree"]>0 else math.inf,
            })
            return True
        return False

    for raw in lines:
        line = raw.strip()

        mN = RE_N.search(line)
        if mN:
            # nếu block trước đã đủ, đẩy vào rows
            flush_if_complete()
            cur = {"N": int(mN.group(1))}
            continue

        if "FIT" in line:
            m = RE_FIT.search(line)
            if m:
                cur["fit_tree"] = float(m.group(1))
                cur["fit_svc"]  = float(m.group(2))
            continue

        if "PRED" in line:
            m = RE_PRED.search(line)
            if m:
                cur["pred_tree"] = float(m.group(1))
                cur["pred_svc"]  = float(m.group(2))
            continue

        if "ACC" in line:
            m = RE_ACC.search(line)
            if m:
                cur["acc_tree"] = float(m.group(1))
                cur["acc_svc"]  = float(m.group(2))
            continue

    # flush lần cuối
    flush_if_complete()
    rows.sort(key=lambda r: r["N"])
    return rows

def save_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def plot_series(xs, ys_dict, title, xlabel, ylabel, outpath, logy=False):
    if not HAS_MPL:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.5,5))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=str)
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()

    if not os.path.isfile(args.logfile):
        print(f"Không tìm thấy logfile: {args.logfile}")
        return

    with open(args.logfile, "r") as f:
        lines = f.readlines()

    rows = parse_log_lines(lines)
    if not rows:
        print("No results parsed. Make sure the log format matches your benchmark output.")
        return

    # CSV
    csv_path = os.path.join(args.outdir, "results.csv")
    save_csv(rows, csv_path)
    print(f"[OK] Đã lưu CSV -> {csv_path}")

    # Dữ liệu vẽ
    N = [r["N"] for r in rows]
    fit_tree = [r["fit_tree_s"] for r in rows]
    fit_svc  = [r["fit_svc_s"]  for r in rows]
    pred_tree= [r["pred_tree_s"] for r in rows]
    pred_svc = [r["pred_svc_s"]  for r in rows]
    acc_tree = [r["acc_tree"] for r in rows]
    acc_svc  = [r["acc_svc"]  for r in rows]
    sp_fit   = [r["speedup_fit"] for r in rows]
    sp_pred  = [r["speedup_pred"] for r in rows]

    # Plot (nếu có matplotlib)
    if HAS_MPL:
        plot_series(N,
                    {"SVM-On-Tree": fit_tree, "sklearn SVC": fit_svc},
                    "Training time vs N", "N (per class; eval on 2N)", "Time (s)",
                    os.path.join(args.outdir, "time_fit.png"), logy=True)

        plot_series(N,
                    {"SVM-On-Tree": pred_tree, "sklearn SVC": pred_svc},
                    "Prediction time vs N", "N (per class; eval on 2N)", "Time (s)",
                    os.path.join(args.outdir, "time_pred.png"), logy=True)

        plot_series(N,
                    {"SVM-On-Tree": acc_tree, "sklearn SVC": acc_svc},
                    "Accuracy vs N (2N evaluation)", "N", "Accuracy",
                    os.path.join(args.outdir, "acc.png"), logy=False)

        plot_series(N,
                    {"Speedup FIT (SVC/Tree)": sp_fit},
                    "Speedup (Training) vs N", "N", "Speedup (×)",
                    os.path.join(args.outdir, "speedup_fit.png"), logy=True)

        plot_series(N,
                    {"Speedup PRED (SVC/Tree)": sp_pred},
                    "Speedup (Prediction) vs N", "N", "Speedup (×)",
                    os.path.join(args.outdir, "speedup_pred.png"), logy=True)

        print(f"[OK] Đã lưu biểu đồ -> {args.outdir}/time_fit.png, time_pred.png, acc.png, speedup_fit.png, speedup_pred.png")
    else:
        print("[Note] Chưa cài matplotlib; đã xuất CSV, bỏ qua hình.")

    # In bảng tóm tắt
    print("\n== Tóm tắt đã parse ==")
    for r in rows:
        print(f"N={r['N']:>6} | FIT Tree/SVC: {r['fit_tree_s']:.6f}s / {r['fit_svc_s']:.6f}s"
              f" | PRED Tree/SVC: {r['pred_tree_s']:.6f}s / {r['pred_svc_s']:.6f}s"
              f" | ACC Tree/SVC: {r['acc_tree']:.4f} / {r['acc_svc']:.4f}"
              f" | Speedup FIT/PRED: {r['speedup_fit']:.2f}x / {r['speedup_pred']:.2f}x")

if __name__ == "__main__":
    main()
