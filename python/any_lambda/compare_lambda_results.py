#!/usr/bin/env python3
"""
Script để so sánh kết quả benchmark với các giá trị lambda khác nhau
"""
import subprocess
import sys
import re
from collections import defaultdict
import pandas as pd

def run_benchmark(lambda_val, sizes=[100, 200, 400, 1000, 2000, 5000], repeats=7):
    """Chạy benchmark cho một giá trị lambda"""
    cmd = [
        sys.executable, "benchmark_svm_tree_vs_svc_2n.py",
        "--sizes"] + [str(s) for s in sizes] + [
        "--repeats", str(repeats),
        "--lamda", str(lambda_val)
    ]
    
    print(f"🔄 Đang chạy benchmark với λ = {lambda_val}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Lỗi khi chạy λ = {lambda_val}: {result.stderr}")
        return None
    
    return result.stdout

def parse_results(output):
    """Parse kết quả từ output của benchmark"""
    results = []
    
    # Regex để parse từng dòng kết quả
    pattern = r'N=\s*(\d+).*?FIT.*?(\d+\.\d+)x.*?PRED.*?(\d+\.\d+)x.*?ACC.*?(\d+\.\d+).*?vs.*?(\d+\.\d+).*?Support=\(s=(\d+), p=(\d+)\), Loss=([-\d+\.\d+]+)'
    
    matches = re.findall(pattern, output, re.DOTALL)
    
    for match in matches:
        n, fit_speedup, pred_speedup, tree_acc, svc_acc, s_support, p_support, loss = match
        results.append({
            'N': int(n),
            'FIT_Speedup': float(fit_speedup),
            'PRED_Speedup': float(pred_speedup),
            'Tree_Accuracy': float(tree_acc),
            'SVC_Accuracy': float(svc_acc),
            'S_Support': int(s_support),
            'P_Support': int(p_support),
            'Loss': float(loss)
        })
    
    return results

def main():
    # Các giá trị lambda để test
    lambda_values = [1.0, 2.0, 5.0, 10.0, 30.0]
    
    all_results = {}
    
    # Chạy benchmark cho từng lambda
    for lambda_val in lambda_values:
        output = run_benchmark(lambda_val)
        if output:
            results = parse_results(output)
            all_results[lambda_val] = results
            print(f"✅ Hoàn thành λ = {lambda_val}")
        else:
            print(f"❌ Thất bại λ = {lambda_val}")
    
    # Tạo bảng so sánh
    print("\n" + "="*80)
    print("📊 BẢNG SO SÁNH KẾT QUẢ THEO LAMBDA")
    print("="*80)
    
    # Tạo DataFrame cho mỗi metric
    sizes = [100, 200, 400, 1000, 2000, 5000, 10000]  # Có thể có 10000 từ kết quả trước
    
    # Bảng FIT Speedup
    print("\n🚀 FIT SPEEDUP (lần)")
    fit_data = defaultdict(dict)
    for lambda_val, results in all_results.items():
        for result in results:
            fit_data[result['N']][f"λ={lambda_val}"] = f"{result['FIT_Speedup']:.2f}x"
    
    fit_df = pd.DataFrame(fit_data).T
    print(fit_df.to_string())
    
    # Bảng PRED Speedup
    print("\n⚡ PREDICTION SPEEDUP (lần)")
    pred_data = defaultdict(dict)
    for lambda_val, results in all_results.items():
        for result in results:
            pred_data[result['N']][f"λ={lambda_val}"] = f"{result['PRED_Speedup']:.1f}x"
    
    pred_df = pd.DataFrame(pred_data).T
    print(pred_df.to_string())
    
    # Bảng Accuracy
    print("\n🎯 ACCURACY COMPARISON")
    acc_data = defaultdict(dict)
    for lambda_val, results in all_results.items():
        for result in results:
            tree_acc = result['Tree_Accuracy']
            svc_acc = result['SVC_Accuracy']
            diff = tree_acc - svc_acc
            acc_data[result['N']][f"λ={lambda_val}"] = f"{tree_acc:.3f} ({diff:+.3f})"
    
    acc_df = pd.DataFrame(acc_data).T
    print(acc_df.to_string())
    
    # Bảng Loss
    print("\n📉 LOSS VALUES")
    loss_data = defaultdict(dict)
    for lambda_val, results in all_results.items():
        for result in results:
            loss_data[result['N']][f"λ={lambda_val}"] = f"{result['Loss']:.1f}"
    
    loss_df = pd.DataFrame(loss_data).T
    print(loss_df.to_string())
    
    # Tóm tắt
    print("\n" + "="*80)
    print("📋 TÓM TẮT")
    print("="*80)
    
    for lambda_val, results in all_results.items():
        if results:
            max_fit = max(r['FIT_Speedup'] for r in results)
            max_pred = max(r['PRED_Speedup'] for r in results)
            avg_acc_diff = sum(r['Tree_Accuracy'] - r['SVC_Accuracy'] for r in results) / len(results)
            
            print(f"λ = {lambda_val:4.1f}: Max FIT = {max_fit:5.2f}x, Max PRED = {max_pred:7.1f}x, Avg Acc Diff = {avg_acc_diff:+.3f}")

if __name__ == "__main__":
    main()
