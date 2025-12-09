import pandas as pd
import numpy as np
import os
import warnings
from sdmetrics.single_table import CSTest, CorrelationSimilarity

# 忽略警告
warnings.filterwarnings("ignore")

# === 配置 ===
REAL_DATA_DIR = './data/train'
SYN_DATA_DIR = './data_to_eval/paft'

DATASETS = [
    "adult", "travel", "us_location", 
    "bejing", "california_housing", "seattle_housing"
]

def evaluate_similarity():
    print(f"{'Dataset':<20} | {'Metric':<25} | {'Score':<10}")
    print("-" * 60)
    
    results = []

    for name in DATASETS:
        real_path = os.path.join(REAL_DATA_DIR, f"{name}.csv")
        syn_path = os.path.join(SYN_DATA_DIR, f"{name}_0.csv")
        
        if not os.path.exists(real_path) or not os.path.exists(syn_path):
            continue

        # 读取数据
        try:
            df_real = pd.read_csv(real_path).dropna()
            df_syn = pd.read_csv(syn_path).dropna()
            
            # 强制列对齐
            df_syn = df_syn[df_real.columns]
            
            # 采样以加快计算 (如果数据太大)
            if len(df_real) > 10000: df_real = df_real.sample(10000, random_state=42)
            if len(df_syn) > 10000: df_syn = df_syn.sample(10000, random_state=42)

            # === Table 4 复现: Column-wise Density (CSTest / Column Similarity) ===
            # SDMetrics 的 CSTest 计算的是 Chi-Squared test p-value 的平均值
            # 或者我们直接用 ColumnShapes (KSComplement) 更直观，表示分布重叠度 (0-1)
            # 论文说 "Higher values indicate more accurate"，这通常对应 `CS` (Column Similarity)
            
            # 这里我们使用 SDMetrics 的 CSTest (Chi-Squared Test) 作为统计相似性
            # 或者使用 KSComplement (数值列) + TVComplement (分类列) 的平均值
            # 为了简单，我们计算 "NewRowSynthesis" (NRS) 之外的形状相似度
            # SDMetrics 有一个综合指标叫 `NewRowSynthesis` (NRS) 和 `ColumnShapes`
            
            # 我们用 CorrelationSimilarity 来复现 Table 5
            corr_sim = CorrelationSimilarity.compute(
                real_data=df_real,
                synthetic_data=df_syn
            )
            
            print(f"{name:<20} | {'Pairwise Correlation':<25} | {corr_sim:.4f}")
            results.append({"Dataset": name, "Metric": "Correlation", "Score": corr_sim})

            # 我们用 CSTest (Chi-Square) 来近似 Table 4 (列分布)
            # 注意：CSTest 返回的是 P-value 或 Score。SDMetrics 里通常 Score 越高越好 (0-1)
            cs_score = CSTest.compute(
                real_data=df_real,
                synthetic_data=df_syn
            )
            print(f"{name:<20} | {'Column Density (CS)':<25} | {cs_score:.4f}")
            results.append({"Dataset": name, "Metric": "ColumnDensity", "Score": cs_score})

        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # 保存
    pd.DataFrame(results).to_csv("paft_similarity_results.csv", index=False)

if __name__ == "__main__":
    evaluate_similarity()