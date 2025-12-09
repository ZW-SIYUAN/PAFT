import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 忽略警告
warnings.filterwarnings("ignore")

# === 配置 ===
REAL_DATA_DIR = './data/train'
SYN_DATA_DIR = './data_to_eval/paft'

# 定义需要评估的数据集
DATASETS = [
    "adult", "travel", "us_location", 
    "bejing", "california_housing", "seattle_housing"
]

# 鉴别器模型参数 (通常使用简单的 MLP)
DISCRIMINATOR_CONFIG = {
    "hidden_layer_sizes": (100, 50),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 500,
    "random_state": 42
}

def load_and_prepare_data(dataset_name):
    real_path = os.path.join(REAL_DATA_DIR, f"{dataset_name}.csv")
    syn_path = os.path.join(SYN_DATA_DIR, f"{dataset_name}_0.csv")
    
    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        print(f"Skipping {dataset_name}: File not found.")
        return None, None

    # 1. 读取数据
    df_real = pd.read_csv(real_path).dropna()
    df_syn = pd.read_csv(syn_path).dropna()
    
    # 2. 列对齐
    # 确保合成数据列顺序与真实数据一致
    common_cols = [c for c in df_real.columns if c in df_syn.columns]
    df_real = df_real[common_cols]
    df_syn = df_syn[common_cols]

    # 3. 严格平衡数据量 (50% Real, 50% Fake)
    # 这一步至关重要，否则 Baseline 就不是 50% 了
    min_len = min(len(df_real), len(df_syn))
    df_real = df_real.sample(n=min_len, random_state=42)
    df_syn = df_syn.sample(n=min_len, random_state=42)

    # 4. 打标签
    df_real['__is_real__'] = 1
    df_syn['__is_real__'] = 0

    # 5. 合并
    df_combined = pd.concat([df_real, df_syn], axis=0)

    # 6. 预处理 (Label Encoding + Normalization)
    features = [c for c in df_combined.columns if c != '__is_real__']
    target = '__is_real__'

    for col in features:
        if df_combined[col].dtype == 'object':
            le = LabelEncoder()
            df_combined[col] = le.fit_transform(df_combined[col].astype(str))
    
    # 分离 X, y
    X = df_combined[features]
    y = df_combined[target]

    # 归一化 (对 MLP 很重要)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def run_discriminator_experiment():
    print(f"{'Dataset':<20} | {'Metric':<20} | {'Score':<10} | {'Ideal':<10}")
    print("-" * 65)
    
    results = []

    for name in DATASETS:
        try:
            X, y = load_and_prepare_data(name)
            if X is None: continue

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # 初始化鉴别器 (MLP)
            clf = MLPClassifier(**DISCRIMINATOR_CONFIG)
            
            # 训练
            clf.fit(X_train, y_train)
            
            # 预测
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)

            # 输出结果
            # Score 越接近 0.5 (50%) 越好
            print(f"{name:<20} | {'Discriminator Acc.':<20} | {acc*100:.2f}%     | 50.00%")
            
            results.append({
                "Dataset": name,
                "Discriminator Accuracy": acc
            })

        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # 保存结果
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv("paft_discriminator_results.csv", index=False)
        print("\n详细鉴别器结果已保存至 paft_discriminator_results.csv")

if __name__ == "__main__":
    run_discriminator_experiment()