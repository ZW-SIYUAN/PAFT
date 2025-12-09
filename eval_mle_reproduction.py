import pandas as pd
import numpy as np
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# 忽略警告
warnings.filterwarnings("ignore")

# === 1. 配置路径 ===
REAL_DATA_DIR = './data/train'
SYN_DATA_DIR = './data_to_eval/paft'

# === 2. 任务配置 ===
TASK_CONFIG = {
    "adult": {
        "target": "income",
        "type": "classification"
    },
    "travel": {
        "target": "Target",
        "type": "classification"
    },
    "us_location": {
        "target": "state_code", 
        "type": "classification"
    },
    "bejing": {
        "target": "pm2.5",
        "type": "regression"
    },
    "california_housing": {
        "target": "median_house_value",
        "type": "regression"
    },
    "seattle_housing": {
        "target": "price",
        "type": "regression"
    }
}

# === 3. 模型参数 (论文 Table 10) ===
MODELS_CONFIG = {
    "RF": {
        "class": (RandomForestClassifier, RandomForestRegressor),
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "LR": {
        "class": (LogisticRegression, LinearRegression),
        "params": {"max_iter": 100} # LinearRegression 会动态去掉这个参数
    },
    "NN": {
        "class": (MLPClassifier, MLPRegressor),
        "params": {
            "hidden_layer_sizes": (150, 100, 50),
            "max_iter": 300,
            "learning_rate_init": 0.001,
            "random_state": 42
        }
    }
}

def load_and_preprocess(dataset_name, target_col, task_type):
    real_path = os.path.join(REAL_DATA_DIR, f"{dataset_name}.csv")
    syn_path = os.path.join(SYN_DATA_DIR, f"{dataset_name}_0.csv")
    
    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        print(f"Skipping {dataset_name}: File not found.")
        return None, None, None, None

    # 读取并丢弃空值
    df_real = pd.read_csv(real_path).dropna()
    df_syn = pd.read_csv(syn_path).dropna()

    # --- 特殊修正: Travel 数据集目标列类型 ---
    if dataset_name == 'travel':
        df_real[target_col] = df_real[target_col].astype(int)
        df_syn[target_col] = pd.to_numeric(df_syn[target_col], errors='coerce')
        df_syn.dropna(subset=[target_col], inplace=True)
        df_syn[target_col] = df_syn[target_col].astype(int)

    # --- 1. 强制列对齐 (PAFT生成的列顺序可能不同) ---
    df_syn = df_syn[df_real.columns]

    # --- 2. 划分真实测试集 (TSTR 策略) ---
    # 80% Real 用于训练的替补(虽然这里没用上), 20% Real 用于最终测试
    _, df_real_test = train_test_split(df_real, test_size=0.2, random_state=42)

    # 训练集 = 全部合成数据
    df_train = df_syn.copy()
    # 测试集 = 20% 真实数据
    df_test = df_real_test.copy()

    # --- 3. 智能编码 (修复回归任务被 LabelEncode 的 Bug) ---
    for col in df_train.columns:
        # 判断是否为回归任务的目标列
        is_regression_target = (col == target_col and task_type == 'regression')

        # A. 如果是回归目标：强制转数值，不做 LabelEncode
        if is_regression_target:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
            # 清洗转换失败的非数字
            df_train.dropna(subset=[col], inplace=True)
            df_test.dropna(subset=[col], inplace=True)
            continue 

        # B. 如果是文本特征 OR 分类任务目标：做 LabelEncode
        if df_train[col].dtype == 'object' or (col == target_col and task_type == 'classification'):
            le = LabelEncoder()
            
            # 统一转字符串，合并训练集和测试集的值来训练 Encoder，防止 Unseen label 报错
            train_vals = df_train[col].astype(str).tolist()
            test_vals = df_test[col].astype(str).tolist()
            
            all_vals = list(set(train_vals + test_vals))
            le.fit(all_vals)
            
            df_train[col] = le.transform(train_vals)
            df_test[col] = le.transform(test_vals)

    # --- 4. 分离特征和标签 ---
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # --- 5. 归一化 (对 NN 和 LR 很重要) ---
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def evaluate_models():
    # 打印表头
    print(f"{'Dataset':<20} | {'Model':<5} | {'Metric':<10} | {'Score':<10}")
    print("-" * 55)

    results_summary = []

    for name, config in TASK_CONFIG.items():
        target = config["target"]
        task_type = config["type"]
        
        try:
            # 加载数据
            X_train, y_train, X_test, y_test = load_and_preprocess(name, target, task_type)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

        if X_train is None: continue

        for model_name, model_cfg in MODELS_CONFIG.items():
            # 获取模型类
            ModelClass = model_cfg["class"][0] if task_type == "classification" else model_cfg["class"][1]
            
            # --- 动态参数处理 ---
            current_params = model_cfg["params"].copy()
            # 如果是 LinearRegression，删除 max_iter 参数防止报错
            if ModelClass == LinearRegression and "max_iter" in current_params:
                del current_params["max_iter"]

            # 初始化模型
            model = ModelClass(**current_params)
            
            try:
                # 训练 (在合成数据上)
                model.fit(X_train, y_train)
                # 预测 (在真实数据上)
                preds = model.predict(X_test)
                
                # 计算指标
                if task_type == "classification":
                    # 准确率
                    score = accuracy_score(y_test, preds)
                    metric_name = "Accuracy"
                    score_display = f"{score*100:.2f}%"
                else:
                    # MAPE (回归)
                    mask = y_test != 0
                    if mask.sum() > 0:
                        score = mean_absolute_percentage_error(y_test[mask], preds[mask])
                    else:
                        score = 0.0
                    metric_name = "MAPE"
                    score_display = f"{score:.4f}" # 0.22 表示 22% 误差

                print(f"{name:<20} | {model_name:<5} | {metric_name:<10} | {score_display:<10}")
                
                results_summary.append({
                    "Dataset": name,
                    "Model": model_name,
                    "Metric": metric_name,
                    "Score": score
                })

            except Exception as e:
                print(f"{name:<20} | {model_name:<5} | FAILED | {str(e)[:20]}...")

    # 保存结果
    if results_summary:
        pd.DataFrame(results_summary).to_csv("paft_mle_results_final.csv", index=False)
        print("\n详细 MLE 结果已保存至 paft_mle_results_final.csv")

if __name__ == "__main__":
    evaluate_models()