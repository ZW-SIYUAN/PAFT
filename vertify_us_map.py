import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置路径
real_data_path = './data/train/us_location.csv'
syn_data_path = './data_to_eval/paft/us_location_0.csv'

# 【关键修改】这里定义你数据里真实的列名
COL_LON = 'lon'          # 经度列名
COL_LAT = 'lat'          # 纬度列名
COL_STATE = 'state_code' # 州名列名

def plot_state_map(state_code):
    print(f"正在绘制 {state_code} 的地图对比...")
    
    if not os.path.exists(real_data_path):
        print("错误：找不到真实数据，无法对比边界。")
        return
    if not os.path.exists(syn_data_path):
        print("错误：找不到合成数据，请检查路径。")
        return

    df_real = pd.read_csv(real_data_path)
    df_syn = pd.read_csv(syn_data_path)
    
    # 筛选数据
    real_state = df_real[df_real[COL_STATE] == state_code]
    syn_state = df_syn[df_syn[COL_STATE] == state_code]

    if len(syn_state) == 0:
        print(f"警告：合成数据中没有生成 {state_code} 的样本！")
        return

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 真实数据
    sns.scatterplot(data=real_state, x=COL_LON, y=COL_LAT, ax=axes[0], s=10, color='blue', alpha=0.5)
    axes[0].set_title(f'Real Data ({state_code})')
    axes[0].set_xlim(-180, -60) 
    axes[0].set_ylim(15, 75)
    
    # 合成数据 (PAFT)
    sns.scatterplot(data=syn_state, x=COL_LON, y=COL_LAT, ax=axes[1], s=10, color='red', alpha=0.5)
    axes[1].set_title(f'PAFT Generated ({state_code})')
    axes[1].set_xlim(-180, -60)
    axes[1].set_ylim(15, 75)

    plt.show()


'''
if __name__ == "__main__":
    print("绘制全美地图...")
    
    if not os.path.exists(syn_data_path):
        print(f"错误：找不到文件 {syn_data_path}")
    else:
        df_syn = pd.read_csv(syn_data_path)
        
        # 打印一下前几行，确保列名正确
        print("数据列名:", df_syn.columns.tolist())
        
        plt.figure(figsize=(12, 7))
        # hue=COL_STATE 让不同州显示不同颜色
        sns.scatterplot(data=df_syn, x=COL_LON, y=COL_LAT, hue=COL_STATE, s=5, legend=False, palette='viridis')
        plt.title("PAFT Generated US Map (All States)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
        print("绘图完成！")
'''
if __name__ == "__main__":
    # === 修改这里：不再画全美，而是指定画论文里的 DE (Delaware) ===
    
    # 论文 Figure 1 的主角：特拉华州
    plot_state_map('DE') 
    
    # 你也可以顺便看看加州 (形状很有辨识度)
    plot_state_map('HI')
    
    # 或者看看德州
    plot_state_map('WV')

    plot_state_map('AK')