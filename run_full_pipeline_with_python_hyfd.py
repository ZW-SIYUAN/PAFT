import os
import json
import argparse
import pandas as pd
import sys
import contextlib

# 确保能导入 hyfd
# 如果 hyfd.py 在当前目录，直接 import
# 如果报错 import error，请确保 hyfd_libs 文件夹在当前目录下
try:
    from hyfd import HyFd
except ImportError:
    print("错误: 无法导入 hyfd。请确保 'hyfd.py' 和 'hyfd_libs' 文件夹在当前目录下。")
    sys.exit(1)

# 模拟 HyFD 需要的参数类
class MockArgs:
    def __init__(self, db_path):
        self.db_path = db_path
        self.separator = ','
        self.ignore_headers = False # 通常 pandas 读取带 header，这里设为 False 让 HyFD 处理原始数据
        self.efft = 0.01
        self.lf = 0.5
        self.ift = 0.01
        self.el = 10e-15
        self.debug = False
        self.mute = True # 静音，不要打印太多日志
        self.logfile = False
        self.restart = False

def run_hyfd_and_save_json():
    # 1. 准备目录
    os.makedirs("fd_finder/json", exist_ok=True)
    os.makedirs("fd_finder/datasets", exist_ok=True)
    
    # 2. 定义要处理的数据集 (确保这些csv在 data/train 下)
    # 如果没有真实数据，请先运行之前提供的 '生成模拟数据' 的脚本
    data_dir = "data/train"
    if not os.path.exists(data_dir):
        print(f"错误: 找不到数据目录 {data_dir}")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    print(f"=== 开始处理 {len(csv_files)} 个数据集 ===")

    for csv_file in csv_files:
        dataset_name = csv_file.split('.')[0]
        input_path = os.path.join(data_dir, csv_file)
        
        # 复制一份到 fd_finder/datasets (因为 paft 脚本也去那里读)
        target_csv_path = f"fd_finder/datasets/{csv_file}"
        df = pd.read_csv(input_path)
        df.to_csv(target_csv_path, index=False)
        
        print(f"\n正在通过 HyFD 分析: {dataset_name} ...")

        # 构造参数并运行 HyFD
        args = MockArgs(target_csv_path)
        # 必须设为 True 以跳过 header 行，因为 HyFD 处理的是纯数据矩阵
        args.ignore_headers = True 
        
        # 捕获 HyFD 实例
        # 注意：hyfd.py 的 HyFd 类在 __init__ 里就执行了 self.execute()
        # 我们需要抑制它的标准输出，只获取结果
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            hyfd_instance = HyFd(args)
        
        # 获取列名映射 (Column Name -> Index)
        col_name_to_idx = {name: i for i, name in enumerate(df.columns)}
        
        # 提取 FD 并转换为 PAFT 需要的索引格式
        # PAFT 格式: list of [[lhs_indices], [rhs_indices]]
        paft_json_structure = []
        
        # hyfd_instance.get_fds() 返回生成器，产出 ([lhs_names], [rhs_names])
        # 注意：这里的 names 其实可能是原始数据的值，我们需要确认 hyfd 是怎么处理 header 的。
        # 如果 args.ignore_headers=True，HyFD 内部 self.records 不包含 header。
        # 它的 get_fds 返回的是 PLI 对象的 .att 属性。
        # 在 HyFD 代码中，如果没有 header，它可能用索引或者第一行数据作为属性名。
        # 为了稳妥，我们需要修改一下获取逻辑，直接利用 hyfd 内部的 fds 结构。
        
        # 重新读取 HyFD 内部发现的 FD
        # hyfd.py 的 get_fds() yield ([lhs_names], [rhs_names])
        # 但我们传入了 ignore_headers=True，所以 HyFD 内部把第一行丢了，
        # 它的列索引就是 0, 1, 2...
        # 让我们直接读取 hyfd_instance.fds.read_fds()，它返回索引 (lhs_indices, rhs_indices)
        
        count = 0
        for lhs_idxs, rhs_idxs in hyfd_instance.fds.read_fds():
            # hyfd 返回的是 set 或 list 的索引
            lhs_list = list(lhs_idxs)
            rhs_list = list(rhs_idxs)
            
            # PAFT 的 json 格式： [ [lhs...], [rhs...] ]
            paft_json_structure.append([lhs_list, rhs_list])
            count += 1

        print(f"  -> 发现 {count} 条函数依赖")

        # 保存为 JSON
        json_output_path = f"fd_finder/json/{dataset_name}.json"
        with open(json_output_path, "w") as f:
            json.dump(paft_json_structure, f)
        print(f"  -> 已保存: {json_output_path}")

if __name__ == "__main__":
    run_hyfd_and_save_json()
    
    print("\n=== 第一阶段完成 ===")
    print("接下来请运行:")
    print("1. python paft_fd_distilation_and_optimization.py")
    print("2. python paft_fine_tuning.py --train --epochs 50")