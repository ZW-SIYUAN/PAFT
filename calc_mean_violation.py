import pandas as pd
import geopandas as gpd
import requests
import os
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm  # 进度条库，如果没有安装 pip install tqdm

# === 配置 ===
SYN_DATA_PATH = './data_to_eval/paft/us_location_0.csv'
GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
LOCAL_GEOJSON_PATH = "us-states.json"

# 列名映射
COL_LON = 'lon'
COL_LAT = 'lat'
COL_STATE = 'state_code'

# 完整的州名映射
STATE_MAPPING = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia'
}

def download_geojson():
    if not os.path.exists(LOCAL_GEOJSON_PATH):
        print(f"正在下载美国州界数据...")
        try:
            r = requests.get(GEOJSON_URL)
            with open(LOCAL_GEOJSON_PATH, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    return True

def main():
    if not download_geojson(): return
    
    # 1. 加载数据
    print("正在加载数据...")
    gdf_states = gpd.read_file(LOCAL_GEOJSON_PATH)
    
    if not os.path.exists(SYN_DATA_PATH):
        print(f"错误：找不到合成数据 {SYN_DATA_PATH}")
        return
    df_syn = pd.read_csv(SYN_DATA_PATH)
    
    results = []
    
    # 2. 遍历所有州进行计算
    print(f"开始计算 51 个地区的违规率...")
    
    # 使用 tqdm 显示进度条
    for code, full_name in tqdm(STATE_MAPPING.items()):
        # 2.1 获取该州的边界多边形
        state_boundary = gdf_states[gdf_states['name'] == full_name]
        if state_boundary.empty:
            # print(f"警告: 地图数据中找不到 {full_name}")
            continue
        polygon = state_boundary.geometry.values[0]
        
        # 2.2 筛选该州的生成数据
        df_state = df_syn[df_syn[COL_STATE] == code].copy()
        total = len(df_state)
        
        if total == 0:
            # print(f"警告: 模型没有生成 {code} 的数据")
            continue
            
        # 2.3 计算几何包含关系
        # 显式转换经纬度为 float，防止数据类型问题
        points = [Point(float(x), float(y)) for x, y in zip(df_state[COL_LON], df_state[COL_LAT])]
        gdf_points = gpd.GeoDataFrame(df_state, geometry=points)
        
        # 判断点是否在多边形内
        is_valid = gdf_points.within(polygon)
        valid_count = is_valid.sum()
        invalid_count = total - valid_count
        violation_rate = (invalid_count / total) * 100
        
        results.append({
            'State': code,
            'Name': full_name,
            'Total': total,
            'Invalid': invalid_count,
            'Violation_Rate': violation_rate
        })
        
    # 3. 汇总统计
    if not results:
        print("没有计算出任何结果。")
        return

    df_res = pd.DataFrame(results)
    
    # 计算平均违规率 (Mean Violation Rate)
    # 论文中的 Figure 2 是各州违规率的简单平均 (Macro Average)
    mean_violation_rate = df_res['Violation_Rate'].mean()
    
    print("\n" + "="*40)
    print(f"  PAFT 全美平均违规率: {mean_violation_rate:.2f}%")
    print("="*40)
    
    # 保存结果
    output_file = 'paft_violation_report.csv'
    df_res.sort_values('Violation_Rate', inplace=True)
    df_res.to_csv(output_file, index=False)
    print(f"\n详细报告已保存至: {output_file}")
    
    # 打印表现最好和最差的 5 个州
    print("\n表现最好的 5 个州 (违规率最低):")
    print(df_res[['State', 'Violation_Rate']].head(5).to_string(index=False))
    
    print("\n表现最差的 5 个州 (违规率最高):")
    print(df_res[['State', 'Violation_Rate']].tail(5).to_string(index=False))

if __name__ == "__main__":
    main()