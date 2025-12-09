import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import os
from shapely.geometry import Point

# === 配置 ===
# 合成数据路径
SYN_DATA_PATH = './data_to_eval/paft/us_location_0.csv'
# 州界数据下载地址 (使用公开的 GeoJSON)
GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
LOCAL_GEOJSON_PATH = "us-states.json"

# 列名映射 (根据你之前的日志确认)
COL_LON = 'lon'
COL_LAT = 'lat'
COL_STATE = 'state_code'

def download_geojson():
    if not os.path.exists(LOCAL_GEOJSON_PATH):
        print(f"正在下载美国州界数据...")
        try:
            r = requests.get(GEOJSON_URL)
            with open(LOCAL_GEOJSON_PATH, 'wb') as f:
                f.write(r.content)
            print("下载完成！")
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    return True

def plot_violation_map(target_state_code):
    print(f"=== 正在分析 {target_state_code} 的边界违规情况 ===")
    
    # 1. 准备地图数据
    if not download_geojson(): return
    gdf_states = gpd.read_file(LOCAL_GEOJSON_PATH)
    
    # 这里的 GeoJSON state name 是全称 (e.g., Delaware)，我们需要映射或者手动找
    # 简单起见，我们通过 id 或 name 筛选。
    # 论文 Figure 1 用的是 Delaware (DE)。
    # 这里做一个简单的缩写映射，或者直接让用户输入全称

    state_mapping = {
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
    
    full_name = state_mapping.get(target_state_code, target_state_code)
    
    # 筛选出该州的边界多边形
    state_boundary = gdf_states[gdf_states['name'] == full_name]
    
    if state_boundary.empty:
        print(f"错误：在地图数据中找不到 {full_name}")
        return
    
    # 获取多边形几何对象
    polygon = state_boundary.geometry.values[0]

    # 2. 读取合成数据
    if not os.path.exists(SYN_DATA_PATH):
        print("错误：找不到合成数据文件")
        return
    
    df = pd.read_csv(SYN_DATA_PATH)
    # 筛选出属于该州的生成点
    df_state = df[df[COL_STATE] == target_state_code].copy()
    
    if len(df_state) == 0:
        print(f"警告：PAFT 没有生成任何 {target_state_code} 的数据")
        return

    # 3. 核心逻辑：判断点是否在多边形内
    print(f"正在计算 {len(df_state)} 个点的几何位置...")
    
    # 创建几何点列
    geometry = [Point(xy) for xy in zip(df_state[COL_LON], df_state[COL_LAT])]
    gdf_points = gpd.GeoDataFrame(df_state, geometry=geometry)
    
    # 判断每个点是否在多边形内 (contains)
    # 注意：GeoJSON通常是 (lon, lat)
    is_valid = gdf_points.within(polygon)
    
    # 统计违规率
    valid_count = is_valid.sum()
    invalid_count = len(df_state) - valid_count
    violation_rate = (invalid_count / len(df_state)) * 100
    
    print(f"  -> 合法点数 (Valid): {valid_count}")
    print(f"  -> 违规点数 (Invalid): {invalid_count}")
    print(f"  -> 违规率 (Violation Rate): {violation_rate:.2f}% (越低越好)")
    
    # 4. 绘图 (复刻论文风格)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 画边界 (黑色轮廓)
    state_boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=5)
    
    # 画内部合法的点 (蓝色)
    gdf_points[is_valid].plot(ax=ax, color='#1f77b4', markersize=15, alpha=0.6, label='Valid')
    
    # 画外部违规的点 (橙色/红色)
    if invalid_count > 0:
        gdf_points[~is_valid].plot(ax=ax, color='#ff7f0e', markersize=15, alpha=0.8, marker='x', label='Invalid')
    
    plt.title(f"PAFT Generated Data for {full_name} ({target_state_code})\nViolation Rate: {violation_rate:.2f}%")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    # 论文 Figure 5 中的 5 个案例：NM, HI, DE, WV, AK
    # 这些州的形状各异，不仅有规则的，还有像 AK 和 HI 这样破碎的岛屿
    states_to_test = ['NM', 'HI', 'DE', 'WV', 'AK']
    
    for state in states_to_test:
        plot_violation_map(state)