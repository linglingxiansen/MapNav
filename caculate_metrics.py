import pandas as pd
import numpy as np

def calculate_metrics(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 计算调整后的success
    df['adjusted_success'] = df['success'].copy()
    df.loc[(df['success'] == 0) & (df['distance_to_goal'] <= 3.2), 'adjusted_success'] = 1
    
    # 计算调整后的SPL
    df['adjusted_spl'] = df['spl'].copy()
    df.loc[(df['success'] == 0) & (df['distance_to_goal'] <= 3.2), 'adjusted_spl'] = 1.0
    
    # 计算整体指标
    total_samples = len(df)
    original_success_rate = df['success'].mean()
    adjusted_success_rate = df['adjusted_success'].mean()
    
    # 计算平均SPL（除以总样本数）
    original_spl = df['spl'].sum() / total_samples  # 修改：除以总样本数
    adjusted_spl = df['adjusted_spl'].sum() / total_samples  # 修改：除以总样本数
    ne = df['distance_to_goal'].sum() / total_samples
    orcale_success = df['orcale_success'].sum()/total_samples
    print(f"总样本数: {total_samples}")
    print(f"原始成功率: {original_success_rate:.4f}")
    print(f"调整后成功率: {adjusted_success_rate:.4f}")
    print(f"原始平均SPL: {original_spl:.4f}")
    print(f"调整后平均SPL: {adjusted_spl:.4f}")
    print(f"ne: {ne:.4f}")
    print(f"os: {orcale_success:.4f}")
    # 输出百分比格式
    print("\n百分比格式：")
    print(f"原始成功率: {original_success_rate*100:.1f}%")
    print(f"调整后成功率: {adjusted_success_rate*100:.1f}%")
    print(f"原始平均SPL: {original_spl*100:.1f}%")
    print(f"调整后平均SPL: {adjusted_spl*100:.1f}%")
    
    return df

# 使用示例
df = calculate_metrics('2hisdagger.csv')