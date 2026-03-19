import joblib
import numpy as np

# 1. 加载训练时保存的scaler文件
scaler_path = "ultimate_high_precision_scaler.joblib"
scaler = joblib.load(scaler_path)

# 2. 提取核心参数（关键：确保特征顺序和训练时一致，x→y→z对应数组第0→1→2位）
# data_min_：每个特征的最小值（对应x、y、z三轴）
scaler_min = scaler.data_min_
# scale_：每个特征的缩放系数（1/(X_max - X_min)）
scaler_scale = scaler.scale_

# 3. 打印参数（用于复制到STM32端的C代码中）
print("=== MinMaxScaler 核心参数（复制到STM32端）===")
print(f"// 特征最小值（x, y, z）：scaler.data_min_")
print(f"const float scaler_min[3] = {{ {', '.join(map(str, scaler_min))} }};")
print(f"\n// 特征缩放系数（x, y, z）：scaler.scale_")
print(f"const float scaler_scale[3] = {{ {', '.join(map(str, scaler_scale))} }};")
print(f"scaler.data_max_ = {scaler.data_max_}")  # 验证用
print(f"scaler.data_min_ = {scaler.data_min_}")  # 验证用
# 4. 可选：验证参数正确性（输入一个原始值，查看归一化结果是否和训练时一致）
test_raw = np.array([[-5, -2, 8]])  # 模拟一组原始三轴数据
test_scaled = (test_raw - scaler_min) * scaler_scale
print(f"\n=== 验证归一化结果 ===")
print(f"原始数据：{test_raw}")
print(f"归一化结果：{test_scaled}")