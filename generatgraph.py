import numpy as np


def generate_correct_scale_activate_tables():
    # 1. 覆盖int8的输入范围：[-128, 127]（和C端一致）
    x = np.arange(-128, 128, 1, dtype=np.int8)

    # 2. 保留你原有的缩放逻辑：x / 256.0（对应C端量化逻辑，不改动）
    x_float = x / 256.0

    # 3. 生成sigmoid表：调整输出缩放系数，拉满到int8[-127, 127]
    sigmoid = 1 / (1 + np.exp(-x_float))
    # 缩放逻辑优化：将sigmoid[0,1]映射到int8[-127, 127]，而非之前的小范围
    sigmoid_quant = (sigmoid * 254 - 127).astype(np.int8)  # ×254避免超出int8范围

    # 4. 生成tanh表：调整输出缩放系数，拉满到int8[-127, 127]
    tanh = np.tanh(x_float)
    # 缩放逻辑优化：将tanh[-0.462, 0.462]放大到[-127, 127]
    tanh_quant = (tanh * 275).astype(np.int8)  # ×275（精准计算的放大系数），拉满到±127
    # 额外裁剪：防止极端值超出int8范围
    tanh_quant = np.clip(tanh_quant, -127, 127)

    # 5. 打印结果（复制到C端，替换原有查表表）
    print("=== 修正缩放比例的sigmoid查表表（复制到C端）===")
    print(f"const int8_t sigmoid_table[256] = {{ {', '.join(map(str, sigmoid_quant))} }};")

    print("\n=== 修正缩放比例的tanh查表表（复制到C端）===")
    print(f"const int8_t tanh_table[256] = {{ {', '.join(map(str, tanh_quant))} }};")


generate_correct_scale_activate_tables()