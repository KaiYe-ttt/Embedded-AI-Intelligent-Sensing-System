import numpy as np


def generate_full_range_sigmoid_table():
    # 1. 覆盖int8的输入范围：[-128, 127]（和C端一致）
    x = np.arange(-128, 128, 1, dtype=np.int8)

    # 2. 优化：放大x_float的范围（×4），让sigmoid输出覆盖更宽区间，突破原有限制
    # 保留你原有的x/256.0，再×4放大，既兼容原有逻辑，又拓宽范围
    x_float = (x / 256.0) * 4.0

    # 3. 生成sigmoid值，映射到int8的[-127, 127]全范围
    sigmoid = 1 / (1 + np.exp(-x_float))
    # 映射逻辑：将sigmoid[0,1]精准转换为int8[-127, 127]
    sigmoid_quant = (sigmoid * 254 - 127).astype(np.int8)

    # 4. 打印结果（复制到C端，替换原有sigmoid_table，tanh表无需改动）
    print("=== 全范围sigmoid查表表（复制到C端，覆盖原有表）===")
    print(f"const int8_t sigmoid_table[256] = {{ {', '.join(map(str, sigmoid_quant))} }};")


# 只生成优化后的sigmoid表（tanh表已正常，无需重新生成）
generate_full_range_sigmoid_table()