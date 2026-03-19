import numpy as np
# 核心修正：导入Python内置os模块（替换错误的np.os）
import os


def generate_stm32_lstm_header(weight_npy_path="extracted_lstm_weights.npy", header_save_path="lstm_quant_weights.h"):
    """
    完整生成适配STM32的LSTM模型C头文件（包含Dense层权重、量化参数）
    :param weight_npy_path: 提取的权重npy文件路径
    :param header_save_path: 生成的C头文件保存路径
    """
    # -------------------------- 1. 加载提取的权重文件 --------------------------
    try:
        extracted_weights = np.load(weight_npy_path, allow_pickle=True).item()
        print(f" 成功加载权重文件：{weight_npy_path}")
    except FileNotFoundError:
        print(f" 错误：未找到权重文件 {weight_npy_path}，请确认文件在当前目录下")
        return
    except Exception as e:
        print(f" 错误：加载权重文件失败 - {e}")
        return

    # -------------------------- 2. 提取核心数据（带空值判断，避免报错） --------------------------
    # 权重数据
    dense16_W = extracted_weights.get("dense16_weights", None)
    dense16_b = extracted_weights.get("dense16_bias", None)
    dense1_W = extracted_weights.get("dense1_weights", None)
    dense1_b = extracted_weights.get("dense1_bias", None)

    # 量化参数
    quant_params = extracted_weights.get("quant_params", {})
    input_scale = quant_params.get("input_scale", 0.003922)
    input_zero_point = quant_params.get("input_zero_point", -128)
    output_scale = quant_params.get("output_scale", 0.003906)
    output_zero_point = quant_params.get("output_zero_point", -128)

    # 验证核心权重是否存在
    weight_check_list = [dense16_W, dense16_b, dense1_W, dense1_b]
    if any(w is None for w in weight_check_list):
        print("  警告：部分权重数据缺失，生成的头文件可能不完整")
        print(f"  Dense16 W: {'存在' if dense16_W is not None else '缺失'}")
        print(f"  Dense16 b: {'存在' if dense16_b is not None else '缺失'}")
        print(f"  Dense1 W: {'存在' if dense1_W is not None else '缺失'}")
        print(f"  Dense1 b: {'存在' if dense1_b is not None else '缺失'}")

    # -------------------------- 3. 初始化C头文件内容（防止重复包含） --------------------------
    c_header = f"""
#ifndef LSTM_QUANT_WEIGHTS_H
#define LSTM_QUANT_WEIGHTS_H

// ==============================================================================
// STM32 LSTM Model Quantization Weights & Parameters
// Adapted to STM32F103 (int8_t quantization, low memory footprint)
// Generated automatically by Python script
// ==============================================================================

// -------------------------- Model Core Parameters --------------------------
#define LSTM_UNITS            64      // LSTM hidden layer units
#define DENSE16_UNITS         16      // Dense16 layer units
#define DENSE1_UNITS          1       // Output layer units (binary classification)
#define INPUT_FEATURE_NUM     3       // Input features (x, y, z)
#define WINDOW_LENGTH         15      // Time sequence window length

// -------------------------- Quantization Parameters --------------------------
#define INPUT_SCALE           {input_scale:.6f}
#define INPUT_ZERO_POINT      {input_zero_point}
#define OUTPUT_SCALE          {output_scale:.6f}
#define OUTPUT_ZERO_POINT     {output_zero_point}

// -------------------------- Dense16 Layer Weights (64, 16) --------------------------
// Weight matrix: LSTM_UNITS × DENSE16_UNITS
"""

    # -------------------------- 4. 写入Dense16层权重 --------------------------
    if dense16_W is not None:
        dense16_W_shape = dense16_W.shape
        c_header += f"const int8_t dense16_weights[LSTM_UNITS][DENSE16_UNITS] = {{\n"

        for i in range(dense16_W_shape[0]):
            c_header += "  {"
            for j in range(dense16_W_shape[1]):
                # 转换为原生int，避免numpy类型格式问题
                weight_val = int(dense16_W[i][j])
                c_header += f"{weight_val}, "
            # 移除末尾多余的逗号和空格，添加行尾
            c_header = c_header[:-2] + "},\n"

        # 移除最后一行多余的换行符，保持语法正确
        c_header = c_header[:-2] + "\n};\n\n"
    else:
        c_header += "const int8_t dense16_weights[LSTM_UNITS][DENSE16_UNITS] = {{ 0 }};\n\n"

    # -------------------------- 5. 写入Dense16层偏置 --------------------------
    c_header += "// Bias vector: DENSE16_UNITS\n"
    if dense16_b is not None:
        dense16_b_shape = dense16_b.shape
        c_header += f"const int8_t dense16_bias[DENSE16_UNITS] = {{\n  "

        for i in range(dense16_b_shape[0]):
            bias_val = int(dense16_b[i])
            c_header += f"{bias_val}, "

        c_header = c_header[:-2] + "\n};\n\n"
    else:
        c_header += "const int8_t dense16_bias[DENSE16_UNITS] = { 0 };\n\n"

    # -------------------------- 6. 写入Dense1层权重 --------------------------
    c_header += "// -------------------------- Dense1 (Output) Layer Weights (16, 1) --------------------------\n"
    c_header += "// Weight matrix: DENSE16_UNITS × DENSE1_UNITS\n"
    if dense1_W is not None:
        dense1_W_shape = dense1_W.shape
        c_header += f"const int8_t dense1_weights[DENSE16_UNITS][DENSE1_UNITS] = {{\n"

        for i in range(dense1_W_shape[0]):
            c_header += "  {"
            for j in range(dense1_W_shape[1]):
                weight_val = int(dense1_W[i][j])
                c_header += f"{weight_val}, "
            c_header = c_header[:-2] + "},\n"

        c_header = c_header[:-2] + "\n};\n\n"
    else:
        c_header += "const int8_t dense1_weights[DENSE16_UNITS][DENSE1_UNITS] = {{ 0 }};\n\n"

    # -------------------------- 7. 写入Dense1层偏置 --------------------------
    c_header += "// Bias vector: DENSE1_UNITS\n"
    if dense1_b is not None:
        dense1_b_shape = dense1_b.shape
        c_header += f"const int8_t dense1_bias[DENSE1_UNITS] = {{\n  "

        for i in range(dense1_b_shape[0]):
            bias_val = int(dense1_b[i])
            c_header += f"{bias_val}, "

        c_header = c_header[:-2] + "\n};\n\n"
    else:
        c_header += "const int8_t dense1_bias[DENSE1_UNITS] = { 0 };\n\n"

    # -------------------------- 8. 结束C头文件 --------------------------
    c_header += "// ==============================================================================\n"
    c_header += "#endif // LSTM_QUANT_WEIGHTS_H\n"

    # -------------------------- 9. 保存C头文件（修正os调用错误） --------------------------
    try:
        with open(header_save_path, "w", encoding="utf-8") as f:
            f.write(c_header)
        print(f" 成功生成C头文件：{header_save_path}")
        # 核心修正：使用原生os模块获取文件绝对路径
        abs_path = os.path.abspath(header_save_path)
        print(f" 文件保存路径：{abs_path}")
    except Exception as e:
        print(f"错误：保存C头文件失败 - {e}")
        return


# -------------------------- 10. 运行函数（直接调用，无需额外修改） --------------------------
if __name__ == "__main__":
    generate_stm32_lstm_header(
        weight_npy_path="extracted_lstm_weights.npy",
        header_save_path="lstm_quant_weights.h"
    )
