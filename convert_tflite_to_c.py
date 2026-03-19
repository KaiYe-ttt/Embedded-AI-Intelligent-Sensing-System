import binascii

# 1. 读取TFLite模型文件（替换为你的tflite模型路径）
# 将TFLite模型（浮点型/int8量化型）转换为C数组.h文件（适配STM32F103）
import os


def tflite_to_c_array(tflite_path, c_header_path):
    # 1. 前置检查：判断TFLite文件是否存在
    if not os.path.exists(tflite_path):
        print(f" 错误：TFLite文件不存在，请检查路径：{tflite_path}")
        return

    # 2. 读取TFLite模型二进制数据
    try:
        with open(tflite_path, "rb") as f:
            tflite_data = f.read()
    except Exception as e:
        print(f" 错误：读取TFLite文件失败：{e}")
        return

    # 3. 检查模型数据是否为空
    if len(tflite_data) == 0:
        print(f" 错误：TFLite模型数据为空，文件可能损坏")
        return

    # 4. 转换为十六进制数组
    c_array = []
    for byte in tflite_data:
        c_array.append(f"0x{byte:02x}")
    model_length = len(c_array)

    # 5. 写入C语言头文件
    try:
        with open(c_header_path, "w", encoding="utf-8") as f:
            # 写入文件注释和头文件保护
            f.write(f"/* 自动生成的LSTM模型C数组（适配STM32F103）*/\n")
            f.write(f"/* 模型来源：{os.path.basename(tflite_path)} */\n")
            f.write(f"/* 模型大小：{model_length} 字节（存储在Flash只读数据段）*/\n\n")
            f.write(f"#ifndef LSTM_MODEL_H\n")
            f.write(f"#define LSTM_MODEL_H\n\n")
            f.write(f"#include <stdint.h>\n\n")

            # 写入模型长度宏
            f.write(f"// 模型数组长度宏（直接调用，无需手动计算）\n")
            f.write(f"#define LSTM_MODEL_LENGTH {model_length}\n\n")

            # 写入模型数组（指定存储在.rodata段，节省RAM）
            f.write(f"// LSTM模型二进制数据（uint8_t 兼容浮点型/int8量化型模型）\n")
            f.write(f"const uint8_t lstm_model[] __attribute__((section(\".rodata\"))) = {{\n")

            # 每16个字节换行，去除最后一行的多余逗号
            for i in range(0, model_length, 16):
                # 截取当前行的16个元素
                line_elements = c_array[i:i + 16]
                line = ", ".join(line_elements)
                # 判断是否为最后一行，最后一行不加逗号
                if i + 16 >= model_length:
                    f.write(f"    {line}\n")
                else:
                    f.write(f"    {line},\n")

            f.write(f"}};\n\n")
            f.write(f"#endif // LSTM_MODEL_H\n")

        print(f" 成功！C数组头文件已保存至：{c_header_path}")
        print(f" 模型长度：{model_length} 字节（约{model_length / 1024:.2f} KB）")
    except Exception as e:
        print(f" 错误：写入C头文件失败：{e}")
        return


# 执行转换（替换为你的TFLite文件路径和输出.h文件路径）
tflite_to_c_array(
    tflite_path="ultimate_lstm_int8.tflite",  # 你的TFLite模型路径（浮点型/int8量化型均可）
    c_header_path="lstm_model_int8.h"  # 输出的C头文件路径
)
