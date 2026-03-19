import tensorflow as tf
import numpy as np

# -------------------------- 1. 配置参数（与之前一致，无需修改） --------------------------
QUANT_MODEL_PATH = "ultimate_lstm_int8.tflite"
HIDDEN_UNIT_NUM = 64  # 你的LSTM隐藏单元数
INPUT_FEATURE_NUM = 3  # 输入特征数
OUTPUT_NODE_NUM = 1  # 输出节点数

# -------------------------- 2. 初始化Interpreter --------------------------
interpreter = tf.lite.Interpreter(model_path=QUANT_MODEL_PATH)
interpreter.allocate_tensors()

# -------------------------- 3. 提取输入/输出量化参数 --------------------------
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print(" 量化参数提取完成：")
print(f"输入量化：scale={input_scale:.6f}, zero_point={input_zero_point}")
print(f"输出量化：scale={output_scale:.6f}, zero_point={output_zero_point}")

# -------------------------- 4. 第一步：打印所有张量的完整信息（核心：找到真实LSTM权重） --------------------------
tensor_details = interpreter.get_tensor_details()

print("\n" + "=" * 80)
print(" 所有张量完整信息（形状+名称），请找到包含LSTM权重的张量：")
print("=" * 80)
for idx, tensor in enumerate(tensor_details):
    tensor_shape = tensor["shape"]
    tensor_name = tensor["name"]
    print(f"索引：{idx} | 形状：{tensor_shape} | 名称：{tensor_name}")

print("\n" + "=" * 80)
print(" 重点关注：形状包含 4*80=320、3（输入特征）、80（隐藏单元）的张量！")
print("=" * 80)

# -------------------------- 5. 第二步：修改权重匹配逻辑（兼容多种形状，友好提示） --------------------------
# 初始化要提取的权重变量
LSTM_Wxh = None
LSTM_Whh = None
LSTM_bh = None
LSTM_Why = None
LSTM_by = None

# 定义LSTM权重的核心特征（包含 4*hidden 和 input+hidden），不强制固定形状
lstm_core_size_1 = 4 * HIDDEN_UNIT_NUM
lstm_core_size_2 = INPUT_FEATURE_NUM + HIDDEN_UNIT_NUM
dense_core_size_1 = HIDDEN_UNIT_NUM
dense_core_size_2 = OUTPUT_NODE_NUM

print("\n 开始根据核心特征匹配权重...")
for tensor in tensor_details:
    tensor_shape = tensor["shape"]
    tensor_idx = tensor["index"]
    tensor_name = tensor["name"]
    shape_tuple = tuple(tensor_shape)
    shape_list = list(tensor_shape)

    # ------------- 匹配LSTM权重（核心特征：包含320和83（3+80）） -------------
    if (lstm_core_size_1 in shape_list) and (lstm_core_size_2 in shape_list):
        print(f"\n 匹配到LSTM权重张量：{tensor_name}，形状：{tensor_shape}")
        # 提取LSTM权重数据并转换为INT8
        lstm_weights = interpreter.get_tensor(tensor_idx).astype(np.int8)

        # 自动调整形状，适配两种存储格式
        if shape_tuple == (lstm_core_size_1, lstm_core_size_2):
            lstm_weights_reshaped = lstm_weights.reshape((lstm_core_size_1, lstm_core_size_2))
        elif shape_tuple == (lstm_core_size_2, lstm_core_size_1):
            lstm_weights_reshaped = lstm_weights.reshape((lstm_core_size_2, lstm_core_size_1)).T
        else:
            lstm_weights_reshaped = lstm_weights.reshape((lstm_core_size_1, lstm_core_size_2))

        # 拆分Wxh和Whh
        LSTM_Wxh = lstm_weights_reshaped[:, :INPUT_FEATURE_NUM]
        LSTM_Whh = lstm_weights_reshaped[:, INPUT_FEATURE_NUM:]
        print(f"  LSTM Wxh形状：{LSTM_Wxh.shape}, Whh形状：{LSTM_Whh.shape}")

    # ------------- 匹配LSTM偏置（核心特征：包含320） -------------
    elif len(shape_tuple) == 1 and shape_tuple[0] == lstm_core_size_1:
        if "bias" in tensor_name.lower() or "bh" in tensor_name.lower() or LSTM_bh is None:
            print(f"\n 匹配到LSTM偏置张量：{tensor_name}，形状：{tensor_shape}")
            LSTM_bh = interpreter.get_tensor(tensor_idx).astype(np.int8)
            print(f"  LSTM bh形状：{LSTM_bh.shape}")

    # ------------- 匹配全连接层权重（核心特征：包含80和1） -------------
    elif (dense_core_size_1 in shape_list) and (dense_core_size_2 in shape_list):
        if "dense" in tensor_name.lower() and "output" in tensor_name.lower():
            print(f"\n 匹配到全连接权重张量：{tensor_name}，形状：{tensor_shape}")
            # 提取全连接权重并调整形状
            dense_weights = interpreter.get_tensor(tensor_idx).astype(np.int8)
            if shape_tuple == (dense_core_size_1, dense_core_size_2):
                LSTM_Why = dense_weights.T.reshape((dense_core_size_2, dense_core_size_1))
            else:
                LSTM_Why = dense_weights.reshape((dense_core_size_2, dense_core_size_1))
            print(f"  全连接Why形状：{LSTM_Why.shape}")

    # ------------- 匹配全连接层偏置（核心特征：包含1） -------------
    elif len(shape_tuple) == 1 and shape_tuple[0] == dense_core_size_2:
        if "dense" in tensor_name.lower() and "output" in tensor_name.lower() and LSTM_by is None:
            print(f"\n 匹配到全连接偏置张量：{tensor_name}，形状：{tensor_shape}")
            LSTM_by = interpreter.get_tensor(tensor_idx).astype(np.int8)
            print(f"  全连接by形状：{LSTM_by.shape}")

# -------------------------- 6. 验证提取结果（友好提示，不强制断言） --------------------------
print("\n" + "=" * 80)
print(" 权重提取结果验证：")
print("=" * 80)
if LSTM_Wxh is not None:
    print(f" LSTM Wxh：提取成功，形状：{LSTM_Wxh.shape}")
else:
    print(f" LSTM Wxh：提取失败，请查看上方张量列表，手动匹配形状")

if LSTM_Whh is not None:
    print(f" LSTM Whh：提取成功，形状：{LSTM_Whh.shape}")
else:
    print(f" LSTM Whh：提取失败，请查看上方张量列表，手动匹配形状")

if LSTM_bh is not None:
    print(f" LSTM bh：提取成功，形状：{LSTM_bh.shape}")
else:
    print(f" LSTM bh：提取失败，请查看上方张量列表，手动匹配形状")

if LSTM_Why is not None:
    print(f" 全连接Why：提取成功，形状：{LSTM_Why.shape}")
else:
    print(f" 全连接Why：提取失败，请查看上方张量列表，手动匹配形状")

if LSTM_by is not None:
    print(f" 全连接by：提取成功，形状：{LSTM_by.shape}")
else:
    print(f" 全连接by：提取失败，请查看上方张量列表，手动匹配形状")

print("\n" + "=" * 80)
print(" 若部分权重提取失败，请根据上方张量列表，手动修改形状匹配逻辑！")
print("=" * 80)
