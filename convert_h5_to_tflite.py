import tensorflow as tf
from tensorflow import keras
import numpy as np

# ===================== 配置项（按需修改路径） =====================
H5_MODEL_PATH = "ultimate_high_precision_lstm.h5"  # 训练好的H5模型路径
INT8_TFLITE_PATH = "ultimate_lstm_int8.tflite"  # 输出的int8量化TFLite模型路径
CALIBRATION_SAMPLES = 100  # 校准数据集样本数（100足够保证量化精度）

# ===================== 步骤1：加载H5模型 =====================
try:
    model = keras.models.load_model(H5_MODEL_PATH)
    print(f"✅ 成功加载H5模型：{H5_MODEL_PATH}")
    print(f"✅ 模型输入形状：{model.input_shape}（需匹配校准数据集形状）")
except Exception as e:
    print(f"❌ 加载H5模型失败：{e}")
    exit()

# ===================== 步骤2：配置TFLite转换器（含量化+LSTM兼容修正） =====================
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 2.1 核心：开启优化（包含8位整数量化）
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# 2.2 核心：定义代表性数据集（用于校准，匹配模型输入形状(1, 12, 3)）
# 作用：让转换器学习数据分布，保证量化后精度不丢失
def representative_dataset():
    # 加载之前保存的训练时序数据
    X_train = np.load("X_train_seq_ultimate.npy").astype(np.float32)
    # 随机选取100个样本（避免顺序影响，保证校准的全面性）
    sample_indices = np.random.choice(len(X_train), 100, replace=False)
    for idx in sample_indices:
        # 增加批次维度（模型输入需要(1, 12, 3)，X_train中每个样本是(12, 3)）
        calibration_input = np.expand_dims(X_train[idx], axis=0)
        yield [calibration_input]

converter.representative_dataset = representative_dataset

# 2.3 核心：指定支持INT8算子+Select TF Ops（兼容LSTM，解决之前的转换报错）
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # 支持INT8轻量化算子（适配STM32）
    tf.lite.OpsSet.SELECT_TF_OPS  # 支持LSTM扩展算子，避免转换报错
]

# 2.4 核心：强制输入输出类型为int8（量化到位，减小模型体积）
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 2.5 核心：禁用张量列表操作降级（解决LSTM的TensorListReserve报错）
converter._experimental_lower_tensor_list_ops = False

# ===================== 步骤3：执行量化转换 =====================
try:
    int8_tflite_model = converter.convert()
    print("✅ 成功完成8位整数量化转换")
except Exception as e:
    print(f"❌ 8位整数量化转换失败：{e}")
    exit()

# ===================== 步骤4：保存量化后的TFLite模型 =====================
try:
    with open(INT8_TFLITE_PATH, "wb") as f:
        f.write(int8_tflite_model)

    model_size = len(int8_tflite_model) / 1024
    print(f"✅ 8位量化TFLite模型已生成：{INT8_TFLITE_PATH}")
    print(f"✅ 量化后模型大小：{model_size:.2f} KB（STM32F103可轻松容纳）")
    print(f"✅ 下一步：使用该模型转换为C数组（可直接复用你的tflite_to_c_array函数）")
except Exception as e:
    print(f"❌ 保存量化TFLite模型失败：{e}")
    exit()