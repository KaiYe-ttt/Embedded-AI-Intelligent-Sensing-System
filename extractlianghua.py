import tensorflow as tf
import numpy as np

# 加载你的TFLite量化模型
tflite_model_path = "ultimate_lstm_int8.tflite"  # 你的TFLite模型路径
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 获取输入和输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 提取输入的scale和zero_point（对应C端量化公式需要）
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]

# 提取输出的scale和zero_point（对应C端反量化公式需要）
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

# 打印结果（复制到C端使用，这是真实校准后的参数，不是固定值）
print("=== TFLite校准后参数（复制到C端）===")
print(f"输入：input_scale = {input_scale}, input_zero_point = {input_zero_point}")
print(f"输出：output_scale = {output_scale}, output_zero_point = {output_zero_point}")
print(f"输入张量形状：{input_details[0]['shape']}")
print(f"输出张量形状：{output_details[0]['shape']}")