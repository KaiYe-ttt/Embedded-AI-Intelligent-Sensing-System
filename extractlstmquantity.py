import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import LSTM

# ===================== 第一步：复用你的训练配置（无需修改，和训练代码保持一致） =====================
LSTM_UNITS = 64  # 你的LSTM隐藏单元数
INPUT_DIM = 3  # 输入特征维度（x/y/z）
MODEL_PATH = "ultimate_high_precision_lstm.h5"  # 你的模型保存路径
QUANT_SCALE = 0.003922  # 量化缩放系数（和Dense层一致）
QUANT_ZERO_POINT = -128  # 量化零点（和Dense层一致）


# ===================== 第二步：加载模型并【自动查找】LSTM层（避免索引错误） =====================
def load_lstm_layer(model_path):
    # 加载训练好的模型
    model = keras.models.load_model(model_path)
    print("=== 你的模型结构详情 ===")
    model.summary()

    # 自动查找LSTM层（无需手动指定索引，更健壮）
    lstm_layer = None
    for layer in model.layers:
        if isinstance(layer, LSTM):
            lstm_layer = layer
            break

    # 验证是否找到LSTM层
    if lstm_layer is None:
        raise ValueError("错误：模型中未找到LSTM层！请检查模型路径是否正确。")

    print(f"\n=== 自动提取的LSTM层信息 ===")
    print(f"层名称：{lstm_layer.name}")
    print(f"层索引：{model.layers.index(lstm_layer)}")
    print(f"隐藏单元数：{lstm_layer.units}")
    print(f"输入形状：{lstm_layer.input_shape}")
    print(f"是否返回序列：{lstm_layer.return_sequences}")
    print(f"遗忘门偏置默认值：{lstm_layer.unit_forget_bias}")

    return lstm_layer, model


# ===================== 第三步：拆分LSTM权重（解决ValueError，兼容所有情况） =====================
def split_lstm_weights(lstm_layer):
    units = lstm_layer.units
    weights = lstm_layer.get_weights()

    print(f"\n=== LSTM层权重原始信息 ===")
    print(f"权重列表长度：{len(weights)}")
    for i, w in enumerate(weights):
        print(f"  权重{i} 形状：{w.shape}")

    # 核心权重：kernel（输入→门）、recurrent_kernel（隐藏→门）
    kernel = weights[0]  # 形状：(3, 256) → 3输入 × 4门×64单元
    recurrent_kernel = weights[1]  # 形状：(64, 256) → 64隐藏 × 4门×64单元

    # 拆分4个门的权重（每个门对应units*4中的一个切片）
    slice_1 = units
    slice_2 = 2 * units
    slice_3 = 3 * units
    slice_4 = 4 * units

    # 拆分kernel（输入→门）
    kernel_f = kernel[:, 0:slice_1]
    kernel_i = kernel[:, slice_1:slice_2]
    kernel_c = kernel[:, slice_2:slice_3]
    kernel_o = kernel[:, slice_3:slice_4]

    # 拆分recurrent_kernel（隐藏→门）
    recurrent_f = recurrent_kernel[:, 0:slice_1]
    recurrent_i = recurrent_kernel[:, slice_1:slice_2]
    recurrent_c = recurrent_kernel[:, slice_2:slice_3]
    recurrent_o = recurrent_kernel[:, slice_3:slice_4]

    # 拼接为STM32所需的权重格式：(隐藏64+输入3, 64) → (67, 64)
    Wf = np.vstack([recurrent_f, kernel_f])
    Wi = np.vstack([recurrent_i, kernel_i])
    Wc = np.vstack([recurrent_c, kernel_c])
    Wo = np.vstack([recurrent_o, kernel_o])

    # 处理偏置（有则提取，无则返回全0，兼容你的模型配置）
    bf = np.zeros((units,))
    bi = np.zeros((units,))
    bc = np.zeros((units,))
    bo = np.zeros((units,))

    if len(weights) >= 3:
        bias = weights[2]  # 形状：(256,) → 4门×64单元
        bf = bias[0:slice_1]
        bi = bias[slice_1:slice_2]
        bc = bias[slice_2:slice_3]
        bo = bias[slice_3:slice_4]

    print(f"\n=== LSTM层权重拆分完成 ===")
    print(f"Wf/Wi/Wc/Wo 形状：{Wf.shape}")
    print(f"bf/bi/bc/bo 形状：{bf.shape}")

    return Wf, Wi, Wc, Wo, bf, bi, bc, bo


# ===================== 第四步：量化权重（转为int8_t，适配STM32） =====================
def quantize_weight(weight):
    """量化公式：和你的Dense层保持一致，裁剪到-128~127"""
    quant_val = (weight / QUANT_SCALE + 0.5).astype(np.int32) + QUANT_ZERO_POINT
    quant_val = np.clip(quant_val, -128, 127)
    return quant_val.astype(np.int8)


# ===================== 第五步：格式化输出（直接复制到STM32的lstm_quant_weights.h） =====================
def print_stm32_lstm_weights(Wf_quant, Wi_quant, Wc_quant, Wo_quant, bf_quant, bi_quant, bc_quant, bo_quant):
    print(f"\n" + "=" * 80)
    print(f"=== 以下内容直接复制到 STM32 的 lstm_quant_weights.h 中 ===")
    print(f"=" * 80)

    # 1. 遗忘门 Wf (67, 64)
    print(f"\n// 遗忘门权重 Wf (67, 64)：输入3维+隐藏64维→64维")
    print(f"const int8_t lstm_Wf[67][64] = {{")
    for row in Wf_quant:
        row_str = ", ".join(map(str, row))
        print(f"  {{ {row_str} }},")
    print(f"}};")

    # 2. 遗忘门偏置 bf (64)
    print(f"\n// 遗忘门偏置 bf (64)")
    bf_str = ", ".join(map(str, bf_quant))
    print(f"const int8_t lstm_bf[64] = {{ {bf_str} }};")

    # 3. 输入门 Wi (67, 64)
    print(f"\n// 输入门权重 Wi (67, 64)")
    print(f"const int8_t lstm_Wi[67][64] = {{")
    for row in Wi_quant:
        row_str = ", ".join(map(str, row))
        print(f"  {{ {row_str} }},")
    print(f"}};")

    # 4. 输入门偏置 bi (64)
    print(f"\n// 输入门偏置 bi (64)")
    bi_str = ", ".join(map(str, bi_quant))
    print(f"const int8_t lstm_bi[64] = {{ {bi_str} }};")

    # 5. 细胞门 Wc (67, 64)
    print(f"\n// 细胞门权重 Wc (67, 64)")
    print(f"const int8_t lstm_Wc[67][64] = {{")
    for row in Wc_quant:
        row_str = ", ".join(map(str, row))
        print(f"  {{ {row_str} }},")
    print(f"}};")

    # 6. 细胞门偏置 bc (64)
    print(f"\n// 细胞门偏置 bc (64)")
    bc_str = ", ".join(map(str, bc_quant))
    print(f"const int8_t lstm_bc[64] = {{ {bc_str} }};")

    # 7. 输出门 Wo (67, 64)
    print(f"\n// 输出门权重 Wo (67, 64)")
    print(f"const int8_t lstm_Wo[67][64] = {{")
    for row in Wo_quant:
        row_str = ", ".join(map(str, row))
        print(f"  {{ {row_str} }},")
    print(f"}};")

    # 8. 输出门偏置 bo (64)
    print(f"\n// 输出门偏置 bo (64)")
    bo_str = ", ".join(map(str, bo_quant))
    print(f"const int8_t lstm_bo[64] = {{ {bo_str} }};")


# ===================== 主流程：执行所有步骤 =====================
if __name__ == "__main__":
    try:
        # 1. 加载LSTM层（自动查找，无需手动指定索引）
        lstm_layer, _ = load_lstm_layer(MODEL_PATH)

        # 2. 拆分LSTM权重
        Wf, Wi, Wc, Wo, bf, bi, bc, bo = split_lstm_weights(lstm_layer)

        # 3. 量化所有权重
        Wf_quant = quantize_weight(Wf)
        Wi_quant = quantize_weight(Wi)
        Wc_quant = quantize_weight(Wc)
        Wo_quant = quantize_weight(Wo)
        bf_quant = quantize_weight(bf)
        bi_quant = quantize_weight(bi)
        bc_quant = quantize_weight(bc)
        bo_quant = quantize_weight(bo)

        # 4. 输出STM32可用的权重代码
        print_stm32_lstm_weights(Wf_quant, Wi_quant, Wc_quant, Wo_quant, bf_quant, bi_quant, bc_quant, bo_quant)

        print(f"\n" + "=" * 80)
        print(f"✅ LSTM权重提取+量化完成！直接复制上述内容到STM32工程即可。")
        print(f"=" * 80)
    except Exception as e:
        print(f"\n❌ 执行失败：{e}")