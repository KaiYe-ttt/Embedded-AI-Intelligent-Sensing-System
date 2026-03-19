// scaler.c 或 lstm_infer.c
#include "scaler.h"

// 初始化Scaler参数
const float scaler_min[3] = { -24.0, -17.0, 1.0 }; // 你的实际数值
const float scaler_scale[3] = { 0.04, 0.05263157894736842, 0.024390243902439025 }; // 你的实际数值

/**
 * 和训练一致的MinMaxScaler归一化（feature_range=(0,1)）
 * @param raw_val      单个特征的原始输入值（如ax/ay/az）
 * @param feature_idx  特征索引（0：x轴，1：y轴，2：z轴）
 * @return             归一化后的浮点值（范围：0~1）
 */
float minmax_scaler_normalize(int16_t raw_val, uint8_t feature_idx)
{
    // 1. 边界检查：feature_idx不能超出3个特征（0/1/2）
    if (feature_idx >= 3)
    {
        printf("错误：feature_idx超出范围（必须0~2）\n");
        return 0.0f;
    }

    // 2. 纯浮点运算（和Python端公式完全一致：(x - min) * scale）
    // 注意：所有变量都转为float，避免整数运算溢出
    float x_float = (float)raw_val;
    float min_float = scaler_min[feature_idx];
    float scale_float = scaler_scale[feature_idx];

    float normalized_val = (x_float - min_float) * scale_float;

    // 3. 裁剪到[0, 1]（和Python端一致，防止超出范围）
    if (normalized_val > 1.0f)
    {
        normalized_val = 1.0f;
    }
    if (normalized_val < 0.0f)
    {
        normalized_val = 0.0f;
    }

    // 4. 调试：打印中间结果（确认无异常）
    printf("调试：raw_val=%d, feature_idx=%d, (x-min)=%.6f, scale=%.6f, 归一化结果=%.6f\n",
        raw_val, feature_idx, (x_float - min_float), scale_float, normalized_val);

    return normalized_val;
}