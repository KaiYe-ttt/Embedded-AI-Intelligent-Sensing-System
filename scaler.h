#pragma once
// scaler.h 或 lstm_infer.h
#ifndef SCALER_H
#define SCALER_H

#include "stdint.h"

// 从Python提取的MinMaxScaler核心参数
// 特征最小值（x, y, z）：对应scaler.data_min_
extern const float scaler_min[3];
// 特征缩放系数（x, y, z）：对应scaler.scale_（1/(X_max - X_min)）
extern const float scaler_scale[3];

// 函数声明：和训练一致的MinMaxScaler归一化
float minmax_scaler_normalize(int16_t raw_val, uint8_t feature_idx);

#endif