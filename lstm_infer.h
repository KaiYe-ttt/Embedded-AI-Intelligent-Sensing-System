#ifndef LSTM_INFER_H
#define LSTM_INFER_H
#include <inttypes.h>
#include "stdint.h"
#include "math.h" 
#include "scaler.h"
#include <stdio.h>
#define LSTM_INPUT_DIM     3
#define LSTM_CONCAT_DIM    67
#define WINDOW_BUF_SIZE    15
#define DENSE_INPUT_DIM 64  // Dense层输入维度（等于LSTM隐藏单元数）
#define DENSE_OUTPUT_DIM 2          // 推理结果维度（示例：2分类）

#ifndef DENSE16_UNITS
#define DENSE16_UNITS 16  // 对应你的Dense16层单元数，确保和实际一致
#endif

// 核心：裁剪宏（适配STM32，无额外依赖，用于限制int8输出范围，避免饱和）
// 将值限制在[MIN_VAL, MAX_VAL]之间，避免溢出到int8极值126/-127
#define CLAMP_INT8(val, min_val, max_val) \
    ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))

// 推荐裁剪范围（可根据你的量化参数微调，避免饱和且保留区分度）
#define DENSE16_CLAMP_MIN -64
#define DENSE16_CLAMP_MAX 63
#define DENSE1_CLAMP_MIN -64
#define DENSE1_CLAMP_MAX 63

#ifndef __CLAMP
#define __CLAMP(val, min_val, max_val) \
    ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))
#endif

// 导入生成的权重头文件（根据你的工程目录调整路径）
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
#define INPUT_SCALE           0.003921568859368563
#define INPUT_ZERO_POINT      -128
#define OUTPUT_SCALE          0.00390625
#define OUTPUT_ZERO_POINT     -128

// ===================== 第二步：权重变量仅做声明（加extern，去掉初始化） =====================
// LSTM层权重 - 仅声明（告诉编译器变量存在，不分配内存）
extern const int8_t lstm_Wf[67][64];
extern const int8_t lstm_bf[64];
extern const int8_t lstm_Wi[67][64];
extern const int8_t lstm_bi[64];
extern const int8_t lstm_Wc[67][64];
extern const int8_t lstm_bc[64];
extern const int8_t lstm_Wo[67][64];
extern const int8_t lstm_bo[64];

// Dense层权重 - 仅声明
extern const int8_t dense16_weights[64][16];
extern const int8_t dense16_bias[16];
extern const int8_t dense1_weights[16][1];
extern const int8_t dense1_bias[1];

// MinMaxScaler参数 - 仅声明
extern const float scaler_min[3];
extern const float scaler_scale[3];

// ===================== 其他函数声明（保持不变） =====================
void lstm_single_step_infer(int8_t* x_t, int8_t* h_prev, int32_t* c_prev,
    int8_t* h_curr, int32_t* c_curr);
void lstm_sequence_infer(int8_t input_window[WINDOW_BUF_SIZE][LSTM_INPUT_DIM], int8_t* h_final);
int lstm_complete_infer(int8_t* lstm_output, float* class_prob);
int8_t lstm_quantize_input(float raw_input);
float lstm_dequantize_output(int8_t quant_output);
int8_t lstm_tanh_activate(int8_t x);
int8_t lstm_sigmoid_activate(int8_t x);
void lstm_dense16_infer(int8_t* input, int8_t* output);
void lstm_dense1_infer(int8_t* input, int8_t* output);
void lstm_infer_test(void);

// ===================== 头文件卫士结尾 =====================
#endif // LSTM_INFER_H