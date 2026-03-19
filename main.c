#include "lstm_infer.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>

// 全局变量声明
int8_t lstm_final_h[LSTM_UNITS] = { 0 }; // LSTM层输出
float infer_prob = 0.0f;
int infer_result = 0;

// 定义循环缓冲区（存储15个时间步，每个时间步3个特征）
int8_t input_window_buf[WINDOW_BUF_SIZE][3] = { 0 };
uint8_t window_buf_idx = 0; // 缓冲区当前索引
// 新增：窗口是否填满标识（辅助判断，避免未填满时的无效数据）
bool window_is_full = false;


void update_input_window(int8_t* new_input, int8_t ordered_window[WINDOW_BUF_SIZE][3])
{
    // 边界判断：空指针直接返回
    if (new_input == NULL || ordered_window == NULL) return;

    // ========== 保留你原有的核心逻辑：循环写入缓冲区 ==========
    // 1. 将新数据写入缓冲区当前索引
    input_window_buf[window_buf_idx][0] = new_input[0];
    input_window_buf[window_buf_idx][1] = new_input[1];
    input_window_buf[window_buf_idx][2] = new_input[2];

    // 2. 更新索引（循环覆盖，保持最近15组数据）
    window_buf_idx = (window_buf_idx + 1) % WINDOW_BUF_SIZE;

    // 3. 新增：标记窗口是否已填满（写入满15次后，即为填满状态）
    static int32_t write_count = 0;
    write_count++;
    if (write_count >= WINDOW_BUF_SIZE)
    {
        window_is_full = true;
    }

    // ========== 整理有序时间序列（最旧→最新） ==========
    // 思路：根据当前写入索引，从缓冲区中提取最近15组数据，按时间顺序排列
    int32_t current_read_idx = 0;
    for (int32_t i = 0; i < WINDOW_BUF_SIZE; i++)
    {
        // 计算有序序列中第i个位置对应的缓冲区索引
        // 窗口填满后：从当前写入索引开始（最旧数据），向后取15组
        // 窗口未填满：从索引0开始，向后取已写入的组（其余补0）
        if (window_is_full)
        {
            current_read_idx = (window_buf_idx + i) % WINDOW_BUF_SIZE;
        }
        else
        {
            current_read_idx = i;
        }

        // 将缓冲区数据复制到有序序列中
        ordered_window[i][0] = input_window_buf[current_read_idx][0];
        ordered_window[i][1] = input_window_buf[current_read_idx][1];
        ordered_window[i][2] = input_window_buf[current_read_idx][2];
    }

    // ==========调试打印（验证有序序列是否正确） ==========
    printf("=== 整理后的有序窗口（前3步+后3步，最旧→最新）===\n");
    for (int32_t i = 0; i < 3; i++)
    {
        printf("第%d步：%d, %d, %d\n", i, ordered_window[i][0], ordered_window[i][1], ordered_window[i][2]);
    }
    printf("...\n");
    for (int32_t i = WINDOW_BUF_SIZE - 3; i < WINDOW_BUF_SIZE; i++)
    {
        printf("第%d步：%d, %d, %d\n", i, ordered_window[i][0], ordered_window[i][1], ordered_window[i][2]);
    }
    printf("\n");
}


int main(void)
{
    // 1. 系统初始化（保留你的原有初始化逻辑，此处注释仅为格式完整）
    // HAL_Init();
    // SystemClock_Config();
    // MX_I2C1_Init();
    // MX_USART1_UART_Init();

    // 2. 定义有序窗口数组（关键：LSTM需要的有序输入序列，全局或局部均可，此处局部更安全）
    int8_t ordered_lstm_window[WINDOW_BUF_SIZE][3] = { 0 };

    // 3. 实时推理闭环（无限循环，实现持续实时推理）
    while (1)
    {
        // 3.1 变量定义
        int ax, ay, az;
        int8_t realtime_input[3] = { 0 };

        // 3.2 实时获取数据（保留你的scanf_s逻辑，适配Windows环境）
        //Get_Accel(&ax, &ay, &az);
        scanf_s("%d", &ax);
        scanf_s("%d", &ay);
        scanf_s("%d", &az);

        // 3.3 数据预处理（归一化+量化，保留你的原有逻辑）
        // x轴（特征索引0）
        float norm_ax = minmax_scaler_normalize(ax, 0);
        realtime_input[0] = lstm_quantize_input(norm_ax);

        // y轴（特征索引1）
        float norm_ay = minmax_scaler_normalize(ay, 1);
        realtime_input[1] = lstm_quantize_input(norm_ay);

        // z轴（特征索引2）
        float norm_az = minmax_scaler_normalize(az, 2);
        realtime_input[2] = lstm_quantize_input(norm_az);

        // 3.4 更新时间窗口缓冲区（修正：补全ordered_lstm_window参数，获取有序序列）
        // 修改点1：传入两个参数，第二个参数是输出的有序窗口，供LSTM使用
        update_input_window(realtime_input, ordered_lstm_window);

        // 3.5 生成LSTM层输出（LSTM序列推理）
        // 修改点2：传入整理后的有序窗口ordered_lstm_window，而非原始无序缓冲区input_window_buf
        lstm_sequence_infer(ordered_lstm_window, lstm_final_h); // 输入有序窗口→LSTM推理

        // 3.6 执行Dense层推理（传入真实LSTM输出，保留你的原有逻辑）
        infer_result = lstm_complete_infer(lstm_final_h, &infer_prob);

        // 3.7 串口输出实时推理结果（保留你的原有打印逻辑，格式优化）
        printf("=== Real-Time Infer Result ===\r\n");
        printf("Accel: ax=%d, ay=%d, az=%d\r\n", ax, ay, az);
        printf("Infer Result: %d, Prob: %.4f\r\n\r\n", infer_result, infer_prob);

        // 3.8 延时控制（调节推理频率，此处1000ms/次，保留你的原有逻辑）
        //HAL_Delay(1000);
    }
}
