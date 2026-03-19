该项目由两部分组成
1、python端轻量化lstm模型训练
2、c端lstm模型部署

python端训练主要分为三个步骤

1.基础训练阶段，具体内容在lstm.py文件
2.模型参数提取（方便c端部署),具体内容在extrac系列文件和generate系列文件
3.模型转换，将h5模型转为c模型，两个convert文件

c端lstm模型部署

1.模型参数在lstm_model
2.模型推理函数在lstm_infer
