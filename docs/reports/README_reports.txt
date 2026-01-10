SASRec.pytorch 实验报告文件说明
================================

本目录包含项目的四个实验报告Jupyter Notebook文件，用于对整个项目
进行正式的说明和解释。


文件列表
--------

1. 01_数据与实验设计报告.ipynb
   - 项目背景与问题定义
   - ml-1m数据集介绍
   - 数据预处理流程
   - 实验设置与超参数

2. 02_模型架构与实现报告.ipynb
   - SASRec模型架构
   - TiSASRec时序感知机制
   - mHC流形约束超连接
   - 核心代码实现说明

3. 03_训练与评估报告.ipynb
   - 训练配置与流程
   - 评估指标（HR@K, NDCG@K）
   - 实验结果与分析
   - 消融实验

4. 04_总结与展望报告.ipynb
   - 项目总结
   - 创新点
   - 局限性分析
   - 未来工作方向

数据文件
--------

data/sample_data.txt
   - 从ml-1m数据集截取的1000条示例数据
   - 用于Notebook中的加载演示
   - 格式：用户ID 物品ID 时间戳

DATA_CITATION.txt
   - ml-1m数据集的原始引用说明


使用方式
--------

1. 安装依赖：
   pip install jupyter numpy pandas matplotlib torch

2. 启动Jupyter：
   jupyter notebook

3. 打开对应的.ipynb文件，依次运行代码单元格

4. 可导出为HTML或PDF作为正式报告


注意事项
--------

- 示例数据仅用于演示代码功能，不用于实际训练
- 完整实验结果请参考python/目录下各实验的日志文件
- 所有图表和表格可在Notebook中交互式查看


数据引用
--------

本项目使用MovieLens 1M数据集。

原始数据集来源：https://grouplens.org/datasets/movielens/

引用：F. Maxwell Harper and Joseph A. Konstan. 2015.
      The MovieLens Datasets: History and Context.
      ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
