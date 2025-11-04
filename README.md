# multi_threading
本仓库是多线程编程作业。主程序在main.py
## 主要功能
加载预处理的 SlimPajama 数据集（两个分块）。
使用预训练的 GPTNeoX（Pythia-160m）模型抽取文本 embedding。
利用多线程 KMeans 对 embedding 进行聚类。
可视化每个聚类的样本分布（柱状图）。
保存每个聚类的若干样本到 cluster_examples.json 以便后续分析。
main.py的17，18行定义了数据集与模型的路径。请按自己的环境修改。模型以及数据集会在main.py运行的时候自动下载（可能需要1h）
## 运行方式
./requirements.txt是所有依赖包，请确保它们被安装，然后直接运行main.py

