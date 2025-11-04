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
## 相关结果：
结果大致有三个
### cluster_distributuin.png
每个cluster的频次分布。排序后大致成power law decay($f_i \sim \frac{1}{i^{\alpha}} $)。这与Zipf law相符和。
### cluster_example.json
该文件存了各个cluster（总共300个）中的语料，不过由于时间原因只抽取了16个例子，因此大部分cluster里面都没有语料。
### cpu占用率：
|线程数|CPU占用率|
|-----|---------|
|1|104%|
|2| 123%|
|4|140%|
|32|172%|
