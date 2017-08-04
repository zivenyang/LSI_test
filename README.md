## 目录：
* [实验环境](#实验环境)
* [必要库](#必要库)
* [主要文件](#主要文件)
* [次要文件](#次要文件)
* [使用方法](#使用方法)
* [注意事项](#注意事项)


## 实验环境：
* Python 2.x 或 Python 3.x

## 必要库：
* matplotlib
* numpy
* math
* re

## 主要文件：
* dataset.txt	数据集：由书名编号和书名组成，例如：B1 A Course on Integral Equations
* stop_words.txt	停用词：用于去除书名中与语义无关的停用词，例如："a", "on", "of"...
* new_data.txt	新数据：由书名编号和书名组成，用于测试LSI的更新
* README.txt	说明文档
* lsi_test.py	LSI实现代码

## 次要文件：
* A.txt		用于存放 关键词-书名 矩阵A，每一行为一个关键词在各个书名中出现的词频数
* A_k.txt		用于存放A的 k-秩近似矩阵 ，每一行为一个词向量
* new_A.txt	用于存放更新书名后的 关键词-书名 矩阵A
* new_A_k.txt	用于存放更新书名后A的 k-秩近似矩阵
* svd_test.py 尝试自己实现SVD分解，还有BUG

## 使用方法：
* 命令行下：
  1. 切换到LSI_test文件夹下
  2. 执行 python ./lsi_test.py
* 或者使用Python IDE运行

## 修改参数：
* 在lsi_test.py文件最底部找到
```python
if __name__ == '__main__':
    lsi(book_names_file="./dataset.txt",     # ==>数据集文件路径
        stop_words_file="./stop_words.txt",  # ==>停用词文件路径
        save_A="./A.txt",                    # ==>关键词-书名 矩阵A保存路径
        save_A_k="./A_k.txt",                # ==>A的 k-秩近似矩阵保存路径
        save_new_A="./new_A.txt",            # ==>更新书名后的 关键词-书名 矩阵A保存路径
        save_new_A_k="./new_A_k.txt",        # ==>更新书名后A的 k-秩近似矩阵保存路径
        theta=0.4,                           # ==>k-秩近似矩阵阈值，影响k的取值，由于可视化时只能现实2维数据，因此k=2时可视化结果最为准确，建议无更新时设置theta=0.4，有更新时设置theta=0.3
        cos_treshold=0.9,                    # ==>夹角余弦距离阈值，值越大相似度越高
        # query='Application and theory',
        query=None,                          # ==>查询测试，查询与输入相关的书名，相关书名数量由cos_treshold控制，可设置None关闭
        # update_file="./new_data.txt")
        update_file=None)                    # ==>新书名路径，用于更新测试，可输入None关闭
```
## 注意事项：

* 【注】由与只能可视化2维数据，当theta取值过大导致k>2时可能会出现可视化结果与计算的夹角余弦结果不符，而当k<2时无法二维可视化，因此最好使得k=2
