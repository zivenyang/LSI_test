# coding=utf-8
from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as plt
import math
import numpy as np
import re


# ==================== 预处理 ====================
def load_txt(file_name):
    """ 载入txt文件 """
    if not file_name.split(".")[-1] == "txt":
        print(u"%s不是txt文件" % file_name)
    else:
        f = open(file_name, 'r')
        lines = f.readlines()
        f.close()
        return lines


def save_txt(data, save_file, fmt='%.18e'):
    np.savetxt(fname=save_file, X=data, fmt=fmt)
    print(u'%s保存完毕' % save_file)


def plural(word):
    """ 将单词的单数转换为复数 """
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


def generate_book_names(lines):
    """ 生成书名列表 """
    book_names_list = []
    for line in lines:  # 忽略书名中的符号和大小写
        new_line = re.sub("[^a-zA-Z0-9\s]", "", line)
        words = new_line.lower().split()[1:]
        book_names_list.append(words)
    for i in range(len(book_names_list)):   # 忽略书名中的单复数（有单复数同时出现的单词以复数形式表示）
        for j in range(len(book_names_list[i])):
            word = book_names_list[i][j]
            plural_w = plural(word)
            for k in range(len(book_names_list)):
                if plural_w in book_names_list[k]:
                    book_names_list[i][j] = plural_w
    return book_names_list


def generate_books_list(lines):
    """ 生成书目编号列表 """
    books_list = []
    for line in lines:
        books_list.append(line.split()[0])
    return books_list


def generate_stop_words(lines):
    """ 生成停用词列表 """
    stop_words_list = []
    for line in lines:
        stop_words_list.append(line.split("\n")[0])
    return stop_words_list


def generate_keywords(book_names_lines, stop_words_lines):
    """ 生成关键字列表 """
    book_names_list = generate_book_names(book_names_lines)
    stop_words = generate_stop_words(stop_words_lines)
    words_list = []
    key_words_list = []
    for book in book_names_list:    # 去除停用词，如"and", "of"...
        for word in book:
            if word in stop_words:
                pass
            else:
                words_list.append(word)
    uni_words_list = np.unique(ar=words_list)

    for _word in uni_words_list:    # 将词频 > 1的词作为关键字
        if words_list.count(_word) > 1:
            key_words_list.append(_word)
        else:
            pass
    print(key_words_list)
    return key_words_list


def term_doc_matrix(book_names_list, key_words_list):
    """ 生成 关键字-书名 矩阵 """
    m = len(key_words_list)
    n = len(book_names_list)
    t_d_matrix = np.zeros(shape=[m, n])
    for i in range(n):
        for j in range(m):
            t_d_matrix[j, i] += book_names_list[i].count(key_words_list[j])
    print("A=\n", t_d_matrix, t_d_matrix.shape)
    return t_d_matrix


# ==================== SVD ====================
def svd(matrix):
    [U, S, V] = np.linalg.svd(matrix)
    # print("\nU=\n", U, U.shape)
    # print("\nS=\n", S, S.shape)
    # print("\nV=\n", V, V.shape)
    return U, S, V


def cut_U(U, k):
    U_k = U[:, :k]
    # print("\nU_k=\n", U_k, U_k.shape)
    return U_k


def cut_S(S, theta):
    S_k = []
    k = 0
    threshold = theta * sum(S)
    for value in S:
        if sum(S_k) + value < threshold:
            S_k.append(value)
            k += 1
        else:
            break
    S_k = np.diag(S_k)
    # print("\nS_k=\n", S_k, S_k.shape)
    return S_k, k


def cut_V(V, k):
    V_k = V[:k, :]
    # print("\nV_k=\n", V_k, V_k.shape)
    return V_k


def k_rank_matrix(t_d_matrix, theta):
    U, S, V = svd(t_d_matrix)
    S_k, k = cut_S(S, theta)
    U_k = cut_U(U, k)
    V_k = cut_V(V, k)
    A_k = np.matmul(np.matmul(U_k, S_k), V_k)
    print("\nA_k=\n", A_k, A_k.shape)
    return A_k, U_k, S_k, V_k


# ==================== 查询 ====================
def query_vector(query, key_words_list):
    """ 将查询字符转换为向量 """
    vector = np.zeros(shape=[len(key_words_list), 1])
    line = re.sub("[^a-zA-Z0-9\s]", "", query)
    words = line.lower().split()
    for word in words:
        if word in key_words_list:
            idx = key_words_list.index(word)
            vector[idx, 0] += 1
    return vector


def ac_distance(a, b):
    """ 对两个向量计算夹角余弦距离 """
    product_sum = np.sum(np.multiply(a, b))
    a_mod = np.sqrt(np.sum(np.multiply(a, a)))
    b_mod = np.sqrt(np.sum(np.multiply(b, b)))
    output = product_sum / (a_mod * b_mod)
    return output


def q_similarity(new_A_k_T, books_list, cos_threshold):
    """ 计算查询与各个书名之间的夹角余弦距离，并返回大于cosine threshold的所有书名 """
    original = new_A_k_T[:len(books_list), :]
    query = new_A_k_T[-1]
    result = []
    for i in range(original.shape[0]):
        distance = ac_distance(original[i], query)
        if distance >= cos_threshold:
            result.append([books_list[i], distance])
        else:
            pass
    if len(result) == 0:
        print(u'请降低cos_threshold')
    else:
        result = sorted(result, key=lambda dis: dis[1], reverse=True)
        for j in result:
            print(u'您的查询与%s的夹角余弦距离为%.3f' % (j[0], j[1]))
        return result


# ==================== 可视化 ====================
def reduce_dim(x, y, S):
    """ 降维 """
    new_x = np.matmul(np.matmul(x, y), np.linalg.inv(S))
    return new_x


def cos_thd2degree(cos_treshold):
    """ 将阀值转换为角度 """
    a = math.acos(cos_treshold)
    degree = math.degrees(a)
    return degree


def scope(degree, query_point):
    """ 可视化夹角余弦距离范围 """
    theta = math.degrees(math.atan(query_point[1] / query_point[0]))
    if theta + degree > 90:
        k1 = 0
    else:
        k1 = math.tan(math.radians(theta + degree))
    if theta - degree < -90:
        k2 = 0
    else:
        k2 = math.tan(math.radians(theta - degree))
    return k1, k2


def visualize(key_words_list, books_list, k_words, k_books, query, cos_treshold):
    """ 可视化结果 """
    plt.figure(figsize=[8, 16])

    # 设立直角坐标系：
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    # plt.axis([0, 0.8, -0.8, 0.8])
    plt.scatter(x=k_words[:, 0], y=k_words[:, 1], marker='o')
    for i in range(len(key_words_list)):
        plt.text(x=k_words[i, 0], y=k_words[i, 1], s=key_words_list[i], fontsize=16)
    if query is None:
        plt.scatter(x=k_books[:, 0], y=k_books[:, 1], marker='^')
        for j in range(len(books_list)):
            plt.text(x=k_books[j, 0], y=k_books[j, 1], s=books_list[j], fontsize=16)
        plt.legend(["keywords", "books"], loc='upper right')
    else:
        plt.scatter(x=k_books[:len(books_list), 0], y=k_books[:len(books_list), 1], marker='^')
        for j in range(len(books_list)):
            plt.text(x=k_books[j, 0], y=k_books[j, 1], s=books_list[j], fontsize=16)
        q_point = [k_books[-1, 0], k_books[-1, 1]]
        plt.scatter(x=q_point[0], y=q_point[1], marker='v')
        plt.text(x=q_point[0], y=q_point[1], s='QUERY', fontsize=16, color='r')
        plt.legend(["keywords", "books", "QUERY"], loc='upper right')
        degree = cos_thd2degree(cos_treshold)
        k1, k2 = scope(degree, q_point)
        x = np.linspace(0., q_point[0], 256)
        if k1 == 0:
            pass
        else:
            y1 = k1 * x
            plt.plot(x, y1, 'r--')
        if k2 == 0:
            pass
        else:
            y2 = k2 * x
            plt.plot(x, y2, 'r--')
        y3 = (q_point[1] / q_point[0]) * x
        plt.plot(x, y3, 'r')
    plt.show()


# ==================== 更新 ====================
def update(old_file, new_file):
    lines_old = load_txt(old_file)
    lines_new = load_txt(new_file)
    lines_old.extend(lines_new)
    return lines_old


# ==================== LSI ====================
def lsi(book_names_file, stop_words_file, save_A, save_A_k, save_new_A,  save_new_A_k,
        theta=0.4, cos_treshold=0.6, query=None, update_file=None):
    """
    :param book_names_file: 数据集文件路径
    :param stop_words_file: 停用词文件路径
    :param save_A: 关键词-书名 矩阵A保存路径
    :param save_A_k: A的 k-秩近似矩阵保存路径
    :param save_new_A: 更新书名后的 关键词-书名 矩阵A保存路径
    :param save_new_A_k: 更新书名后A的 k-秩近似矩阵保存路径
    :param theta: k-秩近似矩阵阈值，影响k的取值，由于可视化时只能现实2维数据，因此k=2时可视化结果最为准确，建议无更新时设置theta=0.4，有更新时设置theta=0.3
    :param cos_treshold: 夹角余弦距离阈值，值越大相似度越高
    :param query: 查询测试，查询与输入相关的书名，相关书名数量由cos_treshold控制，可设置None关闭
    :param update_file: 新书名路径，用于更新测试，可输入None关闭
    """
    # 生成 关键字-书名 矩阵A
    if update_file is None:
        books_lines = load_txt(book_names_file)
    else:
        books_lines = update(book_names_file, update_file)
    stop_words_lines = load_txt(stop_words_file)
    books_list = generate_books_list(books_lines)
    book_names_list = generate_book_names(books_lines)
    key_words_list = generate_keywords(books_lines, stop_words_lines)
    A = term_doc_matrix(book_names_list, key_words_list)

    # 生成k-秩近似矩阵A_k
    if query is not None:
        q_vector = query_vector(query, key_words_list)
        A = np.append(A, q_vector, 1)
        [A_k, U_k, S_k, V_k] = k_rank_matrix(A, theta)
        k_words = reduce_dim(x=A_k, y=np.transpose(V_k), S=S_k)
        print(u'\nk维词汇矩阵=\n', k_words, k_words.shape)
        k_books = reduce_dim(x=np.transpose(A_k), y=U_k, S=S_k)
        print(u'\nk维文档矩阵=\n', k_books, k_books.shape)
        q_similarity(k_books, books_list, cos_treshold)

    else:
        [A_k, U_k, S_k, V_k] = k_rank_matrix(A, theta)
        # k_words1 = U_k[:, 0] * S_k[0][0]
        # k_words2 = U_k[:, 1] * S_k[1][1]
        # k_words = [k_words1, k_words2]
        # k_books1 = V_k.T[:, 0] * S_k[0][0]
        # k_books2 = V_k.T[:, 1] * S_k[1][1]
        # k_books = [k_books1, k_books2]
        k_words = reduce_dim(x=A_k, y=np.transpose(V_k), S=S_k)
        print(u'\nk维词汇矩阵=\n', k_words)
        k_books = reduce_dim(x=np.transpose(A_k), y=U_k, S=S_k)
        print(u'\nk维文档矩阵=\n', k_books)

    # 可视化
    visualize(key_words_list, books_list, k_words, k_books, query, cos_treshold)

    # 保存A_k
    if update_file is None:
        save_txt(data=A, save_file=save_A, fmt='%d')
        save_txt(data=A_k, save_file=save_A_k)
    else:
        save_txt(data=A, save_file=save_new_A, fmt='%d')
        save_txt(data=A_k, save_file=save_new_A_k)


if __name__ == '__main__':
    lsi(book_names_file="./dataset.txt",    # ==>数据集文件路径
        stop_words_file="./stop_words.txt", # ==>停用词文件路径
        save_A="./A.txt",                    # ==>关键词-书名 矩阵A保存路径
        save_A_k="./A_k.txt",                # ==>A的 k-秩近似矩阵保存路径
        save_new_A="./new_A.txt",            # ==>更新书名后的 关键词-书名 矩阵A保存路径
        save_new_A_k="./new_A_k.txt",        # ==>更新书名后A的 k-秩近似矩阵保存路径
        theta=0.4,                           # ==>k-秩近似矩阵阈值，影响k的取值，由于可视化时只能现实2维数据，因此k=2时可视
                                             #    化结果最为准确，建议无更新时设置theta=0.4，有更新时设置theta=0.3
        cos_treshold=0.9,                    # ==>夹角余弦距离阈值，值越大相似度越高
        # query='Application and theory',
        query=None,                           # ==>查询测试，查询与输入相关的书名，相关书名数量由cos_treshold控制，
                                              #    可设置None关闭
        # update_file="./new_data.txt")
        update_file=None)                     # ==>新书名路径，用于更新测试，可输入None关闭

