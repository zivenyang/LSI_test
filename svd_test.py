# coding=utf-8
from __future__ import absolute_import, division, print_function
import numpy as np


def svd(A): # 奇异值分解
    AAT = np.matmul(A, A.T)
    eigen_values2, eigen_vectors2 = np.linalg.eig(AAT)

    ATA = np.matmul(A.T, A)
    eigen_values, eigen_vectors = np.linalg.eig(ATA)
    print(u"特征值=" , eigen_values)
    print(u"特征向量=", eigen_vectors)
    print(u"特征值2=" , eigen_values2)
    print(u"特征向量2=", eigen_vectors2)

    U = eigen_vectors2
    V = eigen_vectors.T
    for i in range(eigen_values2.size):
        if eigen_values2[i] > 0:
            eigen_values2[i] = np.sqrt(eigen_values2[i])
        else:
            eigen_values2[i] = 0


    S = np.zeros([eigen_values2.shape[0], eigen_values2.shape[0]])
    for i in range(eigen_values2.shape[0]):
        if eigen_values2[i] is not 0:
            S[i][i] = eigen_values2[i]
    s = S[:U.shape[0], :V.shape[0]]

    result = np.matmul(np.matmul(U, s), V)

    print('U=', U)
    print('S=', s)
    print('V=', V.T)
    print('A=', result)
    # sort_id = np.argsort(-sigma)
    # new_sigma = sigma[sort_id]
    # print('new_sigma=', new_sigma)

    # new_V = V[:, sort_id]
    # print('new_V=', new_V)

    # A_V = np.matmul(A, new_V)
    # print(A_V)
    # U = np.matmul(A_V, np.linalg.inv(np.diag(new_sigma)))

    # print('U=', U)

    # result = np.matmul(np.matmul(U, np.diag(new_sigma)), new_V.T)
    # print('result=',result)
    # S = new_sigma
    # V = new_V

    return U, S, V

# A = np.array([[5, 5],[-1, 7]])
A = np.array([[1, 1], [1, 1], [0,0]])
if __name__ == '__main__':
    svd(A)