# coding=utf-8
from __future__ import absolute_import, division, print_function
import numpy as np


def svd(A): # 奇异值分解
    ATA = np.matmul(A.T, A)
    eigen_values, eigen_vector = np.linalg.eig(ATA)

    sigma = np.sqrt(eigen_values)
    sort_id = np.argsort(-sigma)
    sigma = sigma[sort_id]
    V = eigen_vector[:, sort_id]
    VT = V.T
    S = np.diag(sigma)
    _S = np.linalg.inv(S)

    U = np.matmul(np.matmul(A, V), _S)

    result = np.matmul(np.matmul(U, S), VT)
    print('U=\n', U)
    print('\nS=\n', S)
    print('\nV=\n', V)
    print('\nA=\n', result)
    print('\nVT=\n', VT)

    return U, sigma, VT



# A = np.array([[5, 5],[-1, 7]])
A = np.array([[1, 1], [1, 1], [0,0]])
# A = np.array([[4, 0], [3, -5]])
if __name__ == '__main__':
    svd(A)
    [U, S, V] = np.linalg.svd(A)
    print('\n_U=\n', U)
    print('\n_S=\n', S)
    print('\n_V=\n', V)