#1, 2번 문제 소스코드
import cv2
import numpy as np
import math

# load img
imageFile = './realow.jpg'
A = cv2.imread(imageFile, 0)
A = A / 255
n = len(A)


# denormalized_HaarMatrix 반환 함수
def haarMatrix(n, normalized=False):
    if n == 1:
        return np.array([1.0])
    else:
        hm = np.kron(haarMatrix(int(n / 2)), ([1], [1]))
        hi = np.kron(np.identity(int(n / 2), dtype=int), ([1], [-1]))
        h = np.hstack([hm, hi])
        h = np.array(h, dtype=float)
        print(haarMatrix)
    if normalized:
        h = h * np.sum(np.abs(h), axis=0)**(-1/2)

    return h


# HaarMatrix normalize 함수
def normalize(h, n):
    for i in range(0, n):
        temp = 0
        j = 0

        for j in range(0, n):
            if h[j][i] != 0:
                temp += 1

        if temp != 0:
            h[:, i] = 1 / math.sqrt(temp) * h[:, i]

    return h


# B hat 생성 항수
def make_B_hat(B, s):
    k = 2**s

    B[:, k:] = 0
    B[k:, :] = 0
    return B


s = int(input("s: "))
H = normalize(haarMatrix(n), n)
B = H.T @ A @ H
B_hat = make_B_hat(B, s)

A_hat = H @ B_hat @ H.T
A = A_hat * 255


cv2.imwrite('./lower' + str(s) + '.jpg', A)
cv2.waitKey(0)