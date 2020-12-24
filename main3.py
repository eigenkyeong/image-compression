#3번 문제 소스코드
import cv2
import numpy as np
import math

# load img
imageFile = './lena.bmp'
A = cv2.imread(imageFile, 0)
#A = A / 255
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

# normalize 함수
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


# DHWT B left
def DHWT(A, H):
    return H.T @ A @ H

# IDHWT A left
def IDHWT(B, H):
    return H @ B @ H.T


# compare
def Show_re(m1, m2, a):
    if m1[0, 0] == m2[0, 0]:
        print(m1[0, 0])
    else:
        print(m1[0, 0], m2[0, 0])
        for i in range(n):
            for j in range(n):
                if m1[i, j] != m2[i, j]:
                    print(a + " : False!")
                    print(i, j)
                    return False

    print(a + ": Same!")
    return True


# (a)
#H = normalize(haarMatrix(n), n)
H = haarMatrix(n)
Hl = H.T[:n//2, :]
Hh = H.T[n//2:, :]

matA1 = IDHWT(A, Hl)
matA2 = Hl @ A @ Hh.T
matA3 = Hh @ A @ Hl.T
matA4 = IDHWT(A, Hh)

resultA = np.vstack([np.hstack([matA1, matA2]), np.hstack([matA3, matA4])])

Show_re(DHWT(A, H), resultA, '(a)')

# (b)
matB1 = Hl.T @ Hl @ A @ Hl.T @ Hl
matB2 = Hl.T @ Hl @ A @ Hh.T @ Hh
matB3 = Hh.T @ Hh @ A @ Hl.T @ Hl
matB4 = Hh.T @ Hh @ A @ Hh.T @ Hh

resultB = matB1 + matB2 + matB3 + matB4

Show_re(IDHWT(DHWT(A, H), H), resultB, '(b)')

# (c)
'''
cv2.imwrite('./(tcb1).jpg', matB1 * 255)
cv2.imwrite('./(tcb2).jpg', matB2 * 255)
cv2.imwrite('./(tcb3).jpg', matB3 * 255)
cv2.imwrite('./(tcb4).jpg', matB4 * 255)
'''

# (d)
Hll = Hl[:n//4, :]
Hlh = Hl[n//4:, :]

matD1 = Hll.T @ Hll @ A @ Hll.T @ Hll
matD2 = Hll.T @ Hll @ A @ Hlh.T @ Hlh
matD3 = Hlh.T @ Hlh @ A @ Hll.T @ Hll
matD4 = Hlh.T @ Hlh @ A @ Hlh.T @ Hlh

resultD = matD1 + matD2 + matD3 + matD4
Dl = Hl.T @ Hl @ A @ Hl.T @ Hl

Show_re(Dl, resultD, '(d)')
'''
cv2.imwrite('./(brd1).jpg', matD1 * 255)
cv2.imwrite('./(brd2).jpg', matD2 * 255)
cv2.imwrite('./(brd3).jpg', matD3 * 255)
cv2.imwrite('./(brd4).jpg', matD4 * 255)
'''

