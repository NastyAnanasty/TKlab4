import numpy as np


# -----------------------------4.1---------------------------
def B_matrix():
    B = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                  [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                  [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1], [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                  [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                  [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
    return B


def gen_matr_gol():
    gen = np.eye(12)
    b = B_matrix()
    gen = np.hstack((gen, b))
    return gen


def check_matr_gol():
    check = np.eye(12)
    b = B_matrix()
    check = np.vstack((check, b))
    return check


# -----------------------------4.2---------------------------
def decode():
    small_wold = np.random.randint(0, 2, (1, 12))
    word = np.dot(small_wold, gen_matr_gol()) % 2
    print("Отправленное сообщение:", word)
    mistake = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    er_word = (word + mistake) % 2
    print("Принятое сообщение:", er_word)
    h = check_matr_gol()
    b = B_matrix()
    u = np.zeros([1, 12])
    # ------------пункт 1--------------------
    syndromH = np.dot(er_word, h) % 2
    # syndromH = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1]])
    # ------------пункт 2--------------------
    if np.count_nonzero(syndromH) <= 3:
        u = np.hstack((syndromH, u))
        print("Найдены ошибки в позициях:", u)
        return u
    else:
        # ------------пункт 3--------------------
        for i in range(0, b.shape[0]):
            syndromH_copy = np.zeros([1, 12])
            for j in range(0, b.shape[0]):
                syndromH_copy[0, j] = (syndromH[0, j] + b[i, j]) % 2
            if (np.count_nonzero(syndromH_copy) <= 2):
                u[0, i] = 1
                u = np.hstack((syndromH_copy, u))
                print("Найдены ошибки в позициях:", u)
                return u
            # ------------пункт 4--------------------
        if i == b.shape[0] - 1:
            syndromB = np.dot(syndromH, b)
            # ------------пункт 5--------------------
            if np.count_nonzero(syndromB) <= 3:
                u = np.hstack((u, syndromB))
                print("Найдены ошибки в позициях:", u)
                return u
            else:
                # ------------пункт 6--------------------
                for i in range(0, b.shape[0]):
                    syndromB_copy = np.zeros([1, 12])
                    for j in range(0, b.shape[0]):
                        syndromB_copy[0, j] = (syndromB[0, j] + b[i, j]) % 2
                    if (np.count_nonzero(syndromB_copy) <= 2):
                        u[0, i] = 1
                        u = np.hstack((u, syndromB_copy))
                        print("Найдены ошибки в позициях:", u)
                        return u
                    # ------------пункт 7--------------------
                if j == b.shape[0] - 1:
                    return print("Ошибка не определена, оправьте повторно сообщение")


# -----------------------------4.3---------------------------
def G(r, m):
    result = []
    if (r == 0):
        result = np.random.randint(1, 2, (1, pow(2, m)))
    if (r == m):
        string = np.append(np.zeros(pow(2, m) - 1), np.array(1))
        result = np.vstack((G(m - 1, m), string))
    if (r > 0 and r < m):
        zeros = np.random.randint(0, 1, (1, pow(2, m - 1)))
        string1 = np.hstack((G(r, m - 1), G(r, m - 1)))
        g = G(r - 1, m - 1)
        string2 = np.hstack((np.zeros(g.shape), g))
        result = np.vstack((string1, string2))
    return result


def check_G(r, m):
    eye = np.eye(pow(2, m))
    g = G(r, m)
    return np.concatenate((eye, g))

# ------------------------------------------------------------


if __name__ == '__main__':
    # -------print 4.1----------
    # print(gen_matr_gol())
    # print(check_matr_gol())
    # -------print 4.2----------
    res = decode()
    # -------print 4.3----------
    #print(G(3, 4))
    #print(check_G(3, 4))
