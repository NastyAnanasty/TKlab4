import random

import numpy as np

# -----------------------------4.1---------------------------
import lab2


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

def K_mult(a, b):
    isFirst1 = True
    result = []
    for a_i in a:
        string = []
        isFirst2 = True
        for a_ij in a_i:
            if(isFirst2):
                string = a_ij * b
                isFirst2 = False
            else:
                string = np.hstack((string, a_ij * b))
        if(isFirst1):
            result = string
            isFirst1 = False
        else:
            result = np.concatenate((result, string))
    return result

def H(i, m):
    H = np.array([[1,1],
                  [1,-1]])
    eye1 = np.eye(pow(2, m-i))
    eye2 = np.eye(pow(2, i-1))
    return K_mult(K_mult(eye1, H), eye2)

# ------------------------------------------------------------

# -----------------------------4.4, 4.5-----------------------
def get_code_word(matrix, k):
    k_word = []
    for i in range(k):
        k_word.append(lab2.round_num(random.uniform(0, 1)))
    n_word = lab2.code_word_from_k_to_n(matrix, k_word) % 2
    return n_word


def RM_decode(r, m):
    g_matrix = G(r, m)
    check_matrix = check_G(r, m)
    error_num_arr = [1, 2] if m == 3 else [1, 2, 3, 4]
    for error_num in error_num_arr:
        n_word = get_code_word(g_matrix, g_matrix.shape[0])
        n_word_error = []
        if error_num == 1:
            n_word_error = lab2.make_single_mistake_in_n_word(n_word) % 2
        if error_num == 2:
            n_word_error = lab2.make_double_mistake_in_n_word(n_word) % 2
        if error_num == 3:
            n_word_error = lab2.make_triple_mistake_in_n_word(n_word) % 2
        if error_num == 4:
            n_word_error = lab2.make_quadro_mistake_in_n_word(n_word) % 2

        n_word_error_1 = [-1 if x == 0 else x for x in n_word_error]
        decode_word = n_word_error_1 @ check_matrix
        max_val = max(decode_word, key=abs)
        index_of_max = decode_word.index(max_val)
        index_of_max_1 = index_of_max

        fixed_word = []
        for i in range(m):
            fixed_word[i] = index_of_max_1 % 2
            index_of_max_1 = index_of_max_1 / 2
        fixed_word.insert(0, 1 if decode_word[index_of_max] >= 0 else 0)
        if (np.array_equiv(n_word, fixed_word @ g_matrix)):
            print("Ошибка для r = {}, m = {}, кратности ошибки = {}: ИСПРАВЛЕНА".format(r, m, error_num))
        else:
            print("Ошибка для r = {}, m = {}, кратности ошибки = {}: НЕ ИСПРАВЛЕНА".format(r, m, error_num))


# ------------------------------------------------------------


if __name__ == '__main__':
    # -------print 4.1----------
    a = gen_matr_gol()
    h = check_matr_gol()
    print(gen_matr_gol())
    print(check_matr_gol())
    # -------print 4.2----------
    res = decode()
    # -------print 4.3----------
    # print(G(3, 4))
    # print(H(3, 4))
    # -------print 4.4----------
    RM_decode(1, 3)
    # -------print 4.5----------
    RM_decode(1, 4)

