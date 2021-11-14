import copy

import numpy as np


# Создаем рандомную матрицу, заполненную 0 и 1 по заданному количеству строк и столбцов
def create_matrix(row, column):
    matrix = np.random.randint(0, 9, (row, column))
    for r in range(0, row):
        matrix[r] = matrix[r] % 2
    return matrix


# Поменять местами две заданные строчки матрицы
def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]


# Суммирование двух строчек матрицы, сумма сохраняется в строчку row
def sum_rows(matrix, row, rowsum):
    matrix[row] = np.array([((a + k) % 2) for a, k in zip(matrix[row], matrix[rowsum])])


# Количество строк матрицы
def row_size(matrix):
    return matrix.shape[0]


# Кличество столбцов матрицы
def column_size(matrix):
    return matrix.shape[1]


# Ступенчатый вид
def ref(matrix):
    column = 0  # Первый столбец
    leadrow = 0  # Ведущая строка
    while column < column_size(matrix):  # Идём по столбцам матрицы
        for r in range(leadrow + 1, row_size(matrix)):  # Идём от следующей от ведущей строки до последней строки
            if matrix[r][column] > matrix[leadrow][column]:  # Если элемент больше
                currentrow = r  # Запоминаем строчку
                swap_rows(matrix, currentrow, leadrow)  # Меняем строчку, чтобы была новая ведущая
            if matrix[r][column] == matrix[leadrow][column]:  # Если элементы равны
                sum_rows(matrix, r, leadrow)  # Суммируем строчки чтобы занулить элементы
        if np.all(matrix[leadrow:, column] == 0):  # Если все элементы столбца ниже ведуще строки = 0
            leadrow = leadrow  # Ведущая строка не меняется
        else:
            leadrow += 1  # Ведущая строка становится следующей
        column += 1  # Переходим к следующему столбцу
    return matrix  # Возвращаем полученную матрицу


def rref(originalMatrix):  # приведённый ступенчатый вид
    matrix = copy.deepcopy(no_zero_matrix(originalMatrix))
    k = matrix.shape[0]
    n = matrix.shape[1]
    curRow = 0  # текущая ступенька
    for j in range(n):  # проходим по всем столбцам
        for i in range(curRow + 1, k):  # по строкам кроме уже построенных ступенек
            if matrix[i][j] == 1:  # в этой строчке на данном столбце единиц быть не может
                if matrix[curRow][j] == 0:  # если ступенька не построена
                    matrix[curRow] = (matrix[i] + matrix[curRow]) % 2  # строим ступеньку
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2  # обнуляем единицы в данной строчке
                else:  # если уже построена
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2  # обнуляем единицы в данной строчке
        if matrix[curRow][j] == 1:  # если в даном столбце построили новую ступеньку
            for i in range(curRow):  # по уже построенным ступенькам
                if matrix[i][j] == 1:  # в этой строчке на данном столбце единиц быть не может
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2  # складываем строки
            curRow += 1  # начинаем строить следующую ступеньку
            if curRow == k:  # если все ступеньки построены, выходим
                break
    return matrix[0:curRow]


# Убираем нулевые строчки
def no_zero_matrix(matrix):
    r = 0  # Первая строчка
    while r < row_size(matrix):  # Пока не пройдем все строчки
        if np.all(matrix[r, :] == 0):  # Если все элементы строчки = 0
            matrix = np.delete(matrix, r, 0)  # Удаляем эту строчку
            r = r  # Не сдвигаем номер строчки
        else:
            r += 1  # Если ничего не удаляли, то переходим к новой строчке
    return matrix


class LinearMatrix(object):

    def __init__(self, matrix):
        self.matrix = matrix
        self.ref_matrix = ref(self.matrix)
        self.n = self.ref_matrix.shape[1]  # число столбцов
        self.k = self.ref_matrix.shape[0]  # число строк
        self.lead_columns = []  # лидирующие столбцы ступенчатой матрицы

    def rref(self):
        new_matrix = rref(self.matrix)
        self.n = column_size(new_matrix)
        self.k = row_size(new_matrix)
        return new_matrix

    def get_leading(self):
        lead = []

        for i in range(self.n):
            if len(lead) < self.k:
                for j in range(len(lead), self.k):
                    if self.matrix[j][i] == 1:
                        lead.append(i)
                        break

        # print(lead)
        return lead

    def get_short_matrix(self):
        self.lead_columns = self.get_leading()
        short_matrix = np.delete(self.matrix, self.lead_columns, 1)
        return short_matrix

    def getH(self):
        self.matrix = self.rref()

        short_matrix = self.get_short_matrix()
        identity_matrix = np.identity(self.n - self.k)

        h_matrix_rows = self.n - self.k + row_size(short_matrix)
        h_matrix = create_matrix(h_matrix_rows, self.n - self.k)

        j = 0
        for i in range(0, h_matrix_rows):
            if self.lead_columns.count(i) > 0:
                index = self.lead_columns.index(i)
                h_matrix[i] = short_matrix[index]
            else:
                h_matrix[i] = identity_matrix[j]
                j += 1

        return h_matrix

    def admitted_words(self):
        ref_matrix = ref(self.matrix)
        words = ref_matrix.copy()
        words = np.vstack([words, np.zeros(words.shape[1], dtype=int)])
        for i in range(ref_matrix.shape[0] - 1):
            for j in range(i + 1, ref_matrix.shape[0]):
                row = (ref_matrix[i] + ref_matrix[j]) % 2
                append = True
                for k in range(words.shape[0]):
                    if np.array_equal(words[k], row):
                        append = False
                        break
                if append:
                    words = np.vstack([words, row])
        return words

    def k_admitted_words(self):
        I = np.mat(np.eye(self.k, dtype=int))
        words = I.copy()
        index_to_add = I.shape[0]
        for i in range(index_to_add):
            for j in range(0, I.shape[0]):
                row = (I[i] + I[j]) % 2
                append = True
                for k in range(0, words.shape[0]):
                    if np.array_equal(words[k], row):
                        append = False
                        break
                if append:
                    words = np.vstack([words, row])
                    index_to_add += 1
        return words @ ref(self.matrix) % 2

    def distance(self):
        d = self.ref_matrix.shape[1]
        for i in range(self.ref_matrix.shape[0]):
            for j in range(i + 1, self.ref_matrix.shape[0]):
                d = min(np.count_nonzero((self.ref_matrix[j] - self.ref_matrix[i]) % 2), d)
        t = d - 1
        print("d = ", d)
        print("t = ", t)


if __name__ == '__main__':
    matrix = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                       [1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 1, 1, 1]])
    linear_matrix = LinearMatrix(matrix)

    print("H = ", linear_matrix.getH())
    allowed_words = linear_matrix.admitted_words()
    k_allowed_words = linear_matrix.k_admitted_words()
    print(allowed_words)
    print(k_allowed_words)
    linear_matrix.distance()