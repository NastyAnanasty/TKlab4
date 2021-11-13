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
#------------------------------------------------------------

# -----------------------------4.3---------------------------
def gen_G(r, m):
    result = []
    if(r == 0):
        result = np.random.randint(1, 2, (1, pow(2, m)))
    if(r == m):
        string = np.append(np.zeros(pow(2,m)-1), np.array(1))
        result = np.vstack((G(m-1, m), string))
    if(r > 0 and r < m):
        zeros = np.random.randint(0, 1, (1, pow(2, m-1)))
        string1 = np.hstack((G(r, m-1), G(r, m-1)))
        g = G(r-1, m-1)
        string2 = np.hstack((np.zeros(g.shape),g))
        result = np.vstack((string1, string2))
    return result

def check_G(r, m):
    eye = np.eye(pow(2,m))
    g = G(r, m)
    return np.concatenate((eye, g))
#------------------------------------------------------------


if __name__ == '__main__':
    #-------print 4.1----------
    #print(gen_matr_gol())
    #print(check_matr_gol())
    #-------print 4.3----------
    #print(gen_G(3,4)
    #print(check_G(3,4)

