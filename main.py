import matplotlib.pyplot as plt
import numpy.linalg as lin
import numpy as np


def ll_fucntion(xx, yy):
    return xx + yy


def border(x):
    return x ** 2


def build_intial(n, net, rules):
    # Rules: 0 = 𝑔1(𝑦)
    #        1 = 𝑔2(𝑦)
    #        2 = 𝑔3(𝑥)
    #        3 = 𝑔4(𝑥)
    net[:, 0] = np.array(rules[0]) - 1
    net[:, n - 1] = np.array(rules[1]) - 2/np.pi * np.arctan(np.array(rules[1]))
    net[0, :] = np.array(rules[2])
    net[n - 1, :] = rules[3] - 2/np.pi * np.array(list(map(lambda x: np.arctan(1/x) if x != 0 else 0, rules[3])))
    return net


def lindex(i, j, m):
    # Assume i and j mapped to known(i, j) index
    return (i - 1) * m + (j - (i - 1))


def bulid_matrix_A_b(m, net):
    def known(i, j):
        return i == 0 or j == 0 or i == m or j == m

    def lin_index(i, j):
        # Assume i and j mapped to known(i, j) index
        return (i - 1) * m + (j - (i - 1))

    n_unknown = (m - 1) ** 2
    A = np.zeros((n_unknown, n_unknown))
    b = np.zeros(n_unknown)

    for i in range(1, m):
        for j in range(1, m):
            mid = 1
            east_ind = i, j + 1
            west_ind = i, j - 1
            nord_ind = i + 1, j
            sout_ind = i - 1, j

            A[lin_index(i, j) - 1, lin_index(i, j) - 1] = mid

            if not known(*east_ind):
                A[lin_index(i, j) - 1, lin_index(*east_ind) - 1] = -1 / 4
            else:
                b[lin_index(i, j) - 1] += net[east_ind] / 4

            if not known(*west_ind):
                A[lin_index(i, j) - 1, lin_index(*west_ind) - 1] = -1 / 4
            else:
                b[lin_index(i, j) - 1] += net[west_ind] / 4

            if not known(*sout_ind):
                A[lin_index(i, j) - 1, lin_index(*sout_ind) - 1] = -1 / 4
            else:
                b[lin_index(i, j) - 1] += net[sout_ind] / 4

            if not known(*nord_ind):
                A[lin_index(i, j) - 1, lin_index(*nord_ind) - 1] = -1 / 4
            else:
                b[lin_index(i, j) - 1] += net[nord_ind] / 4

    return A, b


def build_net(M, rules):
    n = M + 1
    net = np.zeros((n, n))
    net = build_intial(n, net, rules)

    return net


def choose_gamma(matrix):
    eig = lin.eigvals(matrix)
    return 2 / (eig.max() + eig.min())


def solve(A, b, M, eps):
    def converge(a, b, epsilon):
        for i in range(len(a)):
            if np.abs(a[i] - b[i]) / max(1, np.abs(b[i])) > epsilon:
                return False
        return True

    gamma = choose_gamma(A)
    n = (M - 1) ** 2
    comp = np.eye(n) - gamma * A
    x = comp.dot(np.ones(n)).T + gamma * b
    x_prev = np.ones(n)
    temp = np.ones(n)
    i = 0
    while not converge(x, temp, eps):
        temp = x_prev
        x = comp.dot(temp).T + gamma * b
        x_prev = x
        i += 1

    print(f'Iterations taken: {i}')
    return x


if __name__ == '__main__':
    import glob

    for file in glob.glob('in*'):
        with open(file) as input:
            lines = input.readlines()
            M, eps = lines[0].split()
            M = int(M)
            eps = float(eps)

            gs = []
            for line in lines[1:]:
                gs.append(list(map(lambda x: float(x),line.split())))

            net = build_net(M, gs)
            print(net)
            A, b = bulid_matrix_A_b(M, net)
            print(b)
            x = np.array(list(map(lambda x: 0 if x <= eps else x, solve(A, b, M, eps))))
            print(x)
            print(lin.solve(A, b))

            # plt.imshow(net)
            # plt.show()
