# Christian Prather
# Took approx 25 minutes
# Had to use numpy offical documentation heavily
import numpy as np

mat = np.zeros((9,9))
def norm_dist(mean, sigma):
    for row in range (9):
        for column in range(9):
            mat[row][column] = (1/(2 * np.pi * sigma **2)) * np.e **(-(((row - mean) ** 2) + ((column - mean) ** 2)/ (2 * sigma ** 2)))
    print(mat)
def main():
    A = np.matrix('4 -2; 1 1')
    B = np.matrix('3 4; 5 -1')
    X = np.array([1,2,3])
    Y = np.array([-1,2,-3])
    print(A)
    print(B)

    print("Det {}".format(np.linalg.det(A)))
    print("Trace {}".format(A.trace()))
    print("Inv {}".format(np.linalg.inv(A)))
    values, vectors = np.linalg.eig(A)
    print("Eigenvalues {} Eigenvector {}".format(values, vectors))
    print("AB {}".format(np.matmul(A, B)))
    print("BA {}".format(np.matmul(B, A)))
    print("X dot Y {}".format(np.dot(X,Y)))
    print("X cross Y {}".format(np.cross(X,Y)))
    norm_dist(0, 1.0)
if __name__ == "__main__":
    main()