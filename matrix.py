from random import random


class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for _ in range(self.cols):
                self.data[i].append(0)

    def __str__(self):
        return str(self.data)

    @staticmethod
    def fromArray(arr):
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.data[i][0] = arr[i]
        return m

    def toArray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                r = random()
                r = r if random() < 0.5 else r - 1
                self.data[i][j] = r

    @staticmethod
    def subtract(a, b):
        result = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

    def add(self, n):
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n

    @staticmethod
    def transpose(matrix):
        result = Matrix(matrix.cols, matrix.rows)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[j][i] = matrix.data[i][j]
        return result

    # Matrix Multiplication
    @staticmethod
    def multiply1(a, b):
        if a.cols != b.rows:
            print('Invalid Matrices!')
            return

        result = Matrix(a.rows, b.cols)

        for i in range(result.rows):
            for j in range(result.cols):
                s = 0.0
                for k in range(a.cols):
                    s += a.data[i][k] * b.data[k][j]
                result.data[i][j] = s
        return result

    def multiply2(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print('Invalid Matrices!')
                return
            # Hadamard Product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else:
            # Scalar Product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    def map1(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])

    @staticmethod
    def map2(m, func):
        result = Matrix(m.rows, m.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.data[i][j] = func(m.data[i][j])
        return result
