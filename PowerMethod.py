import numpy as np
import scipy as sp


class Quaternion:
    def __init__(self, real, i, j, k):
        if np.shape(real) != np.shape(i) or np.shape(real) != np.shape(j) or np.shape(real) != np.shape(k) \
                or np.shape(i) != np.shape(j) or np.shape(i) != np.shape(k) or np.shape(j) != np.shape(k):
            raise ValueError('All parts must be of the same shape')
        self.real = real
        self.i = i
        self.j = j
        self.k = k

    @classmethod
    def from_real(cls, real):
        return Quaternion(real, np.zeros_like(real), np.zeros_like(real), np.zeros_like(real))

    def __repr__(self):
        if np.size(self.real) == 1:
            # return '{:}{:+}i{:+}j{:+}k'.format(float(self.real), float(self.i), float(self.j), float(self.k))
            return '{:.5}{:+.5}i{:+.5}j{:+.5}k'.format(float(self.real), float(self.i), float(self.j), float(self.k))
        else:
            res = np.zeros_like(self.real, dtype=Quaternion)
            for i in range(self.real.shape[0]):
                for j in range(self.real.shape[1]):
                    res[i, j] = Quaternion(self.real[i, j], self.i[i, j], self.j[i, j], self.k[i, j])
            return str(res)

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            if isinstance(other, Dual):
                return DualQuaternion.from_quat(self) + DualQuaternion.from_dual(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_quat(self) + other
            else:
                other = Quaternion.from_real(other)
        return Quaternion(self.real + other.real, self.i + other.i, self.j + other.j, self.k + other.k)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Quaternion(-self.real, -self.i, -self.j, -self.k)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = Quaternion.from_real(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            if isinstance(other, Dual):
                return DualQuaternion.from_quat(self) * DualQuaternion.from_dual(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_quat(self) * other
            else:
                other = Quaternion.from_real(other)
        return Quaternion(self.real * other.real - self.i * other.i - self.j * other.j - self.k * other.k,
                          self.real * other.i + self.i * other.real + self.j * other.k - self.k * other.j,
                          self.real * other.j - self.i * other.k + self.j * other.real + self.k * other.i,
                          self.real * other.k + self.i * other.j - self.j * other.i + self.k * other.real)

    def __rmul__(self, other):
        other = Quaternion(other, np.zeros_like(other), np.zeros_like(other), np.zeros_like(other))
        return other * self

    def __matmul__(self, other):
        if not isinstance(other, Quaternion):
            if isinstance(other, Dual):
                return DualQuaternion.from_quat(self) @ DualQuaternion.from_dual(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_quat(self) @ other
            else:
                other = Quaternion.from_real(other)
        return Quaternion(self.real @ other.real - self.i @ other.i - self.j @ other.j - self.k @ other.k,
                          self.real @ other.i + self.i @ other.real + self.j @ other.k - self.k @ other.j,
                          self.real @ other.j - self.i @ other.k + self.j @ other.real + self.k @ other.i,
                          self.real @ other.k + self.i @ other.j - self.j @ other.i + self.k @ other.real)

    def __rmatmul__(self, other):
        other = Quaternion(other, np.zeros_like(other), np.zeros_like(other), np.zeros_like(other))
        return other @ self

    def conjugate(self):
        return Quaternion(self.real, -self.i, -self.j, -self.k)

    def norm_sqr(self):
        return (self * self.conjugate()).real

    def magnitude(self):
        return np.sqrt(self.norm_sqr())

    def norm(self, mode=0):
        if mode == 'F':
            return np.linalg.norm(np.array([self.real, self.i, self.j, self.k]))
        return np.linalg.norm(self.magnitude())

    def __eq__(self, other):
        if isinstance(other, Dual):
            return (self == other.real) and (other.dual == np.zeros_like(self.real))
        if isinstance(other, DualQuaternion):
            return (self == other.real) and (other.dual == np.zeros_like(self.real))
        if not isinstance(other, Quaternion):
            other = Quaternion.from_real(other)
        return (self.real == other.real) and (self.i == other.i) and (self.j == other.j) and (self.k == other.k)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError
        if not isinstance(other, Quaternion):
            if isinstance(other, Dual):
                return DualQuaternion.from_quat(self) / DualQuaternion.from_dual(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_quat(self) / other
            else:
                return Quaternion(self.real / other, self.i / other, self.j / other, self.k / other)
        norm = other.norm_sqr()
        return self * (Quaternion.conjugate(other)/norm)

    def __rtruediv__(self, other):
        other = Quaternion.from_real(other)
        return other/self

    def inv(self):
        if self == 0:
            raise ZeroDivisionError
        return 1 / self

    def __pow__(self, power, modulo=None):
        if power >= 0:
            res = Quaternion.from_real(np.ones_like(self))
            for i in range(power):
                res *= self
            return res
        else:
            res = Quaternion.inv(self)
            return res ** power

    def transpose(self):
        return Quaternion(np.transpose(self.real), np.transpose(self.i), np.transpose(self.j), np.transpose(self.k))

    def __getitem__(self, item):
        return Quaternion(self.real[item], self.i[item], self.j[item], self.k[item])

    def similarizer(self):
        return Quaternion(np.sqrt(self.i**2 + self.j**2 + self.k**2)+self.i, np.zeros_like(self.i), -self.k, self.j)

    def similarize(self, mode=0):
        if mode:
            A = np.sqrt(self.i ** 2 + self.j ** 2 + self.k ** 2)
            return Quaternion(self.real, A, np.zeros_like(A), np.zeros_like(A))
        x = self.similarizer()
        return x.inv() * self * x

    @classmethod
    def unit_quaternion_to_so3(cls, quat):
        a = quat.real
        b = quat.i
        c = quat.j
        d = quat.k
        return np.array([[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*a*c + 2*b*d],
                         [2*a*d + 2*b*c, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b],
                         [2*b*d - 2*a*c, 2*a*b + 2*c*d, a**2 - b**2 - c**2 + d**2]])

    @classmethod
    def so3_to_unit_quaternion(cls, mat):
        a = np.sqrt(1 + np.trace(mat)) / 2
        if a > 1e-12:
            b = (mat[2, 1] - mat[1, 2])/(4 * a)
            c = (mat[0, 2] - mat[2, 0])/(4 * a)
            d = (mat[1, 0] - mat[0, 1])/(4 * a)
        else:
            b = np.sqrt((mat[1, 1] + mat[2, 2])/-2)
            if b > 1e-12:
                c = (mat[0, 1] + mat[1, 0])/(4 * b)
                d = (mat[2, 0] + mat[0, 2])/(4 * b)
            else:
                c = np.sqrt((mat[0, 0] + mat[2, 2])/-2)
                if c > 1e-12:
                    d = (mat[2, 1] + mat[1, 2])/(4 * c)
                else:
                    d = np.sqrt(mat[2, 2])
        return Quaternion(a, b, c, d)


class Dual:
    def __init__(self, real, dual):
        if np.shape(real) != np.shape(dual):
            raise ValueError('All parts must be of the same shape')
        self.real = real
        self.dual = dual

    @classmethod
    def from_real(cls, real):
        return Dual(real, np.zeros_like(real))

    def __repr__(self):
        if np.size(self.real) == 1:
            # return '{:}{:+}e'.format(float(self.real), float(self.dual))
            return '{:.5}{:+.5}e'.format(float(self.real), float(self.dual))
        else:
            res = np.zeros_like(self.real, dtype=Dual)
            for i in range(self.real.shape[0]):
                for j in range(self.real.shape[1]):
                    res[i, j] = Dual(self.real[i, j], self.dual[i, j])
            return str(res)

    def __add__(self, other):
        if not isinstance(other, Dual):
            if isinstance(other, Quaternion):
                return DualQuaternion.from_dual(self) + DualQuaternion.from_quat(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_dual(self) + other
            else:
                other = Dual.from_real(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = Dual.from_real(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Dual):
            if isinstance(other, Quaternion):
                return DualQuaternion.from_dual(self) * DualQuaternion.from_quat(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_dual(self) * other
            else:
                other = Dual.from_real(other)
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)

    def __rmul__(self, other):
        other = Dual.from_real(other)
        return other * self

    def __matmul__(self, other):
        if not isinstance(other, Dual):
            if isinstance(other, Quaternion):
                return DualQuaternion.from_dual(self) @ DualQuaternion.from_quat(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_dual(self) @ other
            else:
                other = Dual.from_real(other)
        return Dual(self.real @ other.real, self.real @ other.dual + self.dual @ other.real)

    def __rmatmul__(self, other):
        other = Dual.from_real(other)
        return other @ self

    def conjugate(self, mode=0):
        if mode:
            return self
        return Dual(self.real, -self.dual)

    def norm_sqr(self):
        return (self * self.conjugate()).real

    def sqrt(self):
        if self.real > 0:
            return Dual(np.sqrt(self.real), self.dual/(2 * np.sqrt(self.real)))
        elif self == 0:
            return self
        else:
            raise ValueError('Square root does not exist')

    def magnitude(self):
        signs = np.sign(self.real)
        signs[signs == 0] = np.sign(self.dual)[signs == 0]
        return self * signs

    def norm(self, mode=0):
        if mode:
            if mode == 'F':
                return np.sqrt(np.linalg.norm(self.real)**2 + np.linalg.norm(self.dual)**2)
            if self.real.any():
                return ((self.magnitude().transpose()) @ self.magnitude()).sqrt()
            else:
                return Dual(0, np.linalg.norm(self.dual))
        return np.linalg.norm(np.sqrt(self.norm_sqr()))

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return (self.real == other) and (self.dual == np.zeros_like(other.real))
        if isinstance(other, DualQuaternion):
            return (self.real == other.real) and (self.dual == other.dual)
        if not isinstance(other, Dual):
            other = Dual.from_real(other)
        return (self.real == other.real) and (self.dual == other.dual)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError
        if not isinstance(other, Dual):
            if isinstance(other, Quaternion):
                return DualQuaternion.from_dual(self) / DualQuaternion.from_quat(other)
            elif isinstance(other, DualQuaternion):
                return DualQuaternion.from_dual(self) / other
            else:
                return Dual(self.real / other, self.dual / other)
        if other.real == 0:
            if self.real == 0:
                return Dual.from_real(self.dual / other.dual)
            else:
                raise ZeroDivisionError('There is no solution for ({})/({})'.format(self, other))
        norm = other.norm_sqr()
        return self * (Dual.conjugate(other)/norm)

    def __rtruediv__(self, other):
        other = Dual.from_real(other)
        return other/self

    def inv(self):
        if self == 0:
            raise ZeroDivisionError
        return 1 / self

    def __pow__(self, power, modulo=None):
        if power >= 0:
            res = Dual.from_real(np.ones_like(self))
            for i in range(power):
                res *= self
            return res
        else:
            res = Dual.inv(self)
            return res ** power

    def transpose(self):
        return Dual(np.transpose(self.real), np.transpose(self.dual))

    def __getitem__(self, item):
        return Dual(self.real[item], self.dual[item])

    def __gt__(self, other):
        if self.real > other.real:
            return True
        elif (self.real == other.real) and (self.dual > other.dual):
            return True
        else:
            return False


class DualQuaternion:
    def __init__(self, real, ri, rj, rk, dual, di, dj, dk):
        if np.shape(real) != np.shape(dual):
            raise ValueError('All parts must be of the same shape')
        self.real = Quaternion(real, ri, rj, rk)
        self.dual = Quaternion(dual, di, dj, dk)

    @classmethod
    def from_quats(cls, real, dual):
        return DualQuaternion(real.real, real.i, real.j, real.k, dual.real, dual.i, dual.j, dual.k)

    @classmethod
    def from_quat(cls, real):
        dual = np.zeros_like(real.real)
        return DualQuaternion(real.real, real.i, real.j, real.k, dual, dual, dual, dual)

    @classmethod
    def from_dual(cls, dual):
        return DualQuaternion.from_quats(Quaternion.from_real(dual.real), Quaternion.from_real(dual.dual))

    @classmethod
    def from_real(cls, real):
        return DualQuaternion.from_quats(Quaternion.from_real(real), Quaternion.from_real(np.zeros_like(real)))

    def __repr__(self):
        if np.size(self.real.real) == 1:
            return '{}+({})e'.format(self.real, self.dual)
        else:
            res = np.zeros_like(self.real.real, dtype=DualQuaternion)
            for i in range(self.real.real.shape[0]):
                for j in range(self.real.real.shape[1]):
                    res[i, j] = DualQuaternion.from_quats(self.real[i, j], self.dual[i, j])
            return str(res)

    def __add__(self, other):
        if not isinstance(other, DualQuaternion):
            if isinstance(other, Quaternion):
                return self + DualQuaternion.from_quat(other)
            elif isinstance(other, Dual):
                return self + DualQuaternion.from_dual(other)
            else:
                other = DualQuaternion.from_real(other)
        return DualQuaternion.from_quats(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return DualQuaternion.from_quats(-self.real, -self.dual)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = DualQuaternion.from_real(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, DualQuaternion):
            if isinstance(other, Quaternion):
                return self * DualQuaternion.from_quat(other)
            elif isinstance(other, Dual):
                return self * DualQuaternion.from_dual(other)
            else:
                other = DualQuaternion.from_real(other)
        return DualQuaternion.from_quats(self.real * other.real, self.dual * other.real + self.real * other.dual)

    def __rmul__(self, other):
        other = DualQuaternion.from_real(other)
        return other * self

    def __matmul__(self, other):
        if not isinstance(other, DualQuaternion):
            if isinstance(other, Quaternion):
                return self @ DualQuaternion.from_quat(other)
            elif isinstance(other, Dual):
                return self @ DualQuaternion.from_dual(other)
            else:
                other = DualQuaternion.from_real(other)
        return DualQuaternion.from_quats(self.real @ other.real, self.dual @ other.real + self.real @ other.dual)

    def __rmatmul__(self, other):
        other = DualQuaternion.from_real(other)
        return other @ self

    def conjugate(self):
        return DualQuaternion.from_quats(self.real.conjugate(), self.dual.conjugate())

    def norm_sqr(self):
        res = self * self.conjugate()
        return Dual(res.real.real, res.dual.real)

    def magnitude(self):
        if np.size(self.real.real) == 1:
            if self.real == 0:
                return Dual(np.zeros(1), self.dual.magnitude())
            else:
                return self.norm_sqr().sqrt()
        real, dual = np.zeros_like(self.real.real, dtype=float), np.zeros_like(self.real.real, dtype=float)
        for i in range(real.shape[0]):
            real[i] = self[i].magnitude().real
            dual[i] = self[i].magnitude().dual
        return Dual(real, dual)

    def norm(self, mode=0):
        if mode == 'F':
            return np.sqrt(self.real.norm('F') + self.dual.norm('F') ** 2)
        if self.real.real.size == 1:
            return self.magnitude()
        dual = True
        for i in range(self.real.real.shape[0]):
            dual *= (self.real[i] != 0)
        if dual:
            return (self.magnitude().transpose() @ self.magnitude()).sqrt()
        else:
            return self.dual.norm()

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            return (self.real == other) and (self.dual == np.zeros_like(other.real))
        if isinstance(other, Dual):
            return (self.real == other.real) and (self.dual == other.dual)
        if not isinstance(other, DualQuaternion):
            other = DualQuaternion.from_real(other)
        return (self.real == other.real) and (self.dual == other.dual)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError
        if not isinstance(other, DualQuaternion):
            if isinstance(other, Quaternion):
                return self * DualQuaternion.from_quats(other.inv(), Quaternion.from_real(np.zeros_like(other)))
            elif isinstance(other, Dual):
                return self * DualQuaternion.from_dual(other.inv())
            else:
                return DualQuaternion.from_quats(self.real / other, self.dual / other)
        norm = other.norm_sqr()
        return self * (DualQuaternion.conjugate(other)/norm)

    def __rtruediv__(self, other):
        other = DualQuaternion.from_real(other)
        return other/self

    def inv(self):
        if self == 0:
            raise ZeroDivisionError
        return 1 / self

    def __pow__(self, power, modulo=None):
        if power >= 0:
            res = DualQuaternion.from_real(np.ones_like(self))
            for i in range(power):
                res *= self
            return res
        else:
            res = DualQuaternion.inv(self)
            return res ** power

    def transpose(self):
        return DualQuaternion.from_quats(self.real.transpose(), self.dual.transpose())

    def __getitem__(self, item):
        return DualQuaternion.from_quats(self.real[item], self.dual[item])

    def similarizer(self):
        A = np.sqrt(self.real.i ** 2 + self.real.j ** 2 + self.real.k ** 2)
        B = self.real.i * self.dual.i + self.real.j * self.dual.j + self.real.k * self.dual.k
        return DualQuaternion(A + self.real.i, np.zeros_like(A), -self.real.k, self.real.j,
                              self.dual.i + B / A, np.zeros_like(A), -self.dual.k, self.dual.j)

    def similarize(self, mode=0):
        if mode:
            A = np.sqrt(self.real.i ** 2 + self.real.j ** 2 + self.real.k ** 2)
            B = self.real.i * self.dual.i + self.real.j * self.dual.j + self.real.k * self.dual.k
            return DualQuaternion(self.real.real, A, np.zeros_like(A), np.zeros_like(A),
                                  self.dual.real, B / A, np.zeros_like(A), np.zeros_like(A))
        x = self.similarizer()
        return x.inv() * self * x

    @classmethod
    def se3_to_unit_dual_quaternion(cls, mat, vec):
        q = Quaternion.so3_to_unit_quaternion(mat)
        t = DualQuaternion(1, 0, 0, 0, 0, 1/2*vec[0, 0], 1/2*vec[1, 0], 1/2*vec[2, 0])
        return t*q

    @classmethod
    def unit_dual_quaternion_to_se3(cls, dq):
        q = dq.real
        t = dq * q.conjugate()
        return Quaternion.unit_quaternion_to_so3(q), np.array([[2 * t.dual.i], [2 * t.dual.j], [2 * t.dual.k]])


class Cliff4:
    def __init__(self, lst):
        self.value = lst
        self.shape = lst.shape[1:]

    @classmethod
    def from_real(cls, real):
        if isinstance(real, np.ndarray) and np.size(real) > 1:
            shape = np.shape(real)
            if np.size(shape) > 1:
                res = np.zeros((16, shape[0], shape[1]))
            else:
                res = np.zeros((16, shape[0]))
        else:
            res = np.zeros(16)
        res[0] = real
        return Cliff4(res)

    def __repr__(self):
        if not self.shape:
            res = '{:.5}'.format(self.value[0])
            names = ['', 'a', 'b', 'c', 'd', 'ab', 'ac', 'ad', 'bc', 'bd', 'cd', 'abc', 'abd', 'acd', 'bcd', 'abcd']
            for i in range(1, 16):
                res += '{:+.5}{}'.format(self.value[i], names[i])
            return res
        else:
            res = np.zeros(self.shape, dtype=Cliff4)
            if np.size(self.shape) == 1:
                for i in range(self.shape[0]):
                    res[i] = Cliff4(self.value[:, i])
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        res[i, j] = Cliff4(self.value[:, i, j])
            return str(res)

    def __add__(self, other):
        if not isinstance(other, Cliff4):
            other = Cliff4.from_real(other)
        return Cliff4(self.value + other.value)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Cliff4(-self.value)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = Cliff4.from_real(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Cliff4):
            return Cliff4(other * self.value)
        return Cliff4(np.array([self.value[0] * other.value[0] - self.value[1] * other.value[1]
                                - self.value[2] * other.value[2] - self.value[3] * other.value[3]
                                - self.value[4] * other.value[4] - self.value[5] * other.value[5]
                                - self.value[6] * other.value[6] - self.value[7] * other.value[7]
                                - self.value[8] * other.value[8] - self.value[9] * other.value[9]
                                - self.value[10] * other.value[10] + self.value[11] * other.value[11]
                                + self.value[12] * other.value[12] + self.value[13] * other.value[13]
                                + self.value[14] * other.value[14] + self.value[15] * other.value[15],
                                self.value[0] * other.value[1] + self.value[1] * other.value[0]
                                + self.value[2] * other.value[5] + self.value[3] * other.value[6]
                                + self.value[4] * other.value[7] - self.value[5] * other.value[2]
                                - self.value[6] * other.value[3] - self.value[7] * other.value[4]
                                - self.value[8] * other.value[11] - self.value[9] * other.value[12]
                                - self.value[10] * other.value[13] - self.value[11] * other.value[8]
                                - self.value[12] * other.value[9] - self.value[13] * other.value[10]
                                - self.value[14] * other.value[15] + self.value[15] * other.value[14],
                                self.value[0] * other.value[2] - self.value[1] * other.value[5]
                                + self.value[2] * other.value[0] + self.value[3] * other.value[8]
                                + self.value[4] * other.value[9] + self.value[5] * other.value[1]
                                + self.value[6] * other.value[11] + self.value[7] * other.value[12]
                                - self.value[8] * other.value[3] - self.value[9] * other.value[4]
                                - self.value[10] * other.value[14] + self.value[11] * other.value[6]
                                + self.value[12] * other.value[7] + self.value[13] * other.value[15]
                                - self.value[14] * other.value[10] - self.value[15] * other.value[13],
                                self.value[0] * other.value[3] - self.value[1] * other.value[6]
                                - self.value[2] * other.value[8] + self.value[3] * other.value[0]
                                + self.value[4] * other.value[10] - self.value[5] * other.value[11]
                                + self.value[6] * other.value[1] + self.value[7] * other.value[13]
                                + self.value[8] * other.value[2] + self.value[9] * other.value[14]
                                - self.value[10] * other.value[4] - self.value[11] * other.value[5]
                                - self.value[12] * other.value[15] + self.value[13] * other.value[7]
                                + self.value[14] * other.value[9] + self.value[15] * other.value[12],
                                self.value[0] * other.value[4] - self.value[1] * other.value[7]
                                - self.value[2] * other.value[9] - self.value[3] * other.value[10]
                                + self.value[4] * other.value[0] - self.value[5] * other.value[12]
                                - self.value[6] * other.value[13] + self.value[7] * other.value[1]
                                - self.value[8] * other.value[14] + self.value[9] * other.value[2]
                                + self.value[10] * other.value[3] + self.value[11] * other.value[15]
                                - self.value[12] * other.value[5] - self.value[13] * other.value[6]
                                - self.value[14] * other.value[8] - self.value[15] * other.value[11],
                                self.value[0] * other.value[5] + self.value[1] * other.value[2]
                                - self.value[2] * other.value[1] - self.value[3] * other.value[11]
                                - self.value[4] * other.value[12] + self.value[5] * other.value[0]
                                + self.value[6] * other.value[8] + self.value[7] * other.value[9]
                                - self.value[8] * other.value[6] - self.value[9] * other.value[7]
                                - self.value[10] * other.value[15] - self.value[11] * other.value[3]
                                - self.value[12] * other.value[4] - self.value[13] * other.value[14]
                                + self.value[14] * other.value[13] - self.value[15] * other.value[10],
                                self.value[0] * other.value[6] + self.value[1] * other.value[3]
                                + self.value[2] * other.value[11] - self.value[3] * other.value[1]
                                - self.value[4] * other.value[13] - self.value[5] * other.value[8]
                                + self.value[6] * other.value[0] + self.value[7] * other.value[10]
                                + self.value[8] * other.value[5] + self.value[9] * other.value[15]
                                - self.value[10] * other.value[7] + self.value[11] * other.value[2]
                                + self.value[12] * other.value[14] - self.value[13] * other.value[4]
                                - self.value[14] * other.value[12] + self.value[15] * other.value[9],
                                self.value[0] * other.value[7] + self.value[1] * other.value[4]
                                + self.value[2] * other.value[12] + self.value[3] * other.value[13]
                                - self.value[4] * other.value[1] - self.value[5] * other.value[9]
                                - self.value[6] * other.value[10] + self.value[7] * other.value[0]
                                - self.value[8] * other.value[15] + self.value[9] * other.value[5]
                                + self.value[10] * other.value[6] - self.value[11] * other.value[14]
                                + self.value[12] * other.value[2] + self.value[13] * other.value[3]
                                + self.value[14] * other.value[11] - self.value[15] * other.value[8],
                                self.value[0] * other.value[8] - self.value[1] * other.value[11]
                                + self.value[2] * other.value[3] - self.value[3] * other.value[2]
                                - self.value[4] * other.value[14] + self.value[5] * other.value[6]
                                - self.value[6] * other.value[5] - self.value[7] * other.value[15]
                                + self.value[8] * other.value[0] + self.value[9] * other.value[10]
                                - self.value[10] * other.value[9] - self.value[11] * other.value[1]
                                - self.value[12] * other.value[13] + self.value[13] * other.value[12]
                                - self.value[14] * other.value[4] - self.value[15] * other.value[7],
                                self.value[0] * other.value[9] - self.value[1] * other.value[12]
                                + self.value[2] * other.value[4] + self.value[3] * other.value[14]
                                - self.value[4] * other.value[2] + self.value[5] * other.value[7]
                                + self.value[6] * other.value[15] - self.value[7] * other.value[5]
                                - self.value[8] * other.value[10] + self.value[9] * other.value[0]
                                + self.value[10] * other.value[8] + self.value[11] * other.value[13]
                                - self.value[12] * other.value[1] - self.value[13] * other.value[11]
                                + self.value[14] * other.value[3] + self.value[15] * other.value[6],
                                self.value[0] * other.value[10] - self.value[1] * other.value[13]
                                - self.value[2] * other.value[14] + self.value[3] * other.value[4]
                                - self.value[4] * other.value[3] - self.value[5] * other.value[15]
                                + self.value[6] * other.value[7] - self.value[7] * other.value[6]
                                + self.value[8] * other.value[9] - self.value[9] * other.value[8]
                                + self.value[10] * other.value[0] - self.value[11] * other.value[12]
                                + self.value[12] * other.value[11] - self.value[13] * other.value[1]
                                - self.value[14] * other.value[2] - self.value[15] * other.value[5],
                                self.value[0] * other.value[11] + self.value[1] * other.value[8]
                                - self.value[2] * other.value[6] + self.value[3] * other.value[5]
                                + self.value[4] * other.value[15] + self.value[5] * other.value[3]
                                - self.value[6] * other.value[2] - self.value[7] * other.value[14]
                                + self.value[8] * other.value[1] + self.value[9] * other.value[13]
                                - self.value[10] * other.value[12] + self.value[11] * other.value[0]
                                + self.value[12] * other.value[10] - self.value[13] * other.value[9]
                                + self.value[14] * other.value[7] - self.value[15] * other.value[4],
                                self.value[0] * other.value[12] + self.value[1] * other.value[9]
                                - self.value[2] * other.value[7] - self.value[3] * other.value[15]
                                + self.value[4] * other.value[5] + self.value[5] * other.value[4]
                                + self.value[6] * other.value[14] - self.value[7] * other.value[2]
                                - self.value[8] * other.value[13] + self.value[9] * other.value[1]
                                + self.value[10] * other.value[11] - self.value[11] * other.value[10]
                                + self.value[12] * other.value[0] + self.value[13] * other.value[8]
                                - self.value[14] * other.value[6] + self.value[15] * other.value[3],
                                self.value[0] * other.value[13] + self.value[1] * other.value[10]
                                + self.value[2] * other.value[15] - self.value[3] * other.value[7]
                                + self.value[4] * other.value[6] - self.value[5] * other.value[14]
                                + self.value[6] * other.value[4] - self.value[7] * other.value[3]
                                + self.value[8] * other.value[12] - self.value[9] * other.value[11]
                                + self.value[10] * other.value[1] + self.value[11] * other.value[9]
                                - self.value[12] * other.value[8] + self.value[13] * other.value[0]
                                + self.value[14] * other.value[5] - self.value[15] * other.value[2],
                                self.value[0] * other.value[14] - self.value[1] * other.value[15]
                                + self.value[2] * other.value[10] - self.value[3] * other.value[9]
                                + self.value[4] * other.value[8] + self.value[5] * other.value[13]
                                - self.value[6] * other.value[12] + self.value[7] * other.value[11]
                                + self.value[8] * other.value[4] - self.value[9] * other.value[3]
                                + self.value[10] * other.value[2] - self.value[11] * other.value[7]
                                + self.value[12] * other.value[6] - self.value[13] * other.value[5]
                                + self.value[14] * other.value[0] + self.value[15] * other.value[1],
                                self.value[0] * other.value[15] + self.value[1] * other.value[14]
                                - self.value[2] * other.value[13] + self.value[3] * other.value[12]
                                - self.value[4] * other.value[11] + self.value[5] * other.value[10]
                                - self.value[6] * other.value[9] + self.value[7] * other.value[8]
                                + self.value[8] * other.value[7] - self.value[9] * other.value[6]
                                + self.value[10] * other.value[5] + self.value[11] * other.value[4]
                                - self.value[12] * other.value[3] + self.value[13] * other.value[2]
                                - self.value[14] * other.value[1] + self.value[15] * other.value[0]]))

    def __rmul__(self, other):
        other = Cliff4.from_real(other)
        return other * self

    def __matmul__(self, other):
        if not isinstance(other, Cliff4):
            other = Cliff4.from_real(other)
        return Cliff4(np.array([self.value[0] @ other.value[0] - self.value[1] @ other.value[1]
                                - self.value[2] @ other.value[2] - self.value[3] @ other.value[3]
                                - self.value[4] @ other.value[4] - self.value[5] @ other.value[5]
                                - self.value[6] @ other.value[6] - self.value[7] @ other.value[7]
                                - self.value[8] @ other.value[8] - self.value[9] @ other.value[9]
                                - self.value[10] @ other.value[10] + self.value[11] @ other.value[11]
                                + self.value[12] @ other.value[12] + self.value[13] @ other.value[13]
                                + self.value[14] @ other.value[14] + self.value[15] @ other.value[15],
                                self.value[0] @ other.value[1] + self.value[1] @ other.value[0]
                                + self.value[2] @ other.value[5] + self.value[3] @ other.value[6]
                                + self.value[4] @ other.value[7] - self.value[5] @ other.value[2]
                                - self.value[6] @ other.value[3] - self.value[7] @ other.value[4]
                                - self.value[8] @ other.value[11] - self.value[9] @ other.value[12]
                                - self.value[10] @ other.value[13] - self.value[11] @ other.value[8]
                                - self.value[12] @ other.value[9] - self.value[13] @ other.value[10]
                                - self.value[14] @ other.value[15] + self.value[15] @ other.value[14],
                                self.value[0] @ other.value[2] - self.value[1] @ other.value[5]
                                + self.value[2] @ other.value[0] + self.value[3] @ other.value[8]
                                + self.value[4] @ other.value[9] + self.value[5] @ other.value[1]
                                + self.value[6] @ other.value[11] + self.value[7] @ other.value[12]
                                - self.value[8] @ other.value[3] - self.value[9] @ other.value[4]
                                - self.value[10] @ other.value[14] + self.value[11] @ other.value[6]
                                + self.value[12] @ other.value[7] + self.value[13] @ other.value[15]
                                - self.value[14] @ other.value[10] - self.value[15] @ other.value[13],
                                self.value[0] @ other.value[3] - self.value[1] @ other.value[6]
                                - self.value[2] @ other.value[8] + self.value[3] @ other.value[0]
                                + self.value[4] @ other.value[10] - self.value[5] @ other.value[11]
                                + self.value[6] @ other.value[1] + self.value[7] @ other.value[13]
                                + self.value[8] @ other.value[2] + self.value[9] @ other.value[14]
                                - self.value[10] @ other.value[4] - self.value[11] @ other.value[5]
                                - self.value[12] @ other.value[15] + self.value[13] @ other.value[7]
                                + self.value[14] @ other.value[9] + self.value[15] @ other.value[12],
                                self.value[0] @ other.value[4] - self.value[1] @ other.value[7]
                                - self.value[2] @ other.value[9] - self.value[3] @ other.value[10]
                                + self.value[4] @ other.value[0] - self.value[5] @ other.value[12]
                                - self.value[6] @ other.value[13] + self.value[7] @ other.value[1]
                                - self.value[8] @ other.value[14] + self.value[9] @ other.value[2]
                                + self.value[10] @ other.value[3] + self.value[11] @ other.value[15]
                                - self.value[12] @ other.value[5] - self.value[13] @ other.value[6]
                                - self.value[14] @ other.value[8] - self.value[15] @ other.value[11],
                                self.value[0] @ other.value[5] + self.value[1] @ other.value[2]
                                - self.value[2] @ other.value[1] - self.value[3] @ other.value[11]
                                - self.value[4] @ other.value[12] + self.value[5] @ other.value[0]
                                + self.value[6] @ other.value[8] + self.value[7] @ other.value[9]
                                - self.value[8] @ other.value[6] - self.value[9] @ other.value[7]
                                - self.value[10] @ other.value[15] - self.value[11] @ other.value[3]
                                - self.value[12] @ other.value[4] - self.value[13] @ other.value[14]
                                + self.value[14] @ other.value[13] - self.value[15] @ other.value[10],
                                self.value[0] @ other.value[6] + self.value[1] @ other.value[3]
                                + self.value[2] @ other.value[11] - self.value[3] @ other.value[1]
                                - self.value[4] @ other.value[13] - self.value[5] @ other.value[8]
                                + self.value[6] @ other.value[0] + self.value[7] @ other.value[10]
                                + self.value[8] @ other.value[5] + self.value[9] @ other.value[15]
                                - self.value[10] @ other.value[7] + self.value[11] @ other.value[2]
                                + self.value[12] @ other.value[14] - self.value[13] @ other.value[4]
                                - self.value[14] @ other.value[12] + self.value[15] @ other.value[9],
                                self.value[0] @ other.value[7] + self.value[1] @ other.value[4]
                                + self.value[2] @ other.value[12] + self.value[3] @ other.value[13]
                                - self.value[4] @ other.value[1] - self.value[5] @ other.value[9]
                                - self.value[6] @ other.value[10] + self.value[7] @ other.value[0]
                                - self.value[8] @ other.value[15] + self.value[9] @ other.value[5]
                                + self.value[10] @ other.value[6] - self.value[11] @ other.value[14]
                                + self.value[12] @ other.value[2] + self.value[13] @ other.value[3]
                                + self.value[14] @ other.value[11] - self.value[15] @ other.value[8],
                                self.value[0] @ other.value[8] - self.value[1] @ other.value[11]
                                + self.value[2] @ other.value[3] - self.value[3] @ other.value[2]
                                - self.value[4] @ other.value[14] + self.value[5] @ other.value[6]
                                - self.value[6] @ other.value[5] - self.value[7] @ other.value[15]
                                + self.value[8] @ other.value[0] + self.value[9] @ other.value[10]
                                - self.value[10] @ other.value[9] - self.value[11] @ other.value[1]
                                - self.value[12] @ other.value[13] + self.value[13] @ other.value[12]
                                - self.value[14] @ other.value[4] - self.value[15] @ other.value[7],
                                self.value[0] @ other.value[9] - self.value[1] @ other.value[12]
                                + self.value[2] @ other.value[4] + self.value[3] @ other.value[14]
                                - self.value[4] @ other.value[2] + self.value[5] @ other.value[7]
                                + self.value[6] @ other.value[15] - self.value[7] @ other.value[5]
                                - self.value[8] @ other.value[10] + self.value[9] @ other.value[0]
                                + self.value[10] @ other.value[8] + self.value[11] @ other.value[13]
                                - self.value[12] @ other.value[1] - self.value[13] @ other.value[11]
                                + self.value[14] @ other.value[3] + self.value[15] @ other.value[6],
                                self.value[0] @ other.value[10] - self.value[1] @ other.value[13]
                                - self.value[2] @ other.value[14] + self.value[3] @ other.value[4]
                                - self.value[4] @ other.value[3] - self.value[5] @ other.value[15]
                                + self.value[6] @ other.value[7] - self.value[7] @ other.value[6]
                                + self.value[8] @ other.value[9] - self.value[9] @ other.value[8]
                                + self.value[10] @ other.value[0] - self.value[11] @ other.value[12]
                                + self.value[12] @ other.value[11] - self.value[13] @ other.value[1]
                                - self.value[14] @ other.value[2] - self.value[15] @ other.value[5],
                                self.value[0] @ other.value[11] + self.value[1] @ other.value[8]
                                - self.value[2] @ other.value[6] + self.value[3] @ other.value[5]
                                + self.value[4] @ other.value[15] + self.value[5] @ other.value[3]
                                - self.value[6] @ other.value[2] - self.value[7] @ other.value[14]
                                + self.value[8] @ other.value[1] + self.value[9] @ other.value[13]
                                - self.value[10] @ other.value[12] + self.value[11] @ other.value[0]
                                + self.value[12] @ other.value[10] - self.value[13] @ other.value[9]
                                + self.value[14] @ other.value[7] - self.value[15] @ other.value[4],
                                self.value[0] @ other.value[12] + self.value[1] @ other.value[9]
                                - self.value[2] @ other.value[7] - self.value[3] @ other.value[15]
                                + self.value[4] @ other.value[5] + self.value[5] @ other.value[4]
                                + self.value[6] @ other.value[14] - self.value[7] @ other.value[2]
                                - self.value[8] @ other.value[13] + self.value[9] @ other.value[1]
                                + self.value[10] @ other.value[11] - self.value[11] @ other.value[10]
                                + self.value[12] @ other.value[0] + self.value[13] @ other.value[8]
                                - self.value[14] @ other.value[6] + self.value[15] @ other.value[3],
                                self.value[0] @ other.value[13] + self.value[1] @ other.value[10]
                                + self.value[2] @ other.value[15] - self.value[3] @ other.value[7]
                                + self.value[4] @ other.value[6] - self.value[5] @ other.value[14]
                                + self.value[6] @ other.value[4] - self.value[7] @ other.value[3]
                                + self.value[8] @ other.value[12] - self.value[9] @ other.value[11]
                                + self.value[10] @ other.value[1] + self.value[11] @ other.value[9]
                                - self.value[12] @ other.value[8] + self.value[13] @ other.value[0]
                                + self.value[14] @ other.value[5] - self.value[15] @ other.value[2],
                                self.value[0] @ other.value[14] - self.value[1] @ other.value[15]
                                + self.value[2] @ other.value[10] - self.value[3] @ other.value[9]
                                + self.value[4] @ other.value[8] + self.value[5] @ other.value[13]
                                - self.value[6] @ other.value[12] + self.value[7] @ other.value[11]
                                + self.value[8] @ other.value[4] - self.value[9] @ other.value[3]
                                + self.value[10] @ other.value[2] - self.value[11] @ other.value[7]
                                + self.value[12] @ other.value[6] - self.value[13] @ other.value[5]
                                + self.value[14] @ other.value[0] + self.value[15] @ other.value[1],
                                self.value[0] @ other.value[15] + self.value[1] @ other.value[14]
                                - self.value[2] @ other.value[13] + self.value[3] @ other.value[12]
                                - self.value[4] @ other.value[11] + self.value[5] @ other.value[10]
                                - self.value[6] @ other.value[9] + self.value[7] @ other.value[8]
                                + self.value[8] @ other.value[7] - self.value[9] @ other.value[6]
                                + self.value[10] @ other.value[5] + self.value[11] @ other.value[4]
                                - self.value[12] @ other.value[3] + self.value[13] @ other.value[2]
                                - self.value[14] @ other.value[1] + self.value[15] @ other.value[0]]))

    def __rmatmul__(self, other):
        other = Cliff4.from_real(other)
        return other @ self

    def outer(self, other):
        if not isinstance(other, Cliff4):
            other = Cliff4.from_real(other)
        return Cliff4(np.array([np.outer(self.value[0], other.value[0]) - np.outer(self.value[1], other.value[1])
                                - np.outer(self.value[2], other.value[2]) - np.outer(self.value[3], other.value[3])
                                - np.outer(self.value[4], other.value[4]) - np.outer(self.value[5], other.value[5])
                                - np.outer(self.value[6], other.value[6]) - np.outer(self.value[7], other.value[7])
                                - np.outer(self.value[8], other.value[8]) - np.outer(self.value[9], other.value[9])
                                - np.outer(self.value[10], other.value[10]) + np.outer(self.value[11], other.value[11])
                                + np.outer(self.value[12], other.value[12]) + np.outer(self.value[13], other.value[13])
                                + np.outer(self.value[14], other.value[14]) + np.outer(self.value[15], other.value[15]),
                                np.outer(self.value[0], other.value[1]) + np.outer(self.value[1], other.value[0])
                                + np.outer(self.value[2], other.value[5]) + np.outer(self.value[3], other.value[6])
                                + np.outer(self.value[4], other.value[7]) - np.outer(self.value[5], other.value[2])
                                - np.outer(self.value[6], other.value[3]) - np.outer(self.value[7], other.value[4])
                                - np.outer(self.value[8], other.value[11]) - np.outer(self.value[9], other.value[12])
                                - np.outer(self.value[10], other.value[13]) - np.outer(self.value[11], other.value[8])
                                - np.outer(self.value[12], other.value[9]) - np.outer(self.value[13], other.value[10])
                                - np.outer(self.value[14], other.value[15]) + np.outer(self.value[15], other.value[14]),
                                np.outer(self.value[0], other.value[2]) - np.outer(self.value[1], other.value[5])
                                + np.outer(self.value[2], other.value[0]) + np.outer(self.value[3], other.value[8])
                                + np.outer(self.value[4], other.value[9]) + np.outer(self.value[5], other.value[1])
                                + np.outer(self.value[6], other.value[11]) + np.outer(self.value[7], other.value[12])
                                - np.outer(self.value[8], other.value[3]) - np.outer(self.value[9], other.value[4])
                                - np.outer(self.value[10], other.value[14]) + np.outer(self.value[11], other.value[6])
                                + np.outer(self.value[12], other.value[7]) + np.outer(self.value[13], other.value[15])
                                - np.outer(self.value[14], other.value[10]) - np.outer(self.value[15], other.value[13]),
                                np.outer(self.value[0], other.value[3]) - np.outer(self.value[1], other.value[6])
                                - np.outer(self.value[2], other.value[8]) + np.outer(self.value[3], other.value[0])
                                + np.outer(self.value[4], other.value[10]) - np.outer(self.value[5], other.value[11])
                                + np.outer(self.value[6], other.value[1]) + np.outer(self.value[7], other.value[13])
                                + np.outer(self.value[8], other.value[2]) + np.outer(self.value[9], other.value[14])
                                - np.outer(self.value[10], other.value[4]) - np.outer(self.value[11], other.value[5])
                                - np.outer(self.value[12], other.value[15]) + np.outer(self.value[13], other.value[7])
                                + np.outer(self.value[14], other.value[9]) + np.outer(self.value[15], other.value[12]),
                                np.outer(self.value[0], other.value[4]) - np.outer(self.value[1], other.value[7])
                                - np.outer(self.value[2], other.value[9]) - np.outer(self.value[3], other.value[10])
                                + np.outer(self.value[4], other.value[0]) - np.outer(self.value[5], other.value[12])
                                - np.outer(self.value[6], other.value[13]) + np.outer(self.value[7], other.value[1])
                                - np.outer(self.value[8], other.value[14]) + np.outer(self.value[9], other.value[2])
                                + np.outer(self.value[10], other.value[3]) + np.outer(self.value[11], other.value[15])
                                - np.outer(self.value[12], other.value[5]) - np.outer(self.value[13], other.value[6])
                                - np.outer(self.value[14], other.value[8]) - np.outer(self.value[15], other.value[11]),
                                np.outer(self.value[0], other.value[5]) + np.outer(self.value[1], other.value[2])
                                - np.outer(self.value[2], other.value[1]) - np.outer(self.value[3], other.value[11])
                                - np.outer(self.value[4], other.value[12]) + np.outer(self.value[5], other.value[0])
                                + np.outer(self.value[6], other.value[8]) + np.outer(self.value[7], other.value[9])
                                - np.outer(self.value[8], other.value[6]) - np.outer(self.value[9], other.value[7])
                                - np.outer(self.value[10], other.value[15]) - np.outer(self.value[11], other.value[3])
                                - np.outer(self.value[12], other.value[4]) - np.outer(self.value[13], other.value[14])
                                + np.outer(self.value[14], other.value[13]) - np.outer(self.value[15], other.value[10]),
                                np.outer(self.value[0], other.value[6]) + np.outer(self.value[1], other.value[3])
                                + np.outer(self.value[2], other.value[11]) - np.outer(self.value[3], other.value[1])
                                - np.outer(self.value[4], other.value[13]) - np.outer(self.value[5], other.value[8])
                                + np.outer(self.value[6], other.value[0]) + np.outer(self.value[7], other.value[10])
                                + np.outer(self.value[8], other.value[5]) + np.outer(self.value[9], other.value[15])
                                - np.outer(self.value[10], other.value[7]) + np.outer(self.value[11], other.value[2])
                                + np.outer(self.value[12], other.value[14]) - np.outer(self.value[13], other.value[4])
                                - np.outer(self.value[14], other.value[12]) + np.outer(self.value[15], other.value[9]),
                                np.outer(self.value[0], other.value[7]) + np.outer(self.value[1], other.value[4])
                                + np.outer(self.value[2], other.value[12]) + np.outer(self.value[3], other.value[13])
                                - np.outer(self.value[4], other.value[1]) - np.outer(self.value[5], other.value[9])
                                - np.outer(self.value[6], other.value[10]) + np.outer(self.value[7], other.value[0])
                                - np.outer(self.value[8], other.value[15]) + np.outer(self.value[9], other.value[5])
                                + np.outer(self.value[10], other.value[6]) - np.outer(self.value[11], other.value[14])
                                + np.outer(self.value[12], other.value[2]) + np.outer(self.value[13], other.value[3])
                                + np.outer(self.value[14], other.value[11]) - np.outer(self.value[15], other.value[8]),
                                np.outer(self.value[0], other.value[8]) - np.outer(self.value[1], other.value[11])
                                + np.outer(self.value[2], other.value[3]) - np.outer(self.value[3], other.value[2])
                                - np.outer(self.value[4], other.value[14]) + np.outer(self.value[5], other.value[6])
                                - np.outer(self.value[6], other.value[5]) - np.outer(self.value[7], other.value[15])
                                + np.outer(self.value[8], other.value[0]) + np.outer(self.value[9], other.value[10])
                                - np.outer(self.value[10], other.value[9]) - np.outer(self.value[11], other.value[1])
                                - np.outer(self.value[12], other.value[13]) + np.outer(self.value[13], other.value[12])
                                - np.outer(self.value[14], other.value[4]) - np.outer(self.value[15], other.value[7]),
                                np.outer(self.value[0], other.value[9]) - np.outer(self.value[1], other.value[12])
                                + np.outer(self.value[2], other.value[4]) + np.outer(self.value[3], other.value[14])
                                - np.outer(self.value[4], other.value[2]) + np.outer(self.value[5], other.value[7])
                                + np.outer(self.value[6], other.value[15]) - np.outer(self.value[7], other.value[5])
                                - np.outer(self.value[8], other.value[10]) + np.outer(self.value[9], other.value[0])
                                + np.outer(self.value[10], other.value[8]) + np.outer(self.value[11], other.value[13])
                                - np.outer(self.value[12], other.value[1]) - np.outer(self.value[13], other.value[11])
                                + np.outer(self.value[14], other.value[3]) + np.outer(self.value[15], other.value[6]),
                                np.outer(self.value[0], other.value[10]) - np.outer(self.value[1], other.value[13])
                                - np.outer(self.value[2], other.value[14]) + np.outer(self.value[3], other.value[4])
                                - np.outer(self.value[4], other.value[3]) - np.outer(self.value[5], other.value[15])
                                + np.outer(self.value[6], other.value[7]) - np.outer(self.value[7], other.value[6])
                                + np.outer(self.value[8], other.value[9]) - np.outer(self.value[9], other.value[8])
                                + np.outer(self.value[10], other.value[0]) - np.outer(self.value[11], other.value[12])
                                + np.outer(self.value[12], other.value[11]) - np.outer(self.value[13], other.value[1])
                                - np.outer(self.value[14], other.value[2]) - np.outer(self.value[15], other.value[5]),
                                np.outer(self.value[0], other.value[11]) + np.outer(self.value[1], other.value[8])
                                - np.outer(self.value[2], other.value[6]) + np.outer(self.value[3], other.value[5])
                                + np.outer(self.value[4], other.value[15]) + np.outer(self.value[5], other.value[3])
                                - np.outer(self.value[6], other.value[2]) - np.outer(self.value[7], other.value[14])
                                + np.outer(self.value[8], other.value[1]) + np.outer(self.value[9], other.value[13])
                                - np.outer(self.value[10], other.value[12]) + np.outer(self.value[11], other.value[0])
                                + np.outer(self.value[12], other.value[10]) - np.outer(self.value[13], other.value[9])
                                + np.outer(self.value[14], other.value[7]) - np.outer(self.value[15], other.value[4]),
                                np.outer(self.value[0], other.value[12]) + np.outer(self.value[1], other.value[9])
                                - np.outer(self.value[2], other.value[7]) - np.outer(self.value[3], other.value[15])
                                + np.outer(self.value[4], other.value[5]) + np.outer(self.value[5], other.value[4])
                                + np.outer(self.value[6], other.value[14]) - np.outer(self.value[7], other.value[2])
                                - np.outer(self.value[8], other.value[13]) + np.outer(self.value[9], other.value[1])
                                + np.outer(self.value[10], other.value[11]) - np.outer(self.value[11], other.value[10])
                                + np.outer(self.value[12], other.value[0]) + np.outer(self.value[13], other.value[8])
                                - np.outer(self.value[14], other.value[6]) + np.outer(self.value[15], other.value[3]),
                                np.outer(self.value[0], other.value[13]) + np.outer(self.value[1], other.value[10])
                                + np.outer(self.value[2], other.value[15]) - np.outer(self.value[3], other.value[7])
                                + np.outer(self.value[4], other.value[6]) - np.outer(self.value[5], other.value[14])
                                + np.outer(self.value[6], other.value[4]) - np.outer(self.value[7], other.value[3])
                                + np.outer(self.value[8], other.value[12]) - np.outer(self.value[9], other.value[11])
                                + np.outer(self.value[10], other.value[1]) + np.outer(self.value[11], other.value[9])
                                - np.outer(self.value[12], other.value[8]) + np.outer(self.value[13], other.value[0])
                                + np.outer(self.value[14], other.value[5]) - np.outer(self.value[15], other.value[2]),
                                np.outer(self.value[0], other.value[14]) - np.outer(self.value[1], other.value[15])
                                + np.outer(self.value[2], other.value[10]) - np.outer(self.value[3], other.value[9])
                                + np.outer(self.value[4], other.value[8]) + np.outer(self.value[5], other.value[13])
                                - np.outer(self.value[6], other.value[12]) + np.outer(self.value[7], other.value[11])
                                + np.outer(self.value[8], other.value[4]) - np.outer(self.value[9], other.value[3])
                                + np.outer(self.value[10], other.value[2]) - np.outer(self.value[11], other.value[7])
                                + np.outer(self.value[12], other.value[6]) - np.outer(self.value[13], other.value[5])
                                + np.outer(self.value[14], other.value[0]) + np.outer(self.value[15], other.value[1]),
                                np.outer(self.value[0], other.value[15]) + np.outer(self.value[1], other.value[14])
                                - np.outer(self.value[2], other.value[13]) + np.outer(self.value[3], other.value[12])
                                - np.outer(self.value[4], other.value[11]) + np.outer(self.value[5], other.value[10])
                                - np.outer(self.value[6], other.value[9]) + np.outer(self.value[7], other.value[8])
                                + np.outer(self.value[8], other.value[7]) - np.outer(self.value[9], other.value[6])
                                + np.outer(self.value[10], other.value[5]) + np.outer(self.value[11], other.value[4])
                                - np.outer(self.value[12], other.value[3]) + np.outer(self.value[13], other.value[2])
                                - np.outer(self.value[14], other.value[1]) + np.outer(self.value[15], other.value[0])]))

    def involution(self):
        return Cliff4(np.array([self.value[0], -self.value[1], -self.value[2], -self.value[3], -self.value[4],
                                self.value[5], self.value[6], self.value[7], self.value[8], self.value[9],
                                self.value[10], -self.value[11], -self.value[12], -self.value[13], -self.value[14],
                                self.value[15]]))

    def reverse(self):
        return Cliff4(np.array([self.value[0], self.value[1], self.value[2], self.value[3], self.value[4],
                                -self.value[5], -self.value[6], -self.value[7], -self.value[8], -self.value[9],
                                -self.value[10], -self.value[11], -self.value[12], -self.value[13], -self.value[14],
                                self.value[15]]))

    def conjugate(self):
        return self.involution().reverse()

    def psaudo_conjugate(self):
        return Cliff4(np.array([self.value[0], -self.value[1], -self.value[2], -self.value[3], -self.value[4],
                                -self.value[5], -self.value[6], -self.value[7], -self.value[8], -self.value[9],
                                -self.value[10], -self.value[11], -self.value[12], -self.value[13], -self.value[14],
                                -self.value[15]]))

    def magnitude_sqr(self):
        return Cliff4.conjugate(self) * self

    def sqrt(self):
        if self.value[0] < np.sqrt(np.sum(self.value[11:] ** 2)):
            raise ValueError('The square root does not exist')
        s = np.sqrt((self.value[0] + np.sqrt(self.value[0] ** 2 - np.sum(self.value[11:] ** 2))) / 2)
        z = np.zeros_like(s)
        return Cliff4(np.array([s, z, z, z, z, z, z, z, z, z, z, self.value[11] / (2 * s), self.value[12] / (2 * s),
                                self.value[13] / (2 * s), self.value[14] / (2 * s), self.value[15] / (2 * s)]))

    def magnitude(self, mode=None):
        if mode == 'F':
            return np.linalg.norm(self.value)
        return self.magnitude_sqr().sqrt()

    def norm(self, mode=None):
        if mode == 'F':
            return np.linalg.norm(self.value)
        return (self.conjugate() @ self).sqrt()

    def __eq__(self, other):
        if not isinstance(other, Cliff4):
            other = Cliff4.from_real(other)
        return np.all(self.value == other.value)

    def inv(self):
        if self == 0:
            raise ZeroDivisionError
        mag = self.magnitude_sqr()
        conj = mag.psaudo_conjugate()
        norm = conj * mag
        if norm.value[0] <= 1e-12:
            raise ValueError('Zero divisors are not invertible')
        return (1 / norm.value[0]) * conj * self.conjugate()

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError
        if not isinstance(other, Cliff4):
            return Cliff4(self.value / other)
        return self * other.inv()

    def __rtruediv__(self, other):
        other = Cliff4.from_real(other)
        return other/self

    def __pow__(self, power, modulo=None):
        if power >= 0:
            res = Cliff4.from_real(np.ones_like(self.value[0]))
            for i in range(power):
                res *= self
            return res
        else:
            res = Cliff4.inv(self)
            return res ** power

    def __getitem__(self, item):
        return Cliff4(self.value[..., item])

    def transpose(self):
        return Cliff4(np.swapaxes(self.value, 1, 2))

    def concatenate(self, other, axis):
        return Cliff4(np.concatenate([self.value, other.value], axis=axis))

    @classmethod
    def diagonal(cls, lst):
        res = np.zeros((16, len(lst), len(lst)))
        for i in range(len(lst)):
            res[:, i, i] = lst[i].value
        return Cliff4(res)

    @classmethod
    def unit_to_so4(cls, elem):
        a = elem.value[0]
        b = elem.value[5]
        c = elem.value[6]
        d = elem.value[7]
        e = elem.value[8]
        f = elem.value[9]
        g = elem.value[10]
        h = elem.value[15]
        return np.array([[a**2-b**2-c**2-d**2+e**2+f**2+g**2-h**2, -2*a*b-2*c*e-2*d*f-2*g*h,
                          -2*a*c+2*b*e-2*d*g+2*f*h, -2*a*d+2*b*f+2*c*g-2*e*h],
                         [2*a*b-2*c*e-2*d*f+2*g*h, a**2-b**2+c**2+d**2-e**2-f**2+g**2-h**2,
                          -2*a*e-2*b*c-2*d*h-2*f*g, -2*a*f-2*b*d+2*c*h+2*e*g],
                         [2*a*c+2*b*e-2*d*g-2*f*h, 2*a*e-2*b*c+2*d*h-2*f*g,
                          a**2+b**2-c**2+d**2-e**2+f**2-g**2-h**2, -2*a*g-2*b*h-2*c*d-2*e*f],
                         [2*a*d+2*b*f+2*c*g+2*e*h, 2*a*f-2*b*d-2*c*h+2*e*g,
                          2*a*g+2*b*h-2*c*d-2*e*f, a**2+b**2+c**2-d**2+e**2-f**2-g**2-h**2]])

    @classmethod
    def so4_to_unit(cls, mat):
        m = np.array([[mat[0, 0] + mat[1, 1] + mat[2, 2] + mat[3, 3], mat[1, 0] - mat[0, 1] - mat[3, 2] + mat[2, 3],
                       mat[2, 0] + mat[3, 1] - mat[0, 2] - mat[1, 3], mat[3, 0] - mat[2, 1] + mat[1, 2] - mat[0, 3]],
                      [mat[1, 0] - mat[0, 1] + mat[3, 2] - mat[2, 3], -mat[0, 0] - mat[1, 1] + mat[2, 2] + mat[3, 3],
                       mat[3, 0] - mat[2, 1] - mat[1, 2] + mat[0, 3], -mat[2, 0] - mat[3, 1] - mat[0, 2] - mat[1, 3]],
                      [mat[2, 0] - mat[3, 1] - mat[0, 2] + mat[1, 3], -mat[3, 0] - mat[2, 1] - mat[1, 2] - mat[0, 3],
                       -mat[0, 0] + mat[1, 1] - mat[2, 2] + mat[3, 3], mat[1, 0] + mat[0, 1] - mat[3, 2] - mat[2, 3]],
                      [mat[3, 0] + mat[2, 1] - mat[1, 2] - mat[0, 3], mat[2, 0] - mat[3, 1] + mat[0, 2] - mat[1, 3],
                       -mat[1, 0] - mat[0, 1] - mat[3, 2] - mat[2, 3], -mat[0, 0] + mat[1, 1] + mat[2, 2] - mat[3, 3]]])
        right = np.linalg.norm(m, axis=0) / 4
        col = np.argmax(right)
        left = m[:, col] / (4 * right[col])
        row = np.argmax(np.abs(left))
        for column in range(4):
            if column == col:
                continue
            else:
                right[column] = m[row, column] / (4 * left[row])
        p, q, r, s = right
        a, b, c, d = left
        return Cliff4(1 / 2 * np.array([1 + a, 0, 0, 0, 0, b, c, d, d, -c, b, 0, 0, 0, 0, 1 - a])) * \
               Cliff4(1 / 2 * np.array([1 + p, 0, 0, 0, 0, q, r, s, -s, r, -q, 0, 0, 0, 0, p - 1]))


class DualCliff4:
    def __init__(self, real, dual):
        if np.shape(real) != np.shape(dual):
            raise ValueError('All parts must be of the same shape')
        self.real = Cliff4(real)
        self.dual = Cliff4(dual)
        self.shape = self.real.shape

    @classmethod
    def from_clifs(cls, real, dual):
        return DualCliff4(real.value, dual.value)

    @classmethod
    def from_clif(cls, real):
        dual = np.zeros_like(real.value)
        return DualCliff4(real.value, dual)

    @classmethod
    def from_dual(cls, dual):
        return DualCliff4.from_clifs(Cliff4.from_real(dual.real), Cliff4.from_real(dual.dual))

    @classmethod
    def from_real(cls, real):
        return DualCliff4.from_clif(Cliff4.from_real(real))

    def __repr__(self):
        if not self.shape:
            return '{}+({})e'.format(self.real, self.dual)
        else:
            res = np.zeros(self.shape, dtype=DualCliff4)
            if np.size(self.shape) == 1:
                for i in range(self.shape[0]):
                    res[i] = DualCliff4(self.real.value[:, i], self.dual.value[:, i])
            else:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        res[i, j] = DualCliff4(self.real.value[:, i, j], self.dual.value[:, i, j])
            return str(res)

    def __add__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                return self + DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                return self + DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return DualCliff4.from_clifs(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return DualCliff4.from_clifs(-self.real, -self.dual)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return DualCliff4.from_clifs(self.real * other.real, self.real * other.dual + self.dual * other.real.involution())

    def __rmul__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return other * self

    def __matmul__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return DualCliff4.from_clifs(self.real @ other.real, self.real @ other.dual + self.dual @ other.real.involution())

    def __rmatmul__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return other @ self

    def outer(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return DualCliff4.from_clifs(Cliff4.outer(self.real, other.real),
                                     Cliff4.outer(self.real, other.dual) + Cliff4.outer(self.dual, other.real.involution()))

    def involution(self):
        return DualCliff4.from_clifs(self.real.involution(), -self.dual.involution())

    def reverse(self):
        return DualCliff4.from_clifs(self.real.reverse(), self.dual.conjugate())

    def conjugate(self):
        return self.involution().reverse()

    def magnitude_sqr(self):
        return self.conjugate() * self

    def sqrt(self):
        sqrt = self.real.sqrt()
        c = sqrt.value[np.array([0, 11, 12, 13, 14, 15])]
        delta = c[0] ** 2 - np.sum(c[1:] ** 2)
        d = np.zeros(16)
        mat = np.array([[c[0] ** 2 - c[1] ** 2 - c[2] ** 2, -c[2] * c[3], c[1] * c[3], -c[2] * c[4], c[1] * c[4],
                         c[0] * c[5], -c[2] * c[5], c[1] * c[5], -c[0] * c[4], c[0] * c[3]],
                        [-c[2] * c[3], c[0] ** 2 - c[1] ** 2 - c[3] ** 2, -c[1] * c[2], -c[3] * c[4], -c[0] * c[5],
                         c[1] * c[4], -c[3] * c[5], c[0] * c[4], c[1] * c[5], -c[0] * c[2]],
                        [c[1] * c[3], -c[1] * c[2], c[0] ** 2 - c[2] ** 2 - c[3] ** 2, c[0] * c[5], -c[3] * c[4],
                         c[2] * c[4], -c[0] * c[4], -c[3] * c[5], c[2] * c[5], c[0] * c[1]],
                        [-c[2] * c[4], -c[3] * c[4], c[0] * c[5], c[0] ** 2 - c[1] ** 2 - c[4] ** 2, -c[1] * c[2],
                         -c[1] * c[3], -c[4] * c[5], -c[0] * c[3], c[0] * c[2], c[1] * c[5]],
                        [c[1] * c[4], -c[0] * c[5], -c[3] * c[4], -c[1] * c[2], c[0] ** 2 - c[2] ** 2 - c[4] ** 2,
                         -c[2] * c[3], c[0] * c[3], -c[4] * c[5], -c[0] * c[1], c[2] * c[5]],
                        [c[0] * c[5], c[1] * c[4], c[2] * c[4], -c[1] * c[3], -c[2] * c[3],
                         c[0] ** 2 - c[3] ** 2 - c[4] ** 2, -c[0] * c[2], c[0] * c[1], -c[4] * c[5], c[3] * c[5]],
                        [-c[2] * c[5], -c[3] * c[5], -c[0] * c[4], -c[4] * c[5], c[0] * c[3], -c[0] * c[2],
                         c[0] ** 2 - c[1] ** 2 - c[5] ** 2, -c[1] * c[2], -c[1] * c[3], -c[1] * c[4]],
                        [c[1] * c[5], c[0] * c[4], -c[3] * c[5], -c[0] * c[3], -c[4] * c[5], c[0] * c[1], -c[1] * c[2],
                         c[0] ** 2 - c[2] ** 2 - c[5] ** 2, -c[2] * c[3], -c[2] * c[4]],
                        [-c[0] * c[4], c[1] * c[5], c[2] * c[5], c[0] * c[2], -c[0] * c[1], -c[4] * c[5], -c[1] * c[3],
                         -c[2] * c[3], c[0] ** 2 - c[3] ** 2 - c[5] ** 2, -c[3] * c[4]],
                        [c[0] * c[3], -c[0] * c[2], c[0] * c[1], c[1] * c[5], c[2] * c[5], c[3] * c[5], -c[1] * c[4],
                         -c[2] * c[4], -c[3] * c[4], c[0] ** 2 - c[4] ** 2 - c[5] ** 2]])
        d[5:15] = ((1 / (2 * c[0] * delta) * mat)[:, :] @ self.dual.value[5:15])
        return DualCliff4.from_clifs(sqrt, Cliff4(d))

    def magnitude(self, mode=None):
        if mode == 'F':
            return np.sqrt(np.sum(self.real.value ** 2 + self.dual.value ** 2))
        return self.magnitude_sqr().sqrt()

    def norm(self, mode=None):
        if mode == 'F':
            return np.sqrt(np.sum(self.real.value ** 2 + self.dual.value ** 2))
        return (self.conjugate() @ self).sqrt()

    def __eq__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return (self.real == other.real) and (self.dual == other.dual)

    def inv(self):
        if self == 0:
            raise ZeroDivisionError
        inv = self.real.inv()
        return DualCliff4.from_clifs(inv, -inv * self.dual * (inv.involution()))

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                return self * DualCliff4.from_clif(other.inv())
            elif isinstance(other, Dual):
                return self * DualCliff4.from_dual(other.inv())
            else:
                return DualCliff4.from_clifs(self.real.value / other, self.dual.value / other)
        return self * other.inv()

    def __rtruediv__(self, other):
        if not isinstance(other, DualCliff4):
            if isinstance(other, Cliff4):
                other = DualCliff4.from_clif(other)
            elif isinstance(other, Dual):
                other = DualCliff4.from_dual(other)
            else:
                other = DualCliff4.from_real(other)
        return other/self

    def __pow__(self, power, modulo=None):
        if power >= 0:
            res = DualCliff4.from_real(np.ones_like(self.real.value[0]))
            for i in range(power):
                res *= self
            return res
        else:
            res = DualCliff4.inv(self)
            return res ** power

    def __getitem__(self, item):
        return DualCliff4(self.real.value[..., item], self.dual.value[..., item])

    def transpose(self):
        return DualCliff4.from_clifs(self.real.transpose(), self.dual.transpose())

    def concatenate(self, other, axis):
        return DualCliff4.from_clifs(self.real.concatenate(other.real, axis), self.dual.concatenate(other.dual, axis))

    @classmethod
    def diagonal(cls, lst):
        lstr, lstd = [], []
        for i in range(len(lst)):
            lstr.append(lst[i].real)
            lstd.append(lst[i].dual)
        return DualCliff4.from_clifs(Cliff4.diagonal(lstr), Cliff4.diagonal(lstd))

    @classmethod
    def se4_to_unit_dual_clif(cls, mat, vec):
        q = Cliff4.so4_to_unit(mat)
        t = DualCliff4(np.array([1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                       np.array([0, vec[0, 0], vec[1, 0], vec[2, 0], vec[3, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 2)
        return t * q

    @classmethod
    def unit_dual_clif_to_se4(cls, dc):
        q = dc.real
        t = dc * q.conjugate()
        return Cliff4.unit_to_so4(q), 2 * np.array([[t.dual.value[1]], [t.dual.value[2]],
                                                    [t.dual.value[3]], [t.dual.value[4]]])



def power_method(mat, tol=1e-6, maxiter=500):
    vec = np.random.rand(mat.shape[0], 1)
    vec = vec/np.linalg.norm(vec)
    val = 0
    count = 0
    norm = np.linalg.norm(mat)
    while np.linalg.norm(mat @ vec - vec * val) > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / np.linalg.norm(vec)
        val = (np.transpose(np.conjugate(vec)) @ mat @ vec)[0, 0]
        count += 1
    return vec, val, count


def advanced_power_method(mat, dim, tol=1e-6):
    temp = np.copy(mat)
    vec, val, count = power_method(temp, tol=tol)
    res = np.copy(vec)
    resv = [val]
    resc = [count]
    for i in range(1, dim):
        temp -= val * vec @ vec.transpose()
        vec, val, count = power_method(temp, tol=tol)
        res = np.hstack((res, vec))
        resv.append(val)
        resc.append(count)
    return res, resv, resc


def power_method_quaternion(mat, tol=1e-6, maxiter=500):
    vec = Quaternion(np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1),
                     np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1))
    vec = vec/Quaternion.norm(vec)
    val = Quaternion(0, 0, 0, 0)
    count = 0
    while Quaternion.norm(mat @ vec - vec * val) > tol and count < maxiter:
        vec = mat @ vec
        vec = vec / Quaternion.norm(vec)
        val = Quaternion.transpose(Quaternion.conjugate(vec)) @ mat @ vec
        count += 1
    return vec, val, count


def power_method_quaternion_v2(mat, tol=1e-6, maxiter=500):
    vec = Quaternion(np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1),
                     np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1))
    vec = vec/Quaternion.norm(vec)
    val = Quaternion(0, 0, 0, 0)
    count = 0
    while Quaternion.norm(mat @ vec - vec * val) > tol and count < maxiter:
        vec = mat @ vec
        vec = vec / Quaternion.norm(vec)
        val = Quaternion.transpose(Quaternion.conjugate(vec)) @ mat @ vec
        x = val.similarizer()
        vec = x.inv() * vec * x
        val = val.similarize()
        count += 1
    return vec, val, count


def power_method_quaternion_v3(mat, tol=1e-6, maxiter=500):
    vec = Quaternion(np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1),
                     np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1))
    vec = vec/Quaternion.norm(vec)
    val = Quaternion(0, 0, 0, 0)
    count = 0
    while Quaternion.norm(mat @ vec - vec * val) > tol and count < maxiter:
        vec = mat @ vec
        vec = vec / Quaternion.norm(vec)
        val = Quaternion.transpose(Quaternion.conjugate(vec)) @ mat @ vec
        count += 1
    x = val.similarizer()
    vec = x.inv() * vec * x
    val = val.similarize()
    return vec, val, count


def power_method_dual(mat, tol=1e-6, maxiter=500):
    vec = Dual(np.random.rand(mat.real.shape[0], 1), np.random.rand(mat.real.shape[0], 1))
    vec = vec / vec.norm(1)
    val = Dual(0, 0)
    count = 0
    norm = mat.norm('F')
    while Dual.norm(mat @ vec - vec * val, 1) > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / Dual.norm(vec, 1)
        val = Dual.transpose(Dual.conjugate(vec)) @ mat @ vec
        count += 1
    return vec, val, count


def power_method_dual_quaternion(mat, tol=1e-10, maxiter=500):
    vec = DualQuaternion(np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1))
    vec = vec / vec.norm()
    val = DualQuaternion(0, 0, 0, 0, 0, 0, 0, 0)
    count = 0
    norm = mat.norm('F')
    while DualQuaternion.norm(mat @ vec - vec * val) > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate().transpose() @ mat @ vec
        count += 1
    return vec, val, count


def power_method_dual_quaternion_v2(mat, tol=1e-10, maxiter=500):
    vec = DualQuaternion(np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1))
    vec = vec / vec.norm()
    val = DualQuaternion(0, 0, 0, 0, 0, 0, 0, 0)
    count = 0
    norm = mat.norm('F')
    while DualQuaternion.norm(mat @ vec - vec * val) > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate().transpose() @ mat @ vec
        count += 1
        x = val.similarizer()
        vec = x.inv() * vec * x
        val = val.similarize()
    return vec, val, count


def power_method_dual_quaternion_v3(mat, tol=1e-10, maxiter=500):
    vec = DualQuaternion(np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1),
                         np.random.rand(mat.real.real.shape[0], 1), np.random.rand(mat.real.real.shape[0], 1))
    vec = vec / vec.norm()
    val = DualQuaternion(0, 0, 0, 0, 0, 0, 0, 0)
    count = 0
    norm = mat.norm('F')
    while DualQuaternion.norm(mat @ vec - vec * val) > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate().transpose() @ mat @ vec
        count += 1
    x = val.similarizer()
    vec = x.inv() * vec * x
    val = val.similarize()
    return vec, val, count


def power_method_cliff4_total_random(mat, tol=1e-6, maxiter=500):
    vec = Cliff4(np.random.rand(16, mat.value[0].shape[0]))
    vec = vec / vec.norm()
    val = Cliff4(np.zeros(16))
    count = 0
    norm = mat.norm('F')
    while Cliff4.norm(mat @ vec - vec * val, 'F') > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate() @ mat @ vec
        count += 1
    return vec, val, count


def power_method_cliff4_random_spin(mat, tol=1e-6, maxiter=500):
    a1, a2, a3, a4, a5, a6, a7 = np.random.rand(7, mat.value[0].shape[0])
    z = np.zeros((mat.value[0].shape[0]))
    vec = Cliff4(np.array([a1, z, z, z, z, a2, a3, a4, a5, a6, a7, z, z, z, z, (a2 * a7 + a4 * a5 - a3 * a6)/a1]))
    vec = vec / vec.norm()
    val = Cliff4(np.zeros(16))
    count = 0
    norm = mat.norm('F')
    while Cliff4.norm(mat @ vec - vec * val, 'F') > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate() @ mat @ vec
        count += 1
    return vec, val, count


def power_method_dual_cliff4_total_random(mat, tol=1e-10, maxiter=500):
    vec = DualCliff4.from_clifs(Cliff4(np.random.rand(16, mat.shape[0])), Cliff4(np.random.rand(16, mat.shape[0])))
    vec = vec / vec.norm()
    val = DualCliff4.from_real(0)
    count = 0
    norm = mat.norm('F')
    while DualCliff4.norm(mat @ vec - vec * val, 'F') > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate() @ mat @ vec
        count += 1
    return vec, val, count


def power_method_dual_cliff4_random_spin(mat, tol=1e-10, maxiter=500):
    a1, a2, a3, a4, a5, a6, a7 = np.random.rand(7, mat.shape[0])
    z = np.zeros((mat.real.value[0].shape[0]))
    rot = Cliff4(np.array([a1, z, z, z, z, a2, a3, a4, a5, a6, a7, z, z, z, z, (a2 * a7 + a4 * a5 - a3 * a6)/a1]))
    trans = np.array([z, np.random.normal(0, 1, mat.shape[0]), np.random.normal(0, 1, mat.shape[0]),
                      np.random.normal(0, 1, mat.shape[0]), np.random.normal(0, 1, mat.shape[0]), z, z, z, z, z, z, z,
                      z, z, z, z])
    vec = DualCliff4.from_clifs(Cliff4.from_real(np.ones(mat.shape[0])), Cliff4(trans)) * rot
    vec = vec / vec.norm()
    val = DualCliff4.from_real(0)
    count = 0
    norm = mat.norm('F')
    while DualCliff4.norm(mat @ vec - vec * val, 'F') > tol * norm and count < maxiter:
        vec = mat @ vec
        vec = vec / vec.norm()
        val = vec.conjugate() @ mat @ vec
        count += 1
    return vec, val, count
