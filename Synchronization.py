import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import PowerMethod as pm


def build_random_element(group, dim=None):
    if group == 's1':
        return np.exp(1.0j * np.random.uniform(0, 2 * np.pi))
    elif group == 'se':
        return sp.stats.special_ortho_group.rvs(dim), np.random.normal(0, 1, (dim, 1))
    elif group == 'quat':
        return pm.Quaternion.so3_to_unit_quaternion(sp.stats.special_ortho_group.rvs(3))
    elif group == 'dq':
        return pm.DualQuaternion.se3_to_unit_dual_quaternion(sp.stats.special_ortho_group.rvs(3),
                                                             np.random.normal(0, 1, (3, 1)))
    elif group == 'clif':
        return pm.Cliff4.so4_to_unit(sp.stats.special_ortho_group.rvs(4))
    elif group == 'dc':
        return pm.DualCliff4.se4_to_unit_dual_clif(sp.stats.special_ortho_group.rvs(4), np.random.normal(0, 1, (4, 1)))
    elif group == 'on':
        return sp.stats.ortho_group.rvs(dim)
    else:
        return sp.stats.special_ortho_group.rvs(dim)


def create_random_subset(num, group, dim=None):
    return [build_random_element(group, dim) for i in range(num)]


def so_multiplicative_noise(sigma, dim=None):
    a = np.random.normal(0, sigma, int((dim * (dim - 1))/2))
    noise = np.zeros((dim, dim))
    next = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            noise[i, j] = a[next]
            noise[j, i] = -noise[i, j]
            next += 1
    return sp.linalg.expm(noise)


def subset_to_ratios(lst, group, dim):
    n = len(lst)
    if group == 'quat':
        real, i, j, k = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                mul = lst[row] * pm.Quaternion.conjugate(lst[col])
                real[row, col], i[row, col], j[row, col], k[row, col] = mul.real, mul.i, mul.j, mul.k
        return pm.Quaternion(real, i, j, k)
    elif group == 'dq':
        real, ri, rj, rk = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        dual, di, dj, dk = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                mul = lst[row] * pm.DualQuaternion.conjugate(lst[col])
                real[row, col], ri[row, col], rj[row, col], rk[row, col] = mul.real.real, mul.real.i, mul.real.j, mul.real.k
                dual[row, col], di[row, col], dj[row, col], dk[row, col] = mul.dual.real, mul.dual.i, mul.dual.j, mul.dual.k
        return pm.DualQuaternion(real, ri, rj, rk, dual, di, dj, dk)
    elif group == 'clif':
        mat = np.zeros((16, n, n))
        for row in range(n):
            for col in range(n):
                mul = lst[row] * pm.Cliff4.conjugate(lst[col])
                mat[:, row, col] = mul.value
        return pm.Cliff4(mat)
    elif group == 'dc':
        real, dual = np.zeros((16, n, n)), np.zeros((16, n, n))
        for row in range(n):
            for col in range(n):
                mul = lst[row] * pm.DualCliff4.conjugate(lst[col])
                real[:, row, col] = mul.real.value
                dual[:, row, col] = mul.dual.value
        return pm.DualCliff4(real, dual)
    else:
        if group == 's1':
            res = np.zeros((n, n), dtype=np.complex_)
        else:
            res = np.zeros((n * dim, n * dim))
        for row in range(n):
            for col in range(n):
                if group == 's1':
                    res[row, col] = lst[row] * np.conjugate(lst[col])
                else:
                    res[row * dim:(row + 1) * dim, col * dim:(col + 1) * dim] = lst[row] @ lst[col].transpose()
    return res


def group_rounding(elem, group, dim):
    if group == 's1':
        return elem / np.absolute(elem)
    elif group == 'quat':
        return elem / pm.Quaternion.norm(elem)
    elif group == 'dq':
        return elem / pm.DualQuaternion.magnitude(elem)
    elif group == 'clif':
        return elem / pm.Cliff4.magnitude(elem)
    elif group == 'dc':
        return elem / pm.DualCliff4.magnitude(elem)
    else:
        if group == 'on':
            U, s, Vh = sp.linalg.svd(elem)
            return U @ Vh
        else:
            if np.linalg.det(elem) < 0:
                P = np.identity(dim)
                P[:, [0, 1]] = P[:, [1, 0]]
                return group_rounding(elem @ P, 'so', dim)
            else:
                U, s, Vh = sp.linalg.svd(elem)
                return U @ Vh


def group_error(elems, approx, group, dim):
    if group == 's1':
        minimizer = 0.0j
        for i in range(len(elems)):
            minimizer += np.conjugate(approx[i]) * elems[i]
        minimizer = group_rounding(minimizer, 's1', 1)
        err = 0
        for i in range(len(elems)):
            err += np.linalg.norm(elems[i] - approx[i] * minimizer) ** 2
        return err
    elif group == 'quat':
        minimizer = pm.Quaternion.from_real(0)
        for i in range(len(elems)):
            minimizer += pm.Quaternion.conjugate(approx[i]) * elems[i]
        minimizer = group_rounding(minimizer, 'quat', 1)
        err = 0
        for i in range(len(elems)):
            err += pm.Quaternion.norm(elems[i] - approx[i] * minimizer) ** 2
        return err
    elif group == 'dq':
        minimizer = pm.DualQuaternion.from_real(0)
        for i in range(len(elems)):
            minimizer += pm.DualQuaternion.conjugate(approx[i]) * elems[i]
        minimizer = group_rounding(minimizer, 'quat', 1)
        err = 0
        for i in range(len(elems)):
            err += pm.DualQuaternion.magnitude(elems[i] - approx[i] * minimizer) ** 2
        return err
    elif group == 'clif':
        minimizer = pm.Cliff4.from_real(0)
        for i in range(len(elems)):
            minimizer += pm.Cliff4.conjugate(approx[i]) * elems[i]
        minimizer = group_rounding(minimizer, 'clif', 1)
        err = 0
        for i in range(len(elems)):
            err += pm.Cliff4.magnitude(elems[i] - approx[i] * minimizer, 'F') ** 2
        return err
    elif group == 'dc':
        minimizer = pm.DualCliff4.from_real(0)
        for i in range(len(elems)):
            minimizer += pm.DualCliff4.conjugate(approx[i]) * elems[i]
        minimizer = group_rounding(minimizer, 'dc', 1)
        err = 0
        for i in range(len(elems)):
            err += pm.DualCliff4.magnitude(elems[i] - approx[i] * minimizer, 'F') ** 2
        return err
    else:
        minimizer = np.zeros_like(elems[0])
        for i in range(len(elems)):
            minimizer += approx[i].transpose() @ elems[i]
        minimizer = group_rounding(minimizer, group, dim)
        err = 0
        for i in range(len(elems)):
            err += np.linalg.norm(approx[i] @ minimizer - elems[i]) ** 2
        return err


def spectral_method_full_graph(lst, group, dim, tol=1e-6):
    if isinstance(lst, np.ndarray):
        mat = lst
        num = np.shape(lst)[0] // dim
    elif isinstance(lst[0], pm.Cliff4):
        if isinstance(lst[0].value[0], np.ndarray):
            mat = lst
            num = lst[0].shape[0] // dim
        else:
            mat = subset_to_ratios(lst, group, dim)
            num = len(lst)
    elif isinstance(lst[0], pm.DualCliff4):
        if isinstance(lst[0].real.value[0], np.ndarray):
            mat = lst
            num = lst[0].shape[0] // dim
        else:
            mat = subset_to_ratios(lst, group, dim)
            num = len(lst)
    elif isinstance(lst.real.real, np.ndarray):
        mat = lst
        num = np.shape(lst.real.real)[0] // dim
    else:
        mat = subset_to_ratios(lst, group, dim)
        num = len(lst)

    if group == 'quat':
        eigs, vals, count = pm.power_method_quaternion(mat, tol)
    elif group == 'dq':
        eigs, vals, count = pm.power_method_dual_quaternion(mat, tol)
    elif group == 'clif':
        eigs, vals, count = pm.power_method_cliff4_random_spin(mat, tol)
        # print(count)
    elif group == 'dc':
        eigs, vals, count = pm.power_method_dual_cliff4_random_spin(mat, tol)
        # print(count)
    else:
        eigs, vals, count = pm.advanced_power_method(mat, dim, tol)
    # eigs = sp.linalg.eigh(mat)[1][:, -1:-(dim + 1):-1]
    res = []
    if dim == 1:
        for i in range(num):
            res.append(group_rounding(eigs[i], group, dim))
    else:
        for i in range(num):
            res.append(group_rounding(eigs[dim * i: dim * (i + 1)], group, dim))
    return res


def random_confidence(mat, prob, group, dim):
    if group == 'quat':
        shape = mat.real.shape[0]//dim
    elif group == 'dq':
        shape = mat.real.real.shape[0]//dim
    elif group == 'clif' or group == 'dc':
        shape = mat.shape[0]
    else:
        shape = mat.shape[0]//dim
    res = np.ones((shape, shape), dtype=float)
    for i in range(1, shape):
        for j in range(i):
            if np.random.uniform(0, 1) < prob:
                res[i, j] = 0
                res[j, i] = 0
            else:
                res[i, j] = np.random.uniform(0, 1)
                res[j, i] = res[i, j]
    return res


def spectral_method_partial_graph(ratios, weights, group, dim, tol=1e-6):
    d = 1/np.sqrt(np.sum(weights, axis=1))
    sqrt = sp.linalg.kron(np.diag(d), np.identity(dim))
    mat = ratios * sp.linalg.kron(weights, np.ones((dim, dim)))
    if group == 'quat':
        eigs, vals, count = pm.power_method_quaternion(pm.Quaternion.__rmatmul__(mat @ sqrt, sqrt), tol)
    elif group == 'dq':
        eigs, vals, count = pm.power_method_dual_quaternion(pm.DualQuaternion.__rmatmul__(mat @ sqrt, sqrt), tol)
    elif group == 'clif':
        eigs, vals, count = pm.power_method_cliff4_total_random(pm.Cliff4.__rmatmul__(mat @ sqrt, sqrt), tol)
        print(count)
    elif group == 'dc':
        eigs, vals, count = pm.power_method_dual_cliff4_total_random(pm.DualCliff4.__rmatmul__(mat @ sqrt, sqrt), tol)
        print(count)
    else:
        eigs, vals, count = pm.advanced_power_method(sqrt @ mat @ sqrt, dim, tol)
    # eigs = sp.linalg.eigh(sqrt @ mat @ sqrt)[1][:, -1:-(dim + 1):-1]
    res = []
    if dim == 1:
        for i in range(weights.shape[0]):
            res.append(group_rounding(d[i] * eigs[dim * i], group, dim))
    else:
        for i in range(weights.shape[0]):
            res.append(group_rounding(d[i] * eigs[dim * i: dim * (i + 1)], group, dim))
    return res


def noisy_ratios(mat, prob, group, dim):
    if group == 'clif':
        res = np.copy(mat.value)
    elif group == 'dc':
        resr = np.copy(mat.real.value)
        resd = np.copy(mat.dual.value)
    else:
        res = np.copy(mat)
    weights = np.ones((mat.shape[0] // dim, mat.shape[1] // dim))
    if group == 's1':
        for i in range(1, mat.shape[0]):
            for j in range(i):
                if np.random.uniform(0, 1) > prob:
                    res[i, j] = build_random_element(group, dim)
                    res[j, i] = np.conjugate(res[i, j].transpose())
                    weights[i, j] = np.random.uniform(0, 1/3)
                    weights[j, i] = weights[i, j]
                else:
                    weights[i, j] = np.random.uniform(1/3, 1)
                    weights[j, i] = weights[i, j]
    elif group == 'clif':
        for i in range(1, mat.shape[0]):
            for j in range(i):
                if np.random.uniform(0, 1) > prob:
                    res[:, i, j] = build_random_element(group, dim).value
                    res[:, j, i] = pm.Cliff4(res[:, i, j]).conjugate().value
                    weights[i, j] = np.random.uniform(0, 1 / 3)
                    weights[j, i] = weights[i, j]
                else:
                    weights[i, j] = np.random.uniform(1 / 3, 1)
                    weights[j, i] = weights[i, j]
        return pm.Cliff4(res), weights
    elif group == 'dc':
        for i in range(1, mat.shape[0]):
            for j in range(i):
                if np.random.uniform(0, 1) > prob:
                    rand = build_random_element(group, dim)
                    resr[:, i, j] = rand.real.value
                    resd[:, i, j] = rand.dual.value
                    resr[:, j, i] = rand.conjugate().real.value
                    resd[:, j, i] = rand.conjugate().dual.value
                    weights[i, j] = np.random.uniform(0, 1 / 3)
                    weights[j, i] = weights[i, j]
                else:
                    weights[i, j] = np.random.uniform(1 / 3, 1)
                    weights[j, i] = weights[i, j]
        return pm.DualCliff4(resr, resd), weights
    else:
        for i in range(1, mat.shape[0] // dim):
            for j in range(i):
                if np.random.uniform(0, 1) > prob:
                    res[dim * i: dim * (i + 1), dim * j: dim * (j + 1)] = build_random_element(group, dim)
                    res[dim * j: dim * (j + 1), dim * i: dim * (i + 1)] =\
                        res[dim * i: dim * (i + 1), dim * j: dim * (j + 1)].transpose()
                    weights[i, j] = np.random.uniform(0, 1 / 3)
                    weights[j, i] = weights[i, j]
                else:
                    weights[i, j] = np.random.uniform(1 / 3, 1)
                    weights[j, i] = weights[i, j]
    return res, weights


def experiment(lst, group, sigma, sigmat=None, dim=None):
    n = len(lst)
    if group == 'clif':
        mat = np.zeros((16, n, n))
        for row in range(n):
            for col in range(row, n):
                if row == col:
                    mat[0, row, col] = 1.
                else:
                    noise = so_multiplicative_noise(sigma, 4)
                    mul = lst[row] * pm.Cliff4.conjugate(lst[col]) * pm.Cliff4.so4_to_unit(noise)
                    mat[:, row, col] = mul.value
                    mat[:, col, row] = mul.conjugate().value
        return pm.Cliff4(mat)
    else:
        real, dual = np.zeros((16, n, n)), np.zeros((16, n, n))
        for row in range(n):
            for col in range(row, n):
                if row == col:
                    real[0, row, col] = 1.
                else:
                    noise = (so_multiplicative_noise(sigma, 4), np.random.normal(0, sigmat, (4, 1)))
                    mul = lst[row] * pm.DualCliff4.conjugate(lst[col]) \
                          * pm.DualCliff4.se4_to_unit_dual_clif(noise[0], noise[1])
                    real[:, row, col] = mul.real.value
                    dual[:, row, col] = mul.dual.value
                    real[:, col, row] = mul.conjugate().real.value
                    dual[:, col, row] = mul.conjugate().dual.value
        return pm.DualCliff4(real, dual)


def rotation_error(elems, approx, dim=None):
    minimizer = np.zeros((dim, dim))
    for i in range(len(elems)):
        minimizer += approx[i][0].transpose() @ elems[i][0]
    minimizer = group_rounding(minimizer, 'so', dim)
    err = 0
    for i in range(len(elems)):
        err += np.linalg.norm(approx[i][0] @ minimizer - elems[i][0]) ** 2
    return err


def translation_error(elems, approx, dim=None):
    minimizer = np.zeros((dim, 1))
    for i in range(len(elems)):
        minimizer += approx[i][0].transpose() @ (elems[i][1] - approx[i][1])
    minimizer = 1/len(elems) * minimizer
    err = 0
    for i in range(len(elems)):
        err += np.linalg.norm(approx[i][0] @ minimizer + approx[i][1] - elems[i][1]) ** 2
    return err


rot_err = np.zeros((50, 20))
trans_err = np.zeros((50, 20))
for sig in range(0, 20):
    print(f'sigma = {sig + 1}')
    for repetition in range(50):
        A = create_random_subset(100, 'se', 4)
        B = [pm.DualCliff4.se4_to_unit_dual_clif(elem[0], elem[1]) for elem in A]
        # Bmat = experiment(B, 'dc', (sig + 1) * np.pi / 180, 0)
        # Bmat = experiment(B, 'dc', 0, (sig + 1) * 0.01)
        Bmat = experiment(B, 'dc', (sig + 1) * np.pi / 180, (sig + 1) * 0.01)
        # Amat, weight = noisy_ratios(subset_to_ratios(A, 'dc', 1), 0.7, 'dc', 1)
        # Amat = subset_to_ratios(A, 'dc', 1)
        # weight = random_confidence(Amat, 0.5, 'dc', 1)
        # B = spectral_method_partial_graph(Amat, weight, 'dc', 1, tol=1e-6)
        C = spectral_method_full_graph(Bmat, 'dc', 1)
        D = [pm.DualCliff4.unit_dual_clif_to_se4(approx) for approx in C]
        rot_err[repetition, sig] = rotation_error(A, D, 4)
        trans_err[repetition, sig] = translation_error(A, D, 4)
        print(repetition + 1)
max_r = np.max(rot_err, axis=0)
max_t = np.max(trans_err, axis=0)
min_r = np.min(rot_err, axis=0)
min_t = np.min(trans_err, axis=0)
mean_r = np.mean(rot_err, axis=0)
mean_t = np.mean(trans_err, axis=0)
fig, axs = plt.subplots(1, 2)
x = range(1, 21)
axs[0].plot(x, mean_r)
axs[0].set_ylabel('Rotation Error')
axs[0].fill_between(x, min_r, max_r, alpha=0.2)
axs[0].grid()
axs[0].set_yscale('log')
axs[1].plot(x, mean_t)
axs[1].set_ylabel('Translation Error')
axs[1].fill_between(x, min_t, max_t, alpha=0.2)
axs[1].grid()
axs[1].set_yscale('log')
plt.show()


# def noise_model_group(lst, prob, group, dim):
#     n = len(lst)
#     noisy = np.zeros((dim * n, dim * n))
#     if group == 's1':
#         for i in range(n):
#             for j in range(n):
#                 if np.random.uniform(0, 1) < prob:
#                     noisy[i, j] = lst[i] * np.conjugate(lst[j])
#                 else:
#                     noisy[i, j] = build_random_element('s1', 1)
#     else:
#         for i in range(n):
#             for j in range(n):
#                 if np.random.uniform(0, 1) < prob:
#                     noisy[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = lst[i] @ lst[j].transpose()
#                 else:
#                     noisy[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = build_random_element(group, dim)
#     mat = noisy + prob * subset_to_ratios(lst, group, dim)
#     spec = sp.linalg.eigvals(mat)
#     plt.hist(spec, 50, density=True)
#     plt.show()
