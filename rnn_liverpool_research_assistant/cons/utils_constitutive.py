import os
import numpy as np
from matplotlib import pyplot as plt
import torch

kronecker = np.eye(3)
a = np.einsum('ij, kl->ijkl', kronecker, kronecker)
b = np.einsum('ik, jl->ijkl', kronecker, kronecker) + \
    np.einsum('il, jk->ijkl', kronecker, kronecker)
dsdsigma = np.einsum('ik, jl->ijkl', kronecker, kronecker) - np.einsum('kl, ij->ijkl', kronecker, kronecker) / 3.

kronecker_torch = torch.eye(3)
a_torch = torch.einsum('ij, kl->ijkl', kronecker_torch, kronecker_torch)
b_torch = torch.einsum('ik, jl->ijkl', kronecker_torch, kronecker_torch) + \
    torch.einsum('il, jk->ijkl', kronecker_torch, kronecker_torch)
dsdsigma_torch = torch.einsum('ik, jl->ijkl', kronecker_torch, kronecker_torch) - \
                 torch.einsum('kl, ij->ijkl', kronecker_torch, kronecker_torch) / 3.


def get_elasticMatrix(lam: float, G: float):
    '''https://en.wikipedia.org/wiki/Linear_elasticity'''
    matrix = a * lam + b * G
    # matrix = np.zeros(shape=[3, 3, 3, 3])
    # for i in range(3):
    #     for j in range(3):
    #         matrix[i, i, j, j] += lam
    # for i in range(3):
    #     matrix[i, i, i, i] += 2. * G
    #     matrix[i, (i + 1) % 3, i, (i + 1) % 3] = \
    #         matrix[i, (i + 1) % 3, (i + 1) % 3, i] = \
    #         matrix[(i + 1) % 3, i, (i + 1) % 3, i] = \
    #         matrix[(i + 1) % 3, i, i, (i + 1) % 3] = G
    return matrix


def get_elasticMatrix_torch(lam:torch.float32, G:torch.float32):
    matrix = a_torch*lam+b_torch*G
    return matrix


def tensor4_mul_tensor2(t4: np.ndarray, t2: np.ndarray):
    return np.einsum('ijkl, kl->ij', t4, t2)


def getP(sigma):
    p = np.trace(sigma) / 3.
    return p

def getP_torch(sigma: torch.Tensor):
    return torch.trace(sigma)


def getS(sigma):
    return sigma - np.eye(3) * getP(sigma)


def getJ2(sigma):
    s = getS(sigma)
    return 0.5 * np.sum(s * s)


def getQ(sigma):
    J2 = getJ2(sigma)
    return np.sqrt(3. * J2)

def getI3(sigma):
    return np.linalg.det(sigma)


def getVolStrain(eps):
    return np.trace(eps)


def getQEps(eps):
    e = getS(eps)
    qeps = np.sqrt(2./3.*np.sum(e*e))
    return qeps


def getdEpsMagtitude(deps):
    return np.sqrt(2. * np.sum(deps * deps) / 3.)


def get_deps_s_deps(eps):
    eps_v = getVolStrain(eps)
    eps_s = getQEps(eps)
    deps_s_deps = 2. / eps_s * (eps - eps_v * np.eye(3) / 3.) if eps_s != 0. else np.sqrt(6) * np.eye(3)
    return deps_s_deps


def get_principle_stress(sigma):
    if sigma.shape == (3, 3):  # tensor notion
        sigma_matrix = sigma
    elif sigma.shape == (2, 2):  # tensor notion
        sigma_matrix = sigma

    elif sigma.shape == (6,):
        '''
            00 11 22 01 12 20
            0  1  2  3  4  5
        '''
        sigma_matrix = np.array([
            [sigma[0], sigma[3], sigma[5]],
            [sigma[3], sigma[1], sigma[4]],
            [sigma[5], sigma[4], sigma[2]]])
    else:
        ''' 00 01 11'''
        print('Not support the 2D problem printcipal calculation')
        print('Shape of the input sigma is %s' % sigma.shape)
        raise
    try:
        eigvals = np.linalg.eigvals(sigma_matrix)
    except:
        print(sigma_matrix)
        raise
    return eigvals


def get_dqeps_deps(eps):
    eps_s = getS(eps)
    j2eps = getJ2Eps(eps)
    dqeps_dj2 = 1. / np.sqrt(3. * j2eps) if j2eps != 0. else 2. / np.sqrt(3.)
    dj2deps_s = eps_s
    dqeps_deps = dqeps_dj2 * np.einsum('ij, ijkl->kl', dj2deps_s, dsdsigma)
    return dqeps_deps


def get_dpdsig_dqdsigma(sigma):
    p, q = getP(sigma), getQ(sigma)
    dpdsig = np.eye(3) / 3.
    dqdsig = 1.5 / q * (sigma - p * np.eye(3)) if q != 0. else np.sqrt(3) * 0.5 * np.eye(3)
    return dpdsig, dqdsig


def getInvariantsSigma(sigma):
    ''' https://en.wikipedia.org/wiki/Cauchy_stress_tensor '''
    I1 = np.trace(sigma)
    I2 = 0.5 * (np.trace(sigma) ** 2. - np.trace(sigma ** 2.))
    I3 = np.linalg.det(sigma)
    return I1, I2, I3


def get_qc_smp(sigma):
    i1, i2, i3 = getInvariantsSigma(sigma)
    temp1 = i1 * i2 - i3
    temp2 = i1 * i2 - 9. * i3
    if temp2 == 0.:
        return 0.
    temp = temp1 / temp2
    if temp < 0.:
        return 0.
    qc = 2. * i1 / (3. * np.sqrt(temp) - 1.)
    return qc


def get_dqc_di(sig):
    x, y, z = getInvariantsSigma(sig)
    temp1 = y * x - z
    temp2 = y * x - 9 * z
    if temp2 <= 0.:
        print('I1=%.3f I2=%.3f I3=%.3f' % (x, y, z))
        print('I1*I2-9I3=%.3f' % temp2)
        return np.array([1., 1., 1.])
    temp3 = temp1 / temp2
    if temp3 <= 0.:
        return np.array([1., 1., 1.])
    temp4 = np.sqrt(temp3)  # overflow
    dqc_di1 = 2. / (3 * temp4 - 1.) - 3 * x * y * (1 - temp3) / temp2 / (temp4 * (3 * temp4 - 1.) ** 2.)
    dqc_di2 = 24. * x ** 2 * z / (temp2 ** 2 * temp4 * (3 * temp4 - 1.) ** 2.)
    dqc_di3 = -24. * x ** 2 * y / ((3 * temp4 - 1.) ** 2. * temp4 * temp2 ** 2.)
    return np.array([dqc_di1, dqc_di2, dqc_di3])


def get_di_dsig(sig):
    '''
        https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)
    '''
    i1, i2, i3 = getInvariantsSigma(sig)
    di1_dsig = np.eye(3)
    di2_dsig = i1 * np.eye(3) - sig.T
    di3_dsig = (sig * sig - i1 * sig + i2 * np.eye(3)).T
    # di3_dsig = i2 * np.eye(3) - sig.T @ (i1 * np.eye(3) - sig.T)
    return np.array([di1_dsig, di2_dsig, di3_dsig])


def getSigma_ts(sigma, p, q, qc):
    if q == 0.:
        return sigma
    sigma_ts = p * np.eye(3) + qc / q * (sigma - np.eye(3) * p)
    return sigma_ts


def get_b(sigma1, sigma2, sigma3):
    b = (sigma3 - sigma2) / (sigma3 - sigma1) if sigma3 - sigma1 != 0. else 0.
    return b


def getLode(b):
    return np.arctan((1 - 2. * b) / np.sqrt(3.))


def voigt2tensor(vector, epsFlag):
    """
    vector: 00 11 22 01 12 20
    """
    scaler = 1.0
    if epsFlag:
        scaler = 0.5
    tensor = np.array([[vector[0], vector[3] * scaler, vector[5] * scaler],
                       [vector[3] * scaler, vector[1], vector[4] * scaler],
                       [vector[5] * scaler, vector[4] * scaler, vector[2]]])
    return tensor


def tensor2voigt(tensor):
    """
    vector:     2D  voigt format 00 01 22
                3D  voigt format 00 11 22 01 12 20
    """
    shape = list(tensor.shape)[:-1]
    if shape[-1] == 3:
        shape[-1] = 6
        vector = np.zeros(shape=shape)
        vector[..., 0] = tensor[..., 0, 0]
        vector[..., 1] = tensor[..., 1, 1]
        vector[..., 2] = tensor[..., 2, 2]
        vector[..., 3] = tensor[..., 0, 1]
        vector[..., 4] = tensor[..., 1, 2]
        vector[..., 5] = tensor[..., 2, 0]
    elif shape[-1] == 2:
        shape[-1] = 3
        vector = np.zeros(shape=shape)
        vector[..., 0] = tensor[..., 0, 0]
        vector[..., 1] = tensor[..., 0, 1]
        vector[..., 2] = tensor[..., 1, 1]
    else:
        raise
    return vector


def plotSubFigures(ax, x, y, label, xlabel, ylabel, num=None, color=None):
    if num and num >= 1:
        for i in range(num):
            if color:
                ax.plot(x[i], y[i], label=label[i], lw=3, alpha=0.5, color=color)
            else:
                ax.plot(x[i], y[i], label=label[i], lw=3, alpha=0.5)
    else:
        raise ValueError('Please give the num')
    plt.legend(fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


def loadingPathReader(path='MCCData'):
    path = os.path.join(path, 'loadingPath')
    fileList = [os.path.join(path, i) for i in os.listdir(path) if '.dat' in i]
    loadPathList = []
    for i in fileList:
        pathTemp = np.loadtxt(fname=i, delimiter=',', skiprows=1)
        loadPathList.append(pathTemp)
    return loadPathList


def tensor2_tensor3(t2: np.ndarray):
    t3 = np.zeros(shape=[len(t2), 3, 3])
    t3[:, :2, :2] = t2
    return t3


def tensor2d_to_3d_single(tensor2d):
    '''
        deps in shape of [2, 2]
    '''
    tensor_3d = np.zeros([3, 3])
    tensor_3d[:2, :2] = tensor2d
    return tensor_3d


def D_3d_to_2d_single(D_3d):
    D_2d = D_3d[:2, :2, :2, :2]
    return D_2d


def voigt_2_tensor(voigt: np.ndarray, len_steps=None):
    '''

    :param voigt:  shape of (step, (00, 01, 11))
    :param len_steps: int, total number of steps going to be extracted
    :return: shape of (step, 3)
    '''
    if len(voigt[0]) == 3:
        if len_steps is None:
            tensor = np.zeros([len(voigt), 2, 2])
            for i in range(len(voigt)):
                tensor[i, 0, 0] = voigt[i, 0]
                tensor[i, 0, 1] = tensor[i, 1, 0] = voigt[i, 1]
                tensor[i, 1, 1] = voigt[i, 2]
        else:
            index_temp = np.arange(0, len(voigt), len(voigt)//len_steps)
            tensor = np.zeros([len(index_temp), 2, 2])
            for i in range(len(index_temp)):
                tensor[i, 0, 0] = voigt[index_temp[i], 0]
                tensor[i, 0, 1] = tensor[i, 1, 0] = voigt[index_temp[i], 1]
                tensor[i, 1, 1] = voigt[index_temp[i], 2]
    else:
        raise
    return tensor


def voigt_2_tensor_high(voigt: np.ndarray, len_steps=None):
    '''
    :param voigt: shape of (numg, step, (00, 01, 11)))
    :param len_steps: int, total number of the steps going to extract
    :return: shape of (numg, step, 2, 2)
    '''
    voigt_tensor = []
    for i in voigt:
        voigt_tensor.append(voigt_2_tensor(i, len_steps=len_steps))
    return np.array(voigt_tensor)


def returnedDatasDecode(explicitFlag: bool, datas: list, numg: int, name=None):
    sig_geo = []
    if explicitFlag:
        scenes = []
        if name == '2ml':
            sig_err = []
            for i in range(numg):
                sig_geo.append(datas[i][0])
                sig_err.append(datas[i][1])
                scenes.append(datas[i][2])
            return np.array(sig_geo), sig_err, scenes
        else:
            for i in range(numg):
                sig_geo.append(datas[i][0])
                scenes.append(datas[i][1])
            return np.array(sig_geo), scenes
    else:  # implicit
        D = []
        scenes = []
        for i in range(numg):
            sig_geo.append(datas[i][0])
            D.append(datas[i][1])
            scenes.append(datas[i][2])
        return np.array(sig_geo), np.array(D), scenes


def get_M_compression(theta_degree: float):
    theta_rad = theta_degree / 180. * np.pi
    M = 6. * np.sin(theta_rad) / (3. - np.sin(theta_rad))
    return M


def get_M_current(M_compression, theta):
    '''
        Reference: Pastor, Zienkiewicz (1990), Generalized Plasticity and the Modelling of soil behaviour, Eq. (55)
    :param M_compression:
    :param theta:
    :return:
    '''
    return 18.*M_compression/(18.+3.*(1.-np.sin(3.*theta)))


def get_theta(sig):
    J2 = getJ2(sigma=sig)
    J3 = np.linalg.det(getS(sigma=sig))
    if J2 == 0:
        theta = np.pi/6.
    else:
        temp = J3 / 2. * (3. / J2) ** 1.5
        if np.abs(temp) > 1:
            theta = np.sign(temp) * np.pi / 6.
        else:
            theta = np.arcsin(temp) / 3.
    return theta

#
# def getQ_2d(sig: np.ndarray):
#     '''
#         sig is in [N, 3] (11 12 22)
#     '''
#     q = np.sqrt(
#         sig[:, 0:1] ** 2. +
#         3. * sig[:, 1:2] ** 2. +
#         sig[:, 2:3] ** 2. -
#         sig[:, 0:1] * sig[:, 2:3])
#     return q
#
#
# def get_dq_dsig_2d(sig: np.ndarray):
#     '''
#         sig is in [N, 3] (11 12 22)
#         dqdsig = (11 12 22)
#     '''
#     q = getQ_2d(sig)
#     dqdsig = np.concatenate(
#         ((2. * sig[:, 0:1] - sig[:, 2:3]) / q / 2.,
#          3. * sig[:, 1:2] / q,
#          (2. * sig[:, 2:3] - sig[:, 0:1]) / q / 2.), axis=1)
#     return dqdsig


def get_q_2d(sig):
    '''
    :param sig:  in shape of (2, 2)
    :return:
    '''
    # sig = np.diag([1, 5])
    s = sig - np.trace(sig)/2.*np.eye(2)
    q = np.sqrt(2*np.sum(s*s))
    return q


def get_q_eps_2d(eps):
    '''
    :param eps:
    :return:
    '''
    s = eps - np.trace(eps)/2.*np.eye(2)
    q = np.sqrt(2./3.*np.sum(s*s))
    return q




def get_elastic_matrix_plain_stress(E, nu):
    k = np.eye(2)
    temp1 = np.einsum('ik, jl->ijkl', k, k)*0.5 + np.einsum('il, jk->ijkl', k, k)*0.5
    temp2 = np.einsum('ij, kl->ijkl', k, k)
    De = E/(1 - nu**2)*((1-nu)*temp1 + nu * temp2)
    return De


if __name__ == '__main__':
    tensor = np.random.random(size=[3, 3])
    tensor2voigt(tensor)