import multiprocessing
import time

import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask, constitutiveSingle
from FEMxEPxML.utils_constitutive import tensor2_tensor3, returnedDatasDecode, \
    get_elastic_matrix_plain_stress, getVolStrain, getP, get_q_2d, get_M_current,\
    get_dpdsig_dqdsigma, get_deps_s_deps, getQEps, getS, voigt_2_tensor
from utilSelf.general import echo, mapMask


class vonmisesIdealConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg,  save_path:str, rho: float,
                 p0=0., nu=0.2, E=2e6, yield_stress0=1e5,
                 verboseFlag=False, ndim=2, save_flag=False, nump=1):
        self.cons = [
            vonmisesIdealSingle(
                explicitFlag=explicitFlag,
                 p0=p0, nu=nu, E=E, yield_stress0=yield_stress0, ndim=ndim) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, p0=p0, ndim=ndim, cons=self.cons, rho=rho,
            explicitFlag=explicitFlag, nump=nump,  numg=numg, name='misesideal', save_path=save_path, save_flag=save_flag)


class vonmisesIdealSingle(constitutiveSingle):
    '''
     used for the plain stress problem.

     NOTE: all the of elastic computaion are under plain stress assumption
    '''
    def __init__(self, p0=0., nu=0.2, E=2e6, yield_stress0=1e5, explicitFlag=True, ndim=2):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag)
        self.nu = nu
        self.E = E
        self.K = self.E/(2*(1.-self.nu))
        self.G = self.E/2/(1+self.nu)
        self.De = get_elastic_matrix_plain_stress(E=self.E, nu=self.nu)
        self.yield_stress0 = yield_stress0

        # computation operator
        self.kronecker = np.eye(2)

        # calculation setting
        self.tolerance_ratio = 0.01

        # state
        self.p0 = p0
        self.sig = np.eye(2) * self.p0
        self.eps = np.zeros(shape=[2, 2])
        self.eps_s = 0.
        self.eps_p = np.zeros(shape=[2, 2])
        self.eps_e = np.zeros(shape=[2, 2])
        self.epsvp = 0.
        self.ndim = ndim
        self.eps_abs = np.zeros(shape=[2, 2])
        self.p, self.q = self.p0, 0.
        self.explicitFlag = explicitFlag

    def prediction_multiprocess(self, deps_numg, num_p=4, num_per=10):
        n_samples, start_time = len(deps_numg), time.time()
        sig_numg, q_numg, eps_e_numg, eps_p_numg = [], [], [], []
        for i in range(n_samples//(num_p*num_per)+1):
            if i * num_p * num_per >= n_samples:
                break
            with multiprocessing.Pool(num_p) as pool:
                temp_list = pool.map(self.forward, deps_numg[i*num_p*num_per:min((i+1)*num_p*num_per, n_samples)])
            for temp in temp_list:
                sig_numg.append(temp[0])
                q_numg.append(temp[1])
                eps_e_numg.append(temp[2])
                eps_p_numg.append(temp[3])
            echo('Generated_samples_number=[%d/%d] consumed_time=%e(mins)' %
                 (min((i+1)*num_p*num_per, n_samples), n_samples, (time.time()-start_time)/60.))
        return np.array(sig_numg), np.array(q_numg), np.array(eps_p_numg), np.array(eps_e_numg)

    def prediction_numg(self, deps_numg, verboseFlag=False):
        '''
                :param deps_numg: shape of (numg, steps, ndim, dim)
                :return:
                '''
        sig_numg, q_numg, eps_e_numg, eps_p_numg = [], [], [], []
        num_numg, total_num_numg, start_time = 0, len(deps_numg), time.time()
        for deps in deps_numg:
            temp = self.forward(deps)
            sig_numg.append(temp[0])
            q_numg.append(temp[1])
            eps_e_numg.append(temp[2])
            eps_p_numg.append(temp[3])
            num_numg += 1
            if (num_numg+1) % 10 == 0 and verboseFlag:
                echo('Generated_samples_number=[%d/%d] consumed_time=%e(mins)' %
                     (num_numg+1, total_num_numg, (time.time()-start_time)/60.))
        return np.array(sig_numg), np.array(q_numg), np.array(eps_p_numg), np.array(eps_e_numg)

    def forward(self, deps):
        '''

        :param deps: in shape of [steps, 2, 2]  compression is negative
        :return:
        '''
        sig_pre, q_pre, eps_e_pre, eps_p_pre = [], [], [], []
        for num, i in enumerate(deps):
            sig_temp, scenes_temp = self.solver(deps=-i)
            q_pre.append(scenes_temp[3][2])
            eps_p_pre.append(-scenes_temp[3][0])
            eps_e_pre.append(-scenes_temp[3][3])
            sig_pre.append(-sig_temp)             # output stress, compression is negative
            self.update(*scenes_temp)

        '''
        eps = self.eps
        sig = self.sig
        val, vec = np.linalg.eig(eps)
        val_sig, vec_sig = np.linalg.eig(sig)
        val_epse, vec_epse = np.linalg.eig(self.eps_e)
        vec - vec_sig
        vec_epse - vec_sig
        '''
        self.return2initial()


        return [sig_pre, q_pre, eps_e_pre, eps_p_pre]

    def solver_single(self, deps):
        # elastic trial
        # TODO how the difference between the two sig_trial arises???
        # sig_trial = np.einsum('ijkl, kl->ij', self.De, deps) + self.sig
        sig_trial = np.einsum('ijkl, kl->ij', self.De, deps) + self.sig

        # deps_voigt = deps.reshape(4)[np.array([0, 1, 3])]
        # kronecker = np.array([1., 0., 1.])
        # deps_v = deps_voigt[0] + deps_voigt[2]
        # de = deps_voigt - deps_v/2.*kronecker
        # dsig_ = deps_v*self.K*kronecker + 2*self.G*de

        # deps_v = np.trace(deps)
        # de = (deps-deps_v/2.*self.kronecker)
        # dsig = de * 2.*self.G + self.kronecker*deps_v*self.K
        # sig_trial_residual = (deps-deps_v/2.*self.kronecker)*self.G + self.kronecker*deps_v*self.K - \
        #                      np.einsum('ijkl, kl->ij', self.De, deps)
        # sig_trial = self.sig+dsig
        q_trial = get_q_2d(sig=sig_trial)
        # yield judgement
        if q_trial <= self.yield_stress0:
            scene = [sig_trial, self.eps+deps, self.eps_abs+np.abs(deps),
                     [self.eps_p, np.trace(sig_trial)/2., q_trial, self.eps_e+deps]]
            if self.explicitFlag:
                return sig_trial, scene
            else:
                return sig_trial, self.De, scene
        else:
            p = np.trace(sig_trial)/2.
            sig = p*self.kronecker + (sig_trial-p*self.kronecker)*(self.yield_stress0/q_trial)
            dsig = sig - self.sig
            dp = np.trace(dsig)/2.
            deps_ev = dp/self.K
            deps_es = (dsig-dp*self.kronecker)/self.G/2.
            deps_e = self.kronecker*deps_ev/2. + deps_es
            deps_p = deps - deps_e
            scene = [sig, self.eps + deps, self.eps_abs + np.abs(deps),
                     [self.eps_p+deps_p, np.trace(sig_trial) / 2., get_q_2d(sig), self.eps_e+deps_e]]

            """
            eps = self.eps  +deps
            val, vec = np.linalg.eig(eps)
            val_sig, vec_sig = np.linalg.eig(sig)
            vec - vec_sig
            """
            if self.explicitFlag:
                return sig, scene
            else:
                return sig, self.De, scene

    def get_current_scene(self):
        scene = [self.sig, self.eps, self.eps_abs,
                 [self.eps_p, self.p, self.q, self.eps_e]]
        return scene

    def transformSplit(self, deps):
        dsig = np.einsum('ijkl, kl->ij', self.De, deps)
        minn, maxx = 0., 1.
        err = 1.
        iteration = 0
        while abs(err) > 0.001:
            mid = 0.5*(minn+maxx)
            q_mid = get_q_2d(sig = dsig*mid+self.sig)
            err = q_mid/self.yield_stress0-1.0
            if err > 0:
                maxx = mid
            else:
                minn = mid
            iteration += 1
            if iteration > 20:
                raise RuntimeError('Please check the q_min=%.e q_max=%.e while q_yield=%.e' %
                                   (self.q, get_q_2d(sig = self.sig+dsig), self.yield_stress0))
        return mid, dsig*mid, q_mid

    def update_internal(self, eps_p, p, q, eps_e):
        self.eps_p, self.p, self.q, self.eps_e = eps_p, p, q, eps_e

    def return2initial(self):
        self.sig = np.eye(2) * self.p0
        self.eps = np.zeros(shape=[2, 2])
        self.eps_s = 0.
        self.eps_p = np.zeros(shape=[2, 2])
        self.eps_e = np.zeros(shape=[2, 2])
        self.epsvp = 0.
        self.eps_abs = np.zeros(shape=[2, 2])
        self.p, self.q = self.p0, 0.


if __name__ == '__main__':
    model_demo = vonmisesIdealSingle()
    eps0 = np.array([0., 0., 0.])
    eps1 = np.array([0., 0., -0.08])
    eps2 = np.array([0., 0., -0.04])
    eps3 = np.array([0., 0., -0.1])
    eps_path = np.concatenate(
        (np.linspace(eps0, eps1, 50), np.linspace(eps1, eps2, 100), np.linspace(eps2, eps3, 100),), axis=0)
    deps_path = np.zeros((len(eps_path), 3))
    deps_path[1:] = eps_path[1:] - eps_path[:-1]
    deps_path = np.array([voigt_2_tensor(voigt=deps_path)])
    sig_numg, q_numg, eps_e_numg, eps_p_numg = model_demo.prediction_numg(deps_numg=deps_path)
    import matplotlib.pyplot as plt
    eps = np.zeros_like(deps_path)
    for numg in range(len(deps_path)):
        eps[numg, 0] = deps_path[numg, 0]
        for step in range(1, len(deps_path[0])):
            eps[numg, step] = deps_path[numg, step] + eps[numg, step-1]
    fig = plt.figure(figsize=[6.4*1.8, 4.8])
    fig.add_subplot(121)
    plt.plot(-eps[0, :, 1, 1], -sig_numg[0, :, 0, 0]/1e6, label='$\sigma_{00}$')
    plt.plot(-eps[0, :, 1, 1], -sig_numg[0, :, 1, 1]/1e6, label='$\sigma_{11}$')
    plt.xlabel(r'$\epsilon_{11}$')
    plt.ylabel(r'MPa')
    plt.legend()
    fig.add_subplot(122)
    plt.plot(-eps[0, :, 1, 1], q_numg[0]/1e6, label='$q$')
    plt.ylabel(r'MPa')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print()












