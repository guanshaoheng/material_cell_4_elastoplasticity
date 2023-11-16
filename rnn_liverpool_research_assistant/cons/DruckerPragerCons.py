import multiprocessing
import time
import numpy as np
from constitutive import ConstitutiveMask, constitutiveSingle
from utils_constitutive import get_elastic_matrix_plain_stress, get_q_2d, voigt_2_tensor
from utilSelf.general import echo


class druckerPragerConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, save_path: str, rho: float,
                 p0=1e5, nu=0.2, E=2e6, ndim=2, save_flag=False, nump=4):
        self.cons = [
            druckerPragerSingle(
                explicitFlag=explicitFlag,
                p0=p0, nu=nu, E=E, ndim=2) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, p0=p0, ndim=ndim, cons=self.cons, rho=rho,
            explicitFlag=explicitFlag, nump=nump, numg=numg, name='drucker', save_path=save_path,
            save_flag=save_flag)


class druckerPragerSingle(constitutiveSingle):
    '''
     used for the plain stress problem.

     NOTE: all the of elastic computaion are under plain stress assumption
    '''

    def __init__(
            self,
            p0=1e5, nu=0.2, E=2e6,
            theta=15.,  # in degree
            psi=8.,     # in degree dilatation angle
            explicitFlag=True, ndim=2):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag)
        self.theta = theta / 180 * np.pi  # in rad
        self.M = 2.0 * np.sin(self.theta)
        self.M_dilation = 2.0 * np.sin(psi/180.*np.pi)
        self.nu = nu
        self.E = E
        self.K = self.E / (2 * (1. - self.nu))
        self.G = self.E / 2 / (1 + self.nu)
        self.De = get_elastic_matrix_plain_stress(E=self.E, nu=self.nu)

        # computation operator
        self.kronecker = np.eye(2)

        # calculation setting
        self.tolerance_abs = 10

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
        if not explicitFlag:
            raise ValueError('Currently the drucker model is not implemented for implicit solver!')
        self.yieldValue = self.yieldFunction(p=self.p, q=self.q)

    def prediction_multiprocess(self, deps_numg, num_p=4, num_per=10):
        n_samples, start_time = len(deps_numg), time.time()
        sig_numg, q_numg, eps_e_numg, eps_p_numg = [], [], [], []
        for i in range(n_samples // (num_p * num_per) + 1):
            if i * num_p * num_per >= n_samples:
                break
            with multiprocessing.Pool(num_p) as pool:
                temp_list = pool.map(self.forward,
                                     deps_numg[i * num_p * num_per:min((i + 1) * num_p * num_per, n_samples)])
            for temp in temp_list:
                sig_numg.append(temp[0])
                q_numg.append(temp[1])
                eps_e_numg.append(temp[2])
                eps_p_numg.append(temp[3])
            echo('Generated_samples_number=[%d/%d] consumed_time=%e(mins)' %
                 (min((i + 1) * num_p * num_per, n_samples), n_samples, (time.time() - start_time) / 60.))
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
            if (num_numg + 1) % 10 == 0 and verboseFlag:
                echo('Generated_samples_number=[%d/%d] consumed_time=%e(mins)' %
                     (num_numg + 1, total_num_numg, (time.time() - start_time) / 60.))
        return np.array(sig_numg), np.array(q_numg), np.array(eps_p_numg), np.array(eps_e_numg)

    def forward(self, deps):
        '''

        :param deps: in shape of [steps, 2, 2]  input compression is negative
        :return:
        '''
        sig_pre, q_pre, eps_e_pre, eps_p_pre = [], [], [], []
        for num, i in enumerate(deps):
            sig_temp, scenes_temp = self.solver(deps=-i)
            q_pre.append(scenes_temp[3][2])
            eps_p_pre.append(-scenes_temp[3][0])
            eps_e_pre.append(-scenes_temp[3][3])
            sig_pre.append(-sig_temp)             # output stress, compression is negtive
            self.update(*scenes_temp)
        self.return2initial()
        return [sig_pre, q_pre, eps_e_pre, eps_p_pre]

    def solver_single(self, deps):
        # elastic trial
        # sig_trial = np.einsum('ijkl, kl->ij', self.De, deps)+self.sig

        # deps_voigt = deps.reshape(4)[np.array([0, 1, 3])]
        # kronecker = np.array([1., 0., 1.])
        # deps_v = deps_voigt[0] + deps_voigt[2]
        # de = deps_voigt - deps_v/2.*kronecker
        # dsig_ = deps_v*self.K*kronecker + 2*self.G*de

        deps_v = np.trace(deps)
        de = (deps - deps_v / 2. * self.kronecker)
        dsig = de * 2. * self.G + self.kronecker * deps_v * self.K
        # sig_trial_residual = (deps-deps_v/2.*self.kronecker)*self.G + self.kronecker*deps_v*self.K - \
        #                      np.einsum('ijkl, kl->ij', self.De, deps)
        sig_trial = self.sig + dsig
        q_trial = get_q_2d(sig=sig_trial)
        # yield judgement
        f = self.yieldFunction(p=np.trace(sig_trial)/2., q=q_trial)
        if f <= 0:
            scene = [sig_trial, self.eps + deps, self.eps_abs + np.abs(deps),
                     [self.eps_p, np.trace(sig_trial) / 2., q_trial, self.eps_e + deps, self.yieldValue]]
            if self.explicitFlag:
                return sig_trial, scene
            else:
                return sig_trial, self.De, scene
        elif self.yieldValue < -self.tolerance_abs:
            mid, dsig, q_mid, yieldValue_last = self.transformSplit(deps)
            sig_last = self.sig+dsig
            q_last = q_mid
            return self.plasticReturnMapping(
                deps=deps*(1-mid), sig_last=sig_last, q_last=q_last, eps_last=self.eps+deps*mid,
                eps_abs_last=self.eps_abs+np.abs(deps*mid), eps_e_last=self.eps_e+deps*mid,
                yieldValue_last=yieldValue_last)
        else:
            return self.plasticReturnMapping(
                deps=deps, sig_last=self.sig, q_last=self.q, eps_last=self.eps,
                eps_abs_last=self.eps_abs, eps_e_last=self.eps_e, yieldValue_last=self.yieldValue)

    def plasticReturnMapping(
            self, deps, sig_last, q_last, eps_last, eps_abs_last, eps_e_last, yieldValue_last):
        # if np.trace(sig_trial) / 2. < 0:
        #     print()
        dfdsig, dgdsig = self.dfdsig(sig=sig_last)
        temp1 = np.einsum('ij, ijkl->kl', dfdsig, self.De)
        dlam = (np.einsum('kl, kl->', temp1, deps)+yieldValue_last) / np.einsum('kl, kl->', temp1, dgdsig)
        deps_p = dlam * dgdsig
        deps_e = deps - deps_p
        dsig = np.einsum('ijkl, kl->ij', self.De, deps_e)
        sig = dsig + sig_last

        p = np.trace(sig) / 2.

        # if p < -self.tolerance_abs:
        #     print('p= %.2e' % p)
            # raise RuntimeError('p= %.2e' % p)

        q = get_q_2d(sig)
        yieldValue = self.yieldFunction(p=p, q=q)
        scene = [sig, eps_last + deps, eps_abs_last + np.abs(deps),
                 [self.eps_p + deps_p, p, q, eps_e_last + deps_e, yieldValue]]

        if self.explicitFlag:
            return sig, scene
        else:
            raise
            return sig, self.De, scene

    def yieldFunction(self, p, q):
        f = q - self.M * p
        return f

    def dfdsig(self, sig):
        dpdsig = np.eye(2)*0.5
        if self.q != 0.:
            dqdsig = 2. * (sig - self.kronecker * np.trace(sig) / 2.) / get_q_2d(sig)
        else:
            dqdsig = np.array([[1., 1.], [1., 1.]])
        dfdsig = -self.M * dpdsig + dqdsig
        dgdsig = -self.M_dilation*dpdsig + dqdsig
        return dfdsig, dgdsig

    def get_current_scene(self):
        scene = [self.sig, self.eps, self.eps_abs,
                 [self.eps_p, self.p, self.q, self.eps_e, self.yieldValue]]
        return scene

    def transformSplit(self, deps):
        dsig = np.einsum('ijkl, kl->ij', self.De, deps)
        minn, maxx = 0., 1.
        yieldValue = 1e5
        iteration = 0
        while abs(yieldValue) > self.tolerance_abs:
            mid = 0.5 * (minn + maxx)
            q_mid = get_q_2d(sig=dsig * mid + self.sig)
            p_mid = np.trace(dsig * mid + self.sig)/2.
            yieldValue = self.yieldFunction(p=p_mid, q=q_mid)
            if yieldValue > 0:
                maxx = mid
            else:
                minn = mid
            iteration += 1
            if iteration > 20:
                raise RuntimeError('Please check the q_min=%.e q_max=%.e while q_yield=%.e' %
                                   (self.q, get_q_2d(sig=self.sig + dsig), self.yield_stress0))
        return mid, dsig * mid, q_mid, yieldValue

    def update_internal(self, eps_p, p, q, eps_e, yieldValue):
        self.eps_p, self.p, self.q, self.eps_e, self.yieldValue = eps_p, p, q, eps_e, yieldValue

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
    model_demo = druckerPragerSingle()
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
            eps[numg, step] = deps_path[numg, step] + eps[numg, step - 1]
    fig = plt.figure(figsize=[6.4 * 1.8, 4.8])
    fig.add_subplot(121)
    plt.plot(-eps[0, :, 1, 1], -sig_numg[0, :, 0, 0] / 1e6, label=r'$\sigma_{00}$')
    plt.plot(-eps[0, :, 1, 1], -sig_numg[0, :, 1, 1] / 1e6, label=r'$\sigma_{11}$')
    plt.xlabel(r'$\epsilon_{11}$')
    plt.ylabel(r'MPa')
    plt.legend()
    fig.add_subplot(122)
    plt.plot(-eps[0, :, 1, 1], q_numg[0] / 1e6, label='$q$')
    plt.ylabel(r'MPa')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print()
