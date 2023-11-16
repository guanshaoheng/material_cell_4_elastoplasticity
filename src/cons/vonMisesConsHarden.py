import multiprocessing
import time
import numpy as np
from constitutive import ConstitutiveMask, constitutiveSingle
from utils_constitutive import get_elastic_matrix_plain_stress, get_q_2d, voigt_2_tensor
from utilSelf.general import echo


class MisesHardenConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, save_path: str, rho: float,
                 p0=0., nu=0.2, E=2e6, ndim=2, save_flag=False, nump=4):
        self.cons = [
            MisesHardenSingle(
                explicitFlag=explicitFlag,
                p0=p0, nu=nu, E=E, ndim=2) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, p0=p0, ndim=ndim, cons=self.cons, rho=rho,
            explicitFlag=explicitFlag, nump=nump, numg=numg, name='drucker', save_path=save_path,
            save_flag=save_flag)


class MisesHardenSingle(constitutiveSingle):
    '''
     used for the plain stress problem.

     NOTE: all the of elastic computaion are under plain stress assumption

     Hardening function like:
                    $ H = A ( \| \epsilon^p \| + \epsilon_0 )^B $
    '''

    def __init__(
            self,
            p0=0., nu=0.2, E=2e6, yield_stress0=1e5,
            explicitFlag=True, ndim=2):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag)
        self.nu = nu
        self.E = E
        self.K = self.E / (2 * (1. - self.nu))
        self.G = self.E / 2 / (1 + self.nu)
        self.De = get_elastic_matrix_plain_stress(E=self.E, nu=self.nu)
        self.yield_stress0 = yield_stress0
        # hardening parameters
        self.A = 0.2 * E
        self.B = 0.5
        self.epsilon_0 = np.power(self.yield_stress0/self.A, 1./self.B)

        # computation operator
        self.kronecker = np.eye(2)

        # calculation setting
        self.tolerance_abs = 10

        # state
        self.p0 = p0
        self.sig = np.eye(2) * self.p0
        self.eps = np.zeros(shape=[2, 2])
        self.eps_s = 0.
        self.eps_p_norm = 0.
        self.eps_p = np.zeros(shape=[2, 2])
        self.eps_e = np.zeros(shape=[2, 2])
        self.epsvp = 0.
        self.ndim = ndim
        self.eps_abs = np.zeros(shape=[2, 2])
        self.p, self.q = self.p0, 0.
        self.explicitFlag = explicitFlag
        if not explicitFlag:
            raise ValueError('Currently the drucker model is not implemented for implicit solver!')
        self.H = self.hardeningFunction(eps_p=self.eps_p)
        self.yieldValue = self.yieldFunction(q=self.q, H = self.H)

    def prediction_multiprocess(self, deps_numg, num_p=4, num_per=10):
        n_samples, start_time = len(deps_numg), time.time()
        sig_numg, q_numg, eps_e_numg, eps_p_numg, H_numg = [], [], [], [], []
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
                H_numg.append(temp[4])
            echo('Generated_samples_number=[%d/%d] consumed_time=%e(mins)' %
                 (min((i + 1) * num_p * num_per, n_samples), n_samples, (time.time() - start_time) / 60.))
        return np.array(sig_numg), np.array(q_numg), np.array(eps_p_numg), np.array(eps_e_numg), np.array(H_numg)

    def prediction_numg(self, deps_numg, verboseFlag=False):
        '''
                :param deps_numg: shape of (numg, steps, ndim, dim)
                :return:
        '''
        sig_numg, q_numg, eps_e_numg, eps_p_numg, H_numg = [], [], [], [], []
        num_numg, total_num_numg, start_time = 0, len(deps_numg), time.time()
        for deps in deps_numg:
            temp = self.forward(deps)
            sig_numg.append(temp[0])
            q_numg.append(temp[1])
            eps_e_numg.append(temp[2])
            eps_p_numg.append(temp[3])
            H_numg.append(temp[4])
            num_numg += 1
            if (num_numg + 1) % 10 == 0 and verboseFlag:
                echo('Generated_samples_number=[%d/%d] consumed_time=%e(mins)' %
                     (num_numg + 1, total_num_numg, (time.time() - start_time) / 60.))
        return np.array(sig_numg), np.array(q_numg), np.array(eps_p_numg), \
               np.array(eps_e_numg), np.array(H_numg)

    def forward(self, deps):
        '''

        :param deps: in shape of [steps, 2, 2]  input compression is negative
        :return:
        '''
        sig_pre, q_pre, eps_e_pre, eps_p_pre, H_pre = [], [], [], [], []
        for num, i in enumerate(deps):
            sig_temp, scenes_temp = self.solver(deps=-i)
            q_pre.append(scenes_temp[3][2])
            eps_p_pre.append(-scenes_temp[3][0])
            eps_e_pre.append(-scenes_temp[3][3])
            H_pre.append(scenes_temp[3][5])
            sig_pre.append(-sig_temp)             # output stress, compression is negtive
            self.update(*scenes_temp)
        self.return2initial()
        return [sig_pre, q_pre, eps_e_pre, eps_p_pre, H_pre]

    def hardeningFunction(self, eps_p):
        H = self.A * (self.epsilon_0 + np.linalg.norm(eps_p))**self.B
        return H

    def dHdeps_p(self):
        dHdnorm = self.A*self.B*(self.epsilon_0 + self.eps_p_norm)**(self.B-1.)
        if self.eps_p_norm == 0.:
            dnormdepsp = np.array([[1., 1.], [1., 1.]])
        else:
            dnormdepsp = self.eps_p/self.eps_p_norm
        dHdepsp = dHdnorm * dnormdepsp
        return dHdepsp

    def solver_single(self, deps):
        deps_v = np.trace(deps)
        de = (deps - deps_v / 2. * self.kronecker)
        dsig = de * 2. * self.G + self.kronecker * deps_v * self.K
        # sig_trial_residual = (deps-deps_v/2.*self.kronecker)*self.G + self.kronecker*deps_v*self.K - \
        #                      np.einsum('ijkl, kl->ij', self.De, deps)
        sig_trial = self.sig + dsig
        q_trial = get_q_2d(sig=sig_trial)
        # yield judgement
        f = self.yieldFunction(q=q_trial, H=self.H)
        if f <= 0:
            scene = [sig_trial, self.eps + deps, self.eps_abs + np.abs(deps),
                     [self.eps_p, np.trace(sig_trial) / 2., q_trial, self.eps_e + deps, self.yieldValue, self.H]]
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
        dqdsig = self.dqdsig(sig=sig_last)
        dHdepsp = self.dHdeps_p()
        temp1 = np.einsum('ij, ijkl->kl', dqdsig, self.De)
        upon = np.einsum('kl, kl->', temp1, deps) + yieldValue_last
        below = np.einsum('kl, kl->', temp1, dqdsig) + np.einsum('ij, ij->', dHdepsp, dqdsig)
        dlam = upon/below
        deps_p = dlam * dqdsig
        deps_e = deps - deps_p
        dsig = np.einsum('ijkl, kl->ij', self.De, deps_e)
        sig = dsig + sig_last

        p = np.trace(sig) / 2.

        q = get_q_2d(sig)
        H = self.hardeningFunction(eps_p=self.eps_p+deps_p)
        yieldValue = self.yieldFunction(q=q, H=H)
        scene = [sig, eps_last + deps, eps_abs_last + np.abs(deps),
                 [self.eps_p + deps_p, p, q, eps_e_last + deps_e, yieldValue, H]]

        if self.explicitFlag:
            return sig, scene
        else:
            raise
            return sig, self.De, scene

    def yieldFunction(self, q, H):
        f = q - self.H
        return f

    def dqdsig(self, sig):
        if self.q != 0.:
            dqdsig = 2. * (self.sig - self.kronecker * np.trace(sig) / 2.) / self.q
        else:
            dqdsig = np.array([[1., 1.], [1., 1.]])
        return dqdsig

    def get_current_scene(self):
        scene = [self.sig, self.eps, self.eps_abs,
                 [self.eps_p, self.p, self.q, self.eps_e, self.yieldValue, self.H]]
        return scene

    def transformSplit(self, deps):
        dsig = np.einsum('ijkl, kl->ij', self.De, deps)
        minn, maxx = 0., 1.
        yieldValue = 1e5
        iteration = 0
        while abs(yieldValue) > self.tolerance_abs*5.:
            mid = 0.5 * (minn + maxx)
            q_mid = get_q_2d(sig=dsig * mid + self.sig)
            yieldValue = self.yieldFunction(q=q_mid, H=self.H)
            if yieldValue > 0:
                maxx = mid
            else:
                minn = mid
            iteration += 1
            if iteration > 25:
                raise RuntimeError(
                    'Please check the yield_0=%.2e q_min=%.2e q_max=%.2e while yieldValue=%.2e at mid=%.2e' %
                                   (self.yieldValue, self.q, get_q_2d(sig=self.sig + dsig), yieldValue, mid))
        return mid, dsig * mid, q_mid, yieldValue

    def update_internal(self, eps_p, p, q, eps_e, yieldValue, H):
        self.eps_p, self.p, self.q, self.eps_e, self.yieldValue = eps_p, p, q, eps_e, yieldValue
        self.H = H

    def return2initial(self):
        self.sig = np.eye(2) * self.p0
        self.eps = np.zeros(shape=[2, 2])
        self.eps_s = 0.
        self.eps_p = np.zeros(shape=[2, 2])
        self.eps_e = np.zeros(shape=[2, 2])
        self.epsvp = 0.
        self.eps_abs = np.zeros(shape=[2, 2])
        self.p, self.q = self.p0, 0.
        self.H = self.hardeningFunction(eps_p=self.eps_p)
        self.yieldValue = self.yieldFunction(q = self.q, H = self.H)


if __name__ == '__main__':
    model_demo = MisesHardenSingle()
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
