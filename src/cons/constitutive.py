import multiprocessing
import numpy as np
from utilSelf.general import echo, mapMask
from FEMxEPxML.utils_constitutive import tensor2_tensor3, get_elasticMatrix, returnedDatasDecode, \
    tensor2d_to_3d_single, getQEps
from utilSelf.saveGauss import save_loading
from multiprocessing import Pool


class ConstitutiveMask:
    def __init__(self, p0, ndim, explicitFlag, numg, save_path: str, nump: int,
                 cons=None,  name='general', save_flag=False, rho=2650.):
        '''
            Caution: in geo-mechanics, compression is positive (opposite to the general mechanics)
        '''
        self.p0 = p0

        self.nump = nump
        self.ndim = ndim
        self.explicitFlag = explicitFlag
        self.save_flag = save_flag
        self.save_path = save_path
        self.rho = rho
        self.t = 0
        if p0 < 0:
            echo('Caution: in geo-mechanics, '
                 '         compression is positive '
                 '         (opposite to the general mechanics)')
            raise ValueError
        self.numg = numg
        self.eps = np.zeros(shape=(numg, 3, 3))
        self.internal = None
        self.name = name
        self.cons = cons
        if self.cons is not None:
            self.solvers = [self.cons[i].solver for i in range(self.numg)]
            self.sig = np.array([self.cons[i].sig for i in range(self.numg)])
            if not self.explicitFlag:
                self.D = np.array([self.cons[i].D for i in range(self.numg)])
        else:
            pass
        if self.name == 'csuh':
            self.save_loading_mask(sig_geo=self.sig, eps_geo=self.eps[:, :self.ndim, :self.ndim])
        elif self.name == 'vonmises':
            self.save_loading_mask(
                sig_geo=self.sig, eps_geo=self.eps[:, :self.ndim, :self.ndim],
                H_1=[self.cons[i].H for i in range(numg)])
        else:
            pass
        self.t += 1

    def solver(self, deps):
        if len(deps[0]) == 2 and ('misesideal' not in self.name and self.name != 'drucker'):
            deps = tensor2_tensor3(t2=deps)
        param = list(zip(self.solvers, deps))
        if self.nump > 1:
            with Pool(processes=self.nump) as pool:
                datas = pool.map(mapMask, param)
        else:
            datas = []
            for i in range(self.numg):
                datas.append(mapMask(param[i]))
        if self.explicitFlag:
            sig_geo, scenes = returnedDatasDecode(
                explicitFlag=self.explicitFlag, datas=datas, numg=self.numg, name=self.name)
            if self.save_flag and self.t % 10 == 0:
                H_1 = None
                eps_geo = np.array([scenes[i][1][:self.ndim, :self.ndim] for i in range(self.numg)])
                if 'vonmises' in self.name:
                    H_1= np.array([scenes[i][3][3] for i in range(self.numg)])
                self.save_loading_mask(sig_geo=sig_geo, H_1=H_1, eps_geo=eps_geo)
            self.update(scenes=scenes)
            self.t += 1
            return sig_geo
        else:
            sig_geo, D, scenes = returnedDatasDecode(
                explicitFlag=self.explicitFlag, datas=datas, numg=self.numg)
            return sig_geo, D, scenes

    def update(self, scenes):
        for i in range(self.numg):
            self.cons[i].update(*scenes[i])

    def return2initial(self):
        if type(self.cons) is list:
            for i in range(len(self.cons)):
                self.cons[i].return2initial()

    def save_loading_mask(self, sig_geo, eps_geo, H_1=None):
        kwargs = {
            'sig_last': -np.array([self.cons[i].sig for i in range(self.numg)])[:, :self.ndim, :self.ndim],
            'sig': -sig_geo[:, :self.ndim, :self.ndim],
            'eps': -eps_geo[:, :self.ndim, :self.ndim],
            'eps_abs': np.array([self.cons[i].eps_abs for i in range(self.numg)])[:, :self.ndim, :self.ndim],
        }
        if 'vonmises' in self.name:
            '''
                scene = [sig_trial, self.eps + deps, yieldValue, self.eps_p, self.eps_s_p, self.H]
            '''
            kwargs['H_0'] = np.array([self.cons[i].H for i in range(self.numg)])
            kwargs['H_1'] = H_1
        elif 'csuh' in self.name or 'eb' in self.name or 'misesideal' in self.name:
            pass
        else:
            echo('Waiting to add the %s model\'s saving manipulation' % self.name)
            raise
        save_loading(save_path=self.save_path, t=self.t, **kwargs)
        return


class constitutiveSingle:
    def __init__(self, p0: float, ndim: int, explicitFlag=True):
        self.p0 = p0
        self.sig = np.eye(3)*self.p0
        self.eps = np.zeros(shape=[3, 3])
        self.eps_s = 0.
        self.eps_p = np.zeros(shape=[3, 3])
        self.epsvp = 0.
        self.ndim = ndim
        self.eps_abs = np.zeros(shape=[3, 3])
        self.p, self.q = self.p0, 0.
        self.explicitFlag = explicitFlag

    def update(self, sig, eps, eps_abs, internals):
        self.sig = sig
        self.eps = eps
        self.eps_abs = eps_abs
        self.update_internal(*internals)
        # self.eps_s = getQEps(eps)  # shear strain

    def dsigCal(self, D, deps):
        dsig = np.einsum('ijkl, kl->ij', D, deps)
        return dsig

    def prediction(self, deps_numg):
        '''
        :param deps_numg: shape of (numg, steps, ndim, dim)
        :return:
        '''
        sig_numg = []
        for deps in deps_numg:
            sig_pre = []
            for num, i in enumerate(deps):
                sig_temp, scenes_temp = self.solver(deps=-tensor2d_to_3d_single(tensor2d=i))
                sig_pre.append(-sig_temp[:2, :2])
                self.update(*scenes_temp)
            self.return2initial()
            sig_numg.append(sig_pre)
        prediction = np.array(sig_numg)
        return prediction

    def solver(self, deps):
        deps_norm = np.linalg.norm(deps)
        step_num = int(deps_norm / 0.0002*5.) + 1  # 0.0002=2e-4 but we find in the calculation 4e-5 should be better
        if step_num < 1:
            step_num = 1
        step_size = 1. / step_num
        remain, split_num = 1.0, 0
        scece_safe = self.get_current_scene()
        while remain > 0. and split_num < 10:
            if self.explicitFlag:
                sig, scene = self.solver_single(deps=deps * step_size)
            else:
                sig, D, scene = self.solver_single(deps=deps * step_size)
            remain -= step_size
            self.update(*scene)
        if remain == 1.0:
            # raise
            sig, scene = scece_safe[0], scece_safe
            if not self.explicitFlag:
                D = self.De * 0.1
        self.update(*scece_safe)
        if self.explicitFlag:
            return sig, scene
        else:
            return sig, D, scene

    def yieldFunction(self):
        pass

    def hardeningFunction(self):
        pass

    def dfdsig(self):
        pass

    def dgdsig(self):
        pass

    def dfdH(self):
        pass

    def transformSplit(self, deps):
        pass

    def plasticReturnMapping(self, deps):
        pass

    def update_internal(self, internals):
        pass

    def get_current_scene(self, ):
        pass

    def solver_single(self):
        pass

    def return2initial(self):
        pass