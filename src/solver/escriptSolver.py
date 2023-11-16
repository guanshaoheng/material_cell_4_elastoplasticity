from esys.escript import Vector, Solution, grad, trace, transpose, kronecker, \
    length, interpolate, FunctionOnBoundary, matrix_mult, \
    whereZero, whereNegative, integrate, sup, inf, sqrt, inner, L2, \
    Tensor, Tensor4, Function, Data, Scalar
from esys.escript.pdetools import Locator, Projector
from esys.escript.linearPDEs import LinearPDE, SolverOptions
import numpy as np
from utilSelf.saveGauss import save_loading
import os


class escriptSolver:
    def __init__(self, domain, cons, loadInfor, explicitFlag, timestep=None):
        self.domain = domain
        self.cons = cons
        self.explicitFlag = explicitFlag
        self.timestep = timestep
        if explicitFlag:
            if timestep is None:
                raise ValueError('In explicit mode, please input the timestep')
        # self.pool = pool
        self.savePath = loadInfor
        self.ndim = self.domain.getDim()
        self.pde = LinearPDE(self.domain,
                               numEquations=self.ndim,
                               numSolutions=self.ndim)
        # self.pde.setReducedOrderOn()
        # NOTE: this is important
        if not self.explicitFlag:
            self.pde.getSolverOptions().setSolverMethod(SolverOptions.DIRECT)
            self.pde.setSymmetryOn()
        self.eps = Tensor(0, Function(self.domain))
        self.eps_abs = Tensor(0., Function(self.domain))
        self.numG = len(self.eps.toListOfTuples())
        self.sig = self.setStressTensor(
            sig=-self.cons.sig[:, :self.ndim, :self.ndim])
        self.sig_err = Tensor(0., Function(self.domain))
        self.pde.setValue(X=-self.sig)
        self.u = Vector(0., Solution(self.domain))
        self.volume = Scalar(1., Solution(self.domain))
        if not self.explicitFlag:  # in the implicit mode, we have to set the material matrix
            self.D = self.setMaterialMatrix(
                Dep=np.array(self.cons.D[:, :self.ndim, :self.ndim, :self.ndim, :self.ndim]))
            self.pde.setValue(A=self.D)

    def initialize(self, Y=Data(), y=Data(), q=Data(),
                   r=Data(), D=Data()):
        """
        initialize the model for each time step, e.g. assign parameters
        :param Y: type vector, body force on FunctionSpace, e.g. gravity
        :param y: type vector, boundary traction on FunctionSpace (FunctionOnBoundary)
        :param q: type vector, mask of location for Dirichlet boundary
        :param r: type vector, specified displacement for Dirichlet boundary
        """
        self.pde.setValue(Y=Y, y=y, q=q, r=r, D=D)

    def solve(self, n):
        pass

    def solveExplicit(self, duBoundary, n, t):
        pass

    def setStressTensor(self, sig):
        stress = Tensor(0, Function(self.domain))
        if self.ndim==2:
            for i in range(self.numG):
                stress.setValueOfDataPoint(i, sig[i][:self.ndim, :self.ndim])
        else:
            for i in range(self.numG):
                stress.setValueOfDataPoint(i, sig[i])
        return stress

    def setMaterialMatrix(self, Dep):
        D = Tensor4(0, Function(self.domain))
        if self.ndim == 2:
            for i in range(self.numG):
                D.setValueOfDataPoint(i, Dep[i][:self.ndim, :self.ndim, :self.ndim, :self.ndim])
        else:
            for i in range(self.numG):
                D.setValueOfDataPoint(i, Dep[i])
        return D

    def stressSolver(self, deps):
        # NOTE: in geo-mechanical, compression is positive while stretching is negative
        deps = -np.array(deps.toListOfTuples())
        if self.explicitFlag:
            if self.cons.name == '2ml' or self.cons.name == 'mixed':
                sig_geo, sig_err = self.cons.solver(deps)
                return self.setStressTensor(-sig_geo), self.setStressTensor(sig_err)
            else:
                sig_geo = self.cons.solver(deps)
                return self.setStressTensor(-sig_geo)
        else:
            sig_geo, D, scenes = self.cons.solver(deps)
            return self.setStressTensor(-sig_geo), self.setMaterialMatrix(Dep=D), scenes

    def updateCons(self, scenes):
        '''
            https://stackoverflow.com/questions/53878553/why-multiprocessing-pool-cannot-change-global-variable

            Because you are using multiprocessing.Pool your program runs in multiple processes. Each process has
             its own copy of the global variable, each process modifies its own copy of the global variable, and
             when the work is finished each process is terminated. The master process never modified its copy of
             the global variable.
        '''
        # so we can not use the multiprocess to renew the state of the cons
        # param = list(zip([self.cons[i].update for i in range(self.numG)], scenes))
        # self.pool.map(updateMask, param)
        if type(self.cons) is not list:
            self.cons.update(scenes)
        else:
            for i in range(self.numG):
                self.cons[i].update(*scenes[i])

    def returnedDatasDecode(self, datas):
        sig_geo = []
        if self.explicitFlag:
            scenes = []
            for i in range(self.numG):
                sig_geo.append(datas[i][0])
                scenes.append(datas[i][1])
            sig_data = self.setStressTensor(-np.array(sig_geo))
            return sig_data, scenes
        else:  # implicit
            D = []
            scenes = []
            for i in range(self.numG):
                sig_geo.append(datas[i][0])
                D.append(datas[i][1])
                scenes.append(datas[i][2])
            sig_data = self.setStressTensor(-np.array(sig_geo))
            D_data = self.setMaterialMatrix(D)
            return sig_data, D_data, scenes

    def getAbsStrain(self, D):
        # used in the 2D calculations
        current_strain_abs = D.copy()
        D_list = D.toListOfTuples()
        for i, strain_point in enumerate(D_list):
            current_strain_abs.setValueOfDataPoint(
                i, [[abs(strain_point[0][0]),
                     0.5 * abs(strain_point[0][1] + strain_point[1][0])],
                    [0.5 * abs(strain_point[0][1] + strain_point[1][0]),
                     abs(strain_point[1][1])]])
        return current_strain_abs

    def getFrobeniusNorm(self, D):
        # eye = np.eye(2)
        D = np.array(D.toListOfTuples())
        # frobeniusNormIncrement1 = np.sqrt(np.einsum("ijk, ijl, kl->i", D, D, eye))
        frobeniusNormIncrement = np.sqrt(np.array([np.sum(i*i) for i in D]))
        return frobeniusNormIncrement



