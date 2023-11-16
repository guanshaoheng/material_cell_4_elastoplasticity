from esys.escript import Vector, Solution, grad, trace, transpose, kronecker, \
    length, interpolate, FunctionOnBoundary, matrix_mult, \
    whereZero, whereNegative, integrate, sup, inf, sqrt, inner, L2, \
    Tensor, Tensor4, Function, Data, Scalar,sign, symmetric, interpolate
from esys.escript.pdetools import Locator, Projector
from esys.escript.linearPDEs import LinearPDE, SolverOptions
import numpy as np
from FEMxDEM.escriptSolver import escriptSolver
from utilSelf.saveGauss import save_loading
from utilSelf.general import echo


class explicitSolver(escriptSolver):
    def __init__(self, domain, cons, loadInfor, timestep, domain_size=0.5, save_flag=False, damp=None):
        explicitFlag = True
        escriptSolver.__init__(self, domain=domain, cons=cons, explicitFlag=explicitFlag,
            timestep=timestep, loadInfor=loadInfor)
        self.rho = self.cons.rho
        self.damping = damp
        self.pde.getSolverOptions().setSolverMethod(SolverOptions.HRZ_LUMPING)
        self.v_half = Vector(0., Solution(self.domain))
        self.a = Vector(0., Solution(self.domain))
        self.t = 0.
        self.save_flag = save_flag
        if self.save_flag:
            self.save_loading_mask(t=0, sig_data=self.sig, deps=Tensor(0., Function(self.domain)))
        # energy calculation
        self.v_gauss = interpolate(self.v_half, Function(self.domain))   # -> Gauss points
        temp = 0.5*self.rho*self.domain.getSize()*self.v_gauss**2  # -> Gauss points
        self.energy_kinetic = temp[0] + temp[1]
        temp = self.sig*self.eps  # -> Gauss points
        self.energy_potential = temp[0, 0] + temp[0, 1]+ temp[1, 0]+temp[1, 1]
        self.element_size = domain_size*self.domain.getSize()**2/np.sum((self.domain.getSize()**2).toListOfTuples())

    def solveExplicit(self, duBoundary, n, dt):
        # ... set initial values ...
        # central difference rule
        if n == 0:
            self.v_half = self.v_half + self.a * dt / 2.
        else:
            self.v_half = self.v_half + self.a * dt
        du = self.v_half * dt  # displacement due to the unbalanced force
        du_total = du+duBoundary
        deps = symmetric(grad(du_total))
        if self.cons.name == '2ml' or self.cons.name == 'mixed':
            self.sig, self.sig_err = self.stressSolver(deps)
        else:
            self.sig = self.stressSolver(deps)
        self.pde.setValue(X=-(self.sig))
        self.a = self.pde.getSolution()
        if self.damping:
            self.a = self.a - self.v_half*self.damping/self.rho
        if self.save_flag and n % 10 == 1:
            self.save_loading_mask(t=n, sig_data=self.sig, deps=deps)
        self.update_model(du=du+duBoundary, deps=deps)

        # renew the global variables
        self.t += dt
        n += 1

    def save_loading_mask(self, t, sig_data, deps, iterate=None):
        save_dict = {
            'sig': sig_data,
            'eps': self.eps + deps, 'eps_abs': self.eps_abs}
        save_loading(t=t, iter=iterate, save_path=self.savePath, **save_dict)

    def update_model(self, du, deps):
        self.u = self.u+du
        self.eps = self.eps+deps
        self.volume = trace(self.eps)
        self.domain.setX(self.domain.getX() + du)
        self.eps_abs += deps*sign(deps)
        self.v_gauss = interpolate(self.v_half, Function(self.domain))  # -> Gauss points
        temp = 0.5*self.rho*self.v_gauss**2  # -> Gauss points
        self.energy_kinetic = temp[0] + temp[1]
        temp = self.sig*deps  # -> Gauss points
        self.energy_potential += (temp[0, 0] + temp[0, 1]+ temp[1, 0]+temp[1, 1])
        self.element_size *= (trace(deps)+1)
