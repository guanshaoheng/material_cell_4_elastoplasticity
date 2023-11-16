import os
import sys
import time

import numpy as np
from esys.escript import kronecker, FunctionOnBoundary, Function, \
    whereZero, sup, inf, interpolate, integrate, Data, whereNegative
from esys.finley import ReadGmsh

from FEMxDEM.explicitSolver import explicitSolver
from utilSelf.esysUtils import force_length_calculation, boudary_explicit_2D_hole_plate, plot_model, save_explicit_vtk
from utilSelf.general import echo, check_mkdir, getCons, writeLine, \
    get_load_information, get_time_step, explicit_material_constants


# get the MPI-pool on the server
test_name = 'hole-plate'
mpiFlag = False
nump = 4
explicitFlag = True
smoothFlag = False

# damp = 1e6
gama = 1.0
damp = 2. * gama * np.sqrt(2650 * 2e7)
mode = 'rnn_mises_harden_extract'  # 'mldem' 'dem' 'elastic' 'csuh' 'uh'
                              # 'vonmises' 'mixed' '2ml'
                              # 'misesideal' 'rnn_misesideal'  rnn_misesideal_fc
                              # 'drucker' 'rnn_drucker' rnn_drucker_fc
                              #  mises_harden  rnn_mises_harden  rnn_mises_harden_fc
                              # rnn_mises_harden_epsp rnn_mises_harden_epsp_fc
                              # rnn_mises_harden_extract

axialStrainGoal = 0.10
vel = 0.1
mesh_number = 2
nx, ny = mesh_number, mesh_number * 2

save_flag = True
order = 1
integration_order = 1
mesh_name = 'hole-plate'
mydomain = ReadGmsh('./meshes/hole-plate_msh/%s.msh' % mesh_name, numDim=2,
                    order=order, integrationOrder=integration_order)\

ly = 0.1
lx = 0.1

dim = mydomain.getDim()
k = kronecker(mydomain)
numg = len(Function(mydomain).getX().toListOfTuples())
safety_coefficient = 0.5

# cal the stable timeStep

confining = 0.

p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        p0=confining, nn_name=None)

time_step = get_time_step(
    lam_2G=lam + 2 * G, rho=rho, element_size=inf(mydomain.getSize()),
    safety_coefficient=safety_coefficient)

rateVelocity = vel / ly
out_directory = '../simu/explicit/%s' % test_name

kwargs['scalar'] = None
loadInfor = get_load_information(
    out_directory=out_directory, test_name=test_name, mode=mode, smooth_flag=smoothFlag,
    explicit_flag=explicitFlag, time_step=time_step,
    vel=vel, nx=nx, ny=ny, order=order, numg=numg,
    safety_coefficient=safety_coefficient, mesh_name=mesh_name, damp=damp, **kwargs)

kwargs['save_path'] = loadInfor

check_mkdir(
    out_directory,
    loadInfor,
    os.path.join(loadInfor, 'vtk'),
    os.path.join(loadInfor, 'added_points'),
    os.path.join(loadInfor, 'iteration_gauss'), )

plot_model(domain=mydomain, order=order, integration_order=integration_order, save_path=loadInfor)

# ---------------------- echo ------------------------
echo(
    loadInfor,
    'The stable timeStep: %e' % time_step,
    'Solving mode %s' % ('single' if nump == 0 else ('python multiprocess %d' % nump)),
    'Explicit-%s' % (mode),
    'CWD %s' % loadInfor,
    'Solution Mode: \t%s' % ('Explicit' if explicitFlag else 'Implicit'),
    'lx:\t%.5f' % lx + '\t nx:\t%d' % nx,
    'ly:\t%.5f' % ly + '\t ny:\t%d' % ny,
    'confing:\t%e' % confining,
    'Axial strain:\t%f' % axialStrainGoal)

cons = getCons(mode, numg=numg, nump=nump, explicitFlag=explicitFlag, ndim=2, **kwargs)
# prob
prob = explicitSolver(domain=mydomain, timestep=time_step, cons=cons, loadInfor=loadInfor,
                      save_flag=save_flag, domain_size=lx * ly, damp=damp)

x = mydomain.getX()  # nodal coordinate
bx = FunctionOnBoundary(mydomain).getX()

# fx means the function on boundary, while n means the node
fx_, fx = whereZero(bx[0] - inf(bx[0])), whereZero(bx[0] - sup(bx[0]))
fy_, fy = whereZero(bx[1] - inf(bx[1])), whereZero(bx[1] - sup(bx[1]))
nx_, nx = whereZero(x[0] - inf(x[0])), whereZero(x[0] - sup(x[0]))
ny_, ny = whereZero(x[1] - inf(x[1])), whereZero(x[1] - sup(x[1]))  # bottom & top

forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=fx)
fname = os.path.join(loadInfor, 'biaxial_surf.dat')
writeLine(fname=fname, s='AxialStrain forceTop lengthTop volumeStrain aMax iterNum ke pe work workout\n', mode='w')
# writeLine(fname=fname, s='%.6f %.1f %.6f %.6f %.5f %d' % (0., -5e3, lengthTop, 0., 0., 0) + '\n', mode='a')

# Dirichlet BC positions, smooth at bottom and top, fixed at the center of bottom

Dbc = nx * [1, 0] + nx_ * [1, 0] + whereZero(x[1] - 0.5*ly) * nx_*[1, 1] + ny_ * [0, 1]

# Dirichlet BC values NOTE: if use the explicit format, the boundary value is for the accelerations.
# On the top and the bottom surface, the accelerations on direction y should be 0
Vbc = Data()
Nbc = Data()
prob.initialize(y=Nbc, q=Dbc, r=Vbc, D=kronecker(mydomain) * rho)

"""
    Rate loading is applied on the axial direction.

    The loading rate -100, means du_per_step = -100*timestep*ly = -0.000222907. 
"""

loadStepMax = abs(int(axialStrainGoal / rateVelocity / time_step))
u_loaded = 0.
i, num_step = 0, 200
time_start = time.time()
n, t = 0, 0
work, work_out = 0., 0.
factor = lx * ly / np.sum((prob.domain.getSize() ** 2).toListOfTuples())
x = prob.domain.getX()
while abs(u_loaded) < lx * axialStrainGoal:  # apply 100 load steps

    du = rateVelocity * lx * time_step
    u_loaded += du
    forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=fx)
    work += du * forceTop[0]
    du_coord_ = interpolate((prob.domain.getX() - x), FunctionOnBoundary(prob.domain))
    du_coord = interpolate((prob.domain.getX() - x), FunctionOnBoundary(prob.domain))
    work_out += \
        integrate(fx_ * du_coord * [-confining, 0.] + fx * du_coord * [confining, 0.], FunctionOnBoundary(prob.domain))[
            0]
    x = prob.domain.getX()

    duBoundary = boudary_explicit_2D_hole_plate(domain=prob.domain, du=du, q=ny, mapFlag=True)

    # renew the domain according to the first boundary condition
    prob.solveExplicit(n=n, duBoundary=duBoundary, dt=time_step)
    # save the the information to file loadInformation/results

    if abs(u_loaded) >= i * lx * axialStrainGoal / num_step:
        i += 1
        axialStrainCurrent = u_loaded / lx
        aMax = sup(prob.a)
        total_volume_strain = np.average(np.array(prob.volume.toListOfTuples()))

        ke = np.sum(np.array((prob.element_size * prob.energy_kinetic).toListOfTuples()))
        pe = np.sum(np.array((prob.element_size * prob.energy_potential).toListOfTuples()))

        # save the macro information
        writeLine(fname=fname,
                  s='%.6f %.1f %.6f %.6f %.5f %d %.3f %.3f %.3f %.3f' % \
                    (axialStrainCurrent, forceTop[0], lengthTop, (total_volume_strain), aMax, 0, ke, pe, work,
                     work_out) + '\n',
                  mode='a')
        echo(
            'Loading step # %d \taxialCurrentStrain: %e/%e  time_increment %e, Topforce: %.3e maxA: %.3e Time consuming: %.2e mins' %
            (n, axialStrainCurrent, axialStrainGoal, time_step, forceTop[0], aMax,
             (time.time() - time_start) / 60.))
        save_explicit_vtk(prob=prob, step=n, save_path=loadInfor, test_name=test_name, smooth_flag=smoothFlag,
                          mode=mode)
    n += 1

time_elapse = time.time() - time_start
writeLine(fname=fname, mode='a', s="# Elapsed time in hours: %.2e\n" % (time_elapse / 3600.))
