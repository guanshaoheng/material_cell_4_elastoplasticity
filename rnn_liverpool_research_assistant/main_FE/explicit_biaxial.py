import os
import sys
import time

import numpy as np
from esys.escript import kronecker, FunctionOnBoundary, Function, \
    whereZero, sup, inf, interpolate, integrate
from esys.finley import ReadGmsh

from FEMxDEM.explicitSolver import explicitSolver
from utilSelf.esysUtils import force_length_calculation, boudary_explicit_2D_biaxial, plot_model, save_explicit_vtk
from utilSelf.general import echo, check_mkdir, getCons, writeLine, \
    get_load_information, get_time_step, explicit_material_constants


argvs = sys.argv
gama = None
n_iter, i = None, 0
while i < len(argvs):
    if argvs[i] == '-n':
        n_iter = int(argvs[i + 1])
    elif argvs[i] == '-gama':
        gama = float(argvs[i + 1])
    i += 1

if n_iter:
    nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_%d' % n_iter
else:
    nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_0'

# get the MPI-pool on the server
test_name = 'biaxial'
mpiFlag = False
nump = 4
explicitFlag = True
smoothFlag = False

# damp = 1e6
damp = 2. * gama * np.sqrt(2650 * 2e7) if gama else 1e6
mode = 'rnn_mises_harden_extract'  # 'mldem' 'dem' 'elastic' 'csuh' 'uh'
                              # 'vonmises' 'mixed' '2ml'
                              # 'misesideal' 'rnn_misesideal'  rnn_misesideal_fc
                              # 'drucker' 'rnn_drucker' rnn_drucker_fc
                              #  mises_harden  rnn_mises_harden  rnn_mises_harden_fc
                              # rnn_mises_harden_epsp rnn_mises_harden_epsp_fc
                              # rnn_mises_harden_extract
# if set the scalar as a number means using this number in scalar,
# else if set scalar=0, then we using the adaptive length of the scalar
scalar = 0

if mode == 'eb':
    fric = 30.
axialStrainGoal = 0.10
time_step = 2e-4
# vel = -0.05
vel = -0.1
mesh_number = 2
nx, ny = mesh_number, mesh_number * 2

save_flag = True if mode == 'dem' or mode == 'mldem' else False
order = 1
integration_order = 1
nn_name_temp = nn_name
input_features = nn_name.split('_')[1]
output_features = nn_name.split('_')[3]
# mesh_name = 'biaxial_162_large'
# mesh_name = 'biaxial_0.1_162'
mesh_name = 'biaxial_0.05_548'
mydomain = ReadGmsh('./meshes/biaxial_msh/%s.msh' % mesh_name, numDim=2,
                    order=order, integrationOrder=integration_order)

lx = 2.0 if "large" in mesh_name else 0.5
ly = 2.0 * lx  # sample size, 50mm by 100mm

dim = mydomain.getDim()
k = kronecker(mydomain)
numg = len(Function(mydomain).getX().toListOfTuples())
safety_coefficient = 0.5

# cal the stable timeStep
"""
    critical time step calculation :
        dt_critical = le_min*sqrt(rho/E)
        
    NOTE: Timestep bigger than 0.1*dt will result in non-convergence.
"""
# load information
# material properties'
if mode == 'csuh':
    # csuh_para_line = "kappa:1.906e-01\tlambdaa:2.142e-01\tN:1.931e+00\tZ:2.743e-01\tocr:3.774e+02\ttheta_degree:1.329e+01"
    # parameters inversed from the dem footing simulation
    # csuh_para_line = "kappa:3.3450e-01\tlambdaa:3.4360e-01\tN:2.6870e+00\tZ:-7.7820e-01\tocr:9.1100e+02\tM:6.2040e-01\tm:3.2080e+01\tnu:2.4340e-01"   # paramters inversed from the exfooting dem
    csuh_para_line = "kappa:3.8330e-02\tlambdaa:6.9290e-02\tN:2.9070e+00\tocr:2.0870e+02\tM:3.9230e-01\tm:1.2430e+01\tnu:2.3910e-01"   # paramters inversed from the exbiaxial dem
else:
    csuh_para_line = None


confining = 1e5   # confining pressure
if (mode == 'mises_harden') or (mode == 'misesideal') or (mode == 'rnn_misesideal') or (mode == 'rnn_misesideal_fc') \
        or (mode == 'rnn_mises_harden') or \
        (mode =='rnn_mises_harden_fc') or (mode == 'rnn_mises_harden_epsp') or (mode == 'rnn_mises_harden_epsp_fc') \
        or (mode == 'rnn_mises_harden_extract'):
    confining = 0.

p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs = \
    explicit_material_constants(
        p0=confining, nn_name=nn_name,
        csuh_para_line=csuh_para_line)
kwargs['input_features'] = input_features
kwargs['output_features'] = output_features
kwargs['b_flag'] = True

if 'rnn' in mode:
    kwargs['scalar'] = scalar

# input args
input_args = sys.argv

n, argv_len = 0, len(input_args)
while n < argv_len:
    if '-ocr' in input_args[n]:
        kwargs['ocr'] = float(input_args[n + 1])
        ocr = kwargs['ocr']
    if '-fric' in input_args[n]:
        kwargs['fric'] = float(input_args[n + 1])
        fric = kwargs['fric']
    if '-mode' in input_args[n]:
        mode = input_args[n + 1]
    n += 1

rateVelocity = vel / ly
out_directory = '../simu/explicit/%s' % test_name
loadInfor = get_load_information(
    out_directory=out_directory, test_name=test_name, mode=mode, smooth_flag=smoothFlag,
    explicit_flag=explicitFlag, time_step=time_step,
    vel=vel, nx=nx, ny=ny, order=order, numg=numg,
    safety_coefficient=safety_coefficient, mesh_name=mesh_name, damp=damp, gama=gama, **kwargs)

# loadInfor += '_deepfc_adaptstep'
# loadInfor += 'scalar1'

kwargs['save_path'] = loadInfor
timeStep = get_time_step(
    lam_2G=lam + 2 * G, rho=rho, element_size=inf(mydomain.getSize()),
    safety_coefficient=safety_coefficient)

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
    'The stable timeStep: %e' % timeStep,
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
prob = explicitSolver(domain=mydomain, timestep=timeStep, cons=cons, loadInfor=loadInfor,
                      save_flag=save_flag, domain_size=lx * ly, damp=damp)

x = mydomain.getX()  # nodal coordinate
bx = FunctionOnBoundary(mydomain).getX()

# fx means the function on boundary, while n means the node
fx_, fx = whereZero(bx[0] - inf(bx[0])), whereZero(bx[0] - sup(bx[0]))
fy_, fy = whereZero(bx[1] - inf(bx[1])), whereZero(bx[1] - sup(bx[1]))
nx_, nx = whereZero(x[0] - inf(x[0])), whereZero(x[0] - sup(x[0]))
ny_, ny = whereZero(x[1] - inf(x[1])), whereZero(x[1] - sup(x[1]))

forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=fy)
fname = os.path.join(loadInfor, 'biaxial_surf.dat')
writeLine(fname=fname, s='AxialStrain forceTop lengthTop volumeStrain aMax iterNum ke pe work workout\n', mode='w')
# writeLine(fname=fname, s='%.6f %.1f %.6f %.6f %.5f %d' % (0., -5e3, lengthTop, 0., 0., 0) + '\n', mode='a')

# Dirichlet BC positions, smooth at bottom and top, fixed at the center of bottom
if smoothFlag:
    Dbc = ny * [0, 1] + \
          ny_ * [0, 1] + whereZero(x[0] - .5 * lx) * [1, 1]  # bind the mind point in order not to slide in x direction
else:
    Dbc = ny * [1, 1] + ny_ * [1, 1]  # bind the mind point in order not to slide in x direction

# Dirichlet BC values NOTE: if use the explicit format, the boundary value is for the accelerations.
# On the top and the bottom surface, the accelerations on direction y should be 0
Vbc = nx * [0, 0] + \
      nx_ * [0, 0]  # bind the mind point in order not to slide in x direction
Nbc = fx_ * [confining, 0] + \
      fx * [-confining, 0]
prob.initialize(y=Nbc, q=Dbc, r=Vbc, D=kronecker(mydomain) * rho)

"""
    Rate loading is applied on the axial direction.

    The loading rate -100, means du_per_step = -100*timestep*ly = -0.000222907. 
"""
loadStepMax = abs(int(axialStrainGoal / rateVelocity / timeStep))
u_loaded = 0.
i, num_step = 0, 200
time_start = time.time()
n, t = 0, 0
work, work_out = 0., 0.
factor = lx * ly / np.sum((prob.domain.getSize() ** 2).toListOfTuples())
x = prob.domain.getX()
fx_double = fx + fx_
while abs(u_loaded) < ly * axialStrainGoal:  # apply 100 load steps
    # if prob.cons.cons is not None:
    #     lam_2G = np.max([prob.cons.cons[i].D[0, 0, 0, 0] for i in range(numg)])
    # else:
    #     lam_2G = lam+2.*G
    # element_size = inf(prob.domain.getSize())
    # time_step = get_time_step(
    #     lam_2G=lam_2G, rho=rho, element_size=inf(mydomain.getSize()), safety_coefficient=safety_coefficient)

    du = rateVelocity * ly * time_step
    u_loaded += du
    forceTop, lengthTop = force_length_calculation(sig=prob.sig, domain=prob.domain, where=fy)
    work += du * forceTop[1]
    du_coord_ = interpolate((prob.domain.getX() - x), FunctionOnBoundary(prob.domain))
    du_coord = interpolate((prob.domain.getX() - x), FunctionOnBoundary(prob.domain))
    work_out += \
        integrate(fx_ * du_coord * [-confining, 0.] + fx * du_coord * [confining, 0.], FunctionOnBoundary(prob.domain))[
            0]
    x = prob.domain.getX()

    duBoundary = boudary_explicit_2D_biaxial(domain=prob.domain, du=du, q=ny, mapFlag=True)

    # renew the domain according to the first boundary condition
    prob.solveExplicit(n=n, duBoundary=duBoundary, dt=time_step)
    # save the the information to file loadInformation/results

    if abs(u_loaded) >= i * ly * axialStrainGoal / num_step:
        i += 1
        axialStrainCurrent = u_loaded / ly
        aMax = sup(prob.a)
        total_volume_strain = np.average(np.array(prob.volume.toListOfTuples()))

        ke = np.sum(np.array((prob.element_size * prob.energy_kinetic).toListOfTuples()))
        pe = np.sum(np.array((prob.element_size * prob.energy_potential).toListOfTuples()))

        # save the macro information
        writeLine(fname=fname,
                  s='%.6f %.1f %.6f %.6f %.5f %d %.3f %.3f %.3f %.3f' % \
                    (axialStrainCurrent, forceTop[1], lengthTop, (total_volume_strain), aMax, 0, ke, pe, work,
                     work_out) + '\n',
                  mode='a')
        echo(
            'Loading step # %d \taxialCurrentStrain: %e/%e  time_increment %e, Topforce: %.3e maxA: %.3e Time consuming: %.2e mins' %
            (n, axialStrainCurrent, axialStrainGoal, time_step, forceTop[1], aMax,
             (time.time() - time_start) / 60.))
        save_explicit_vtk(prob=prob, step=n, save_path=loadInfor, test_name=test_name, smooth_flag=smoothFlag,
                          mode=mode)
    n += 1

time_elapse = time.time() - time_start
writeLine(fname=fname, mode='a', s="# Elapsed time in hours: %.2e\n" % (time_elapse / 3600.))
