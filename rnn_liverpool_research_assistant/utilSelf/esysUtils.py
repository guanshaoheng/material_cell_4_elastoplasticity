import matplotlib.pyplot as plt
from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, ReducedSolution, Data, whereNonPositive, wherePositive, inf, Tensor, \
    Function,whereNegative
from esys.escript.pdetools import Projector
from esys.finley import ReadGmsh, Rectangle
from esys.weipa import saveVTK
import os, sys
import numpy as np
from utilSelf.general import echo
from FEMxDEM.explicitSolver import explicitSolver
from FEMxDEM.implicitSolver import ImplicitSolver


def boudaryConditions2D(
        nx_:Data, ny_:Data, nx:Data, ny:Data, mid_ny_:Data,
        fx_:Data, fy_:Data, fx:Data, fy:Data,
        confining=1e5, smoothFlag=True, vel=0.0001):
    q, r, y, Y = Data(), Data(), Data(), Data()
    if smoothFlag:
        q = ny_*[0, 1] + ny*[0, 1] + mid_ny_*[1, 1]
        r = ny*[0, -vel]
    else:
        q = ny_*[1, 1] + ny*[1, 1]
        r = ny*[0, -vel]
    y = fx_*[confining, 0]+fx*[-confining, 0]
    return q, r, y, Y


def boudaryConditions_footing(
        nx_:Data, ny_:Data, nx:Data, ny:Data, by:Data,
        where_load, where_surcharge,
        domain,
        confining=1e5, vel=0.0001, normal_flag=True):
    q, r, y, Y = Data(), Data(), Data(), Data()
    # ----------------------------------------
    # Dirichlet BC, rollers at left and right, fixties at bottom, rigid and rough footing
    q = nx_*[1, 0] + nx*[1, 0] + \
        ny_*[1, 1] + \
        where_load*[1, 1]
    r = nx_*[0, 0] + nx*[0, 0] + \
        ny_*[0, 0] + \
        where_load*[0, -vel]
    # Neumann BC, surcharge at the rest area of the top surface
    if normal_flag:
        normal = domain.getNormal()
        y = -where_surcharge*normal*confining
    else:
        y = where_surcharge*confining*[0, -1]
    return q, r, y, Y


def boudaryCondition_3d(nx_, nx, ny_, ny,
        nz_, nz, fx_, fx, fy_, fy, where_confining, vel, out_normal, confining=1e5, smooth_flag=False):
    if not smooth_flag:
        q = nz_ * [1, 1, 1] + nz * [1, 1, 1]
    else:
        q = nz_ * [1, 1, 1] + nz * [0, 0, 1]
    r = nz_ * [0, 0, 0] + nz * [0, 0, -vel]
    y = where_confining*out_normal*(-confining)
    return q, r, y


def force_length_calculation(sig, domain, where):
    proj = Projector(domain)
    sig_ = proj(sig)
    sig_boundary = interpolate(sig_, FunctionOnBoundary(domain))
    traction = matrix_mult(sig_boundary, domain.getNormal())
    tract_where = traction*where
    force = integrate(tract_where, FunctionOnBoundary(domain))
    length = integrate(where, FunctionOnBoundary(domain))
    return force, length


def get_veps_seps(strain, domain):
    k = kronecker(domain)
    dim = domain.getDim()
    volume_strain = trace(strain)
    dev_strain = symmetric(strain) - volume_strain * k / dim
    shear_strain = sqrt(2 * inner(dev_strain, dev_strain))# \sqrt(3J_2) J_2=0.5s_{ij}s_{ij}
    return volume_strain, shear_strain


def get_boundary_u_traction(domain):
    x = domain.getX()  # nodal coordinate
    bx = FunctionOnBoundary(domain).getX()
    fx_ = whereZero(bx[0] - inf(bx[0]))
    fx = whereZero(bx[0] - sup(bx[0]))
    fy_ = whereZero(bx[1]-inf(bx[1]))
    fy = whereZero(bx[1] - sup(bx[1]))
    nx_ = whereZero(x[0]-inf(x[0]))
    nx = whereZero(x[0] - sup(x[0]))
    ny_ = whereZero(x[1]-inf(x[1]))
    ny = whereZero(x[1] - sup(x[1]))
    mid_ny_ = ny_ * whereZero(x[0] - .5 *sup(x[0]))
    return nx_, nx, ny_, ny, fx_, fx, fy_, fy, mid_ny_


def boudary_explicit_2D_biaxial(domain, du, q, mapFlag=True):
    """
    uSolution: the displacement domain
    u: the displacement at current step
    q: the mask of the displacement (where the displacement is applied)
    mapFlag: use map method or not (default: True)
    """
    x = domain.getX()
    if mapFlag:
        l = sup(x[1])-inf(x[1])
        q_map = x[1]/l
        du_boundary = [0, du]*q_map
    else:
        du_boundary = [0, du]*q
    return du_boundary


def boudary_explicit_2D_hole_plate(domain, du, q, mapFlag=True):
    """
    uSolution: the displacement domain
    u: the displacement at current step
    q: the mask of the displacement (where the displacement is applied)
    mapFlag: use map method or not (default: True)
    """
    x = domain.getX()
    if mapFlag:
        l = sup(x[0])-inf(x[0])
        q_map = x[0]/l
        du_boundary = [du, 0]*q_map
    else:
        du_boundary = [du, 0]*q
    return du_boundary


def boundary_explicit_2D_retaining(domain, du, q, base_height, mapFlag=True):
    x = domain.getX()
    if mapFlag:
        l = sup(x[0])-inf(x[0])
        q_map = wherePositive(x[1]-base_height)*x[0]/l
        du_boundary = [du, 0]*q_map
    else:
        du_boundary = [du, 0]*q
    return du_boundary


def boundary_explicit_2D_footing(domain, du, q, load_width, map_flag=True):
    x = domain.getX()
    if map_flag:
        l = sup(x[1])-inf(x[1])
        q_map = whereNonPositive(x[0]-load_width)*x[1]/l
        du_boundary = [0, du]*q_map
    else:
        du_boundary = [0, du]*q
    return du_boundary


def plot_model(domain, order, integration_order, save_path=None):
    x = np.array(domain.getX().toListOfTuples())
    gx = np.array(Function(domain).getX().toListOfTuples())
    plt.scatter(x=x[:, 0], y=x[:, 1], c='k', s=10)
    plt.scatter(x=gx[:, 0], y=gx[:, 1], c='r', s=6)
    # for i in range(len(x)-1):
    #     plt.plot(x[i])
    plt.axis('equal')
    plt.title('order%d_integration%d' % (order, integration_order))
    plt.tight_layout()
    if save_path:
        fname = os.path.join(save_path, 'model.png')
        plt.savefig(fname, dpi=200)
    else:
        plt.show()
        plt.close()
    return


def plot_model_gauss_points_num(domain, save_path=None):
    gx = np.array(Function(domain).getX().toListOfTuples())
    n_gauss = len(gx)
    index_gauss = list(range(0, n_gauss, 15))
    plt.scatter(x=gx[index_gauss, 0], y=gx[index_gauss, 1], c='r', s=6)
    for i in index_gauss:
        plt.text(x=gx[i, 0], y=gx[i, 1], s='%d' % i)
    plt.axis('equal')
    plt.tight_layout()
    if save_path:
        fname = os.path.join(save_path, 'gauss_points_with_num.svg')
        plt.savefig(fname, dpi=300)
        echo('Fig saved as %s' % fname)
    else:
        plt.show()


def save_explicit_vtk(prob: explicitSolver, save_path, test_name, smooth_flag, step, mode=None):
    domain = prob.domain
    disp = prob.u
    stress = prob.sig
    strain = prob.eps
    acceleration = prob.a
    volume_strain = trace(strain)
    dev_strain = symmetric(strain) - volume_strain * kronecker(domain) / domain.getDim()
    shear = sqrt(2 * inner(dev_strain, dev_strain))

    proj = Projector(domain, reduce=False)
    stress = proj(stress)
    strain = proj(strain)
    shear = proj(shear)
    ke = proj(prob.energy_kinetic)
    pe = proj(prob.energy_potential)
    if mode == '2ml' or 'mixed':
        sig_err = prob.sig_err
        sig_err = proj(sig_err)
        saveVTK(
            os.path.join(save_path, "vtk/%s%s_%d.vtu" % (test_name, '_smooth' if smooth_flag else '_rough', step)),
            domain=domain,
            disp=disp,
            shear=shear,
            strain=strain,
            stress=stress,
            acceleration=acceleration,
            v=prob.v_half,
            ke = ke,
            pe = pe,
            sig_err=sig_err,
        )
    else:
        saveVTK(
            os.path.join(save_path, "vtk/%s%s_%d.vtu" % (test_name, '_smooth' if smooth_flag else '_rough', step)),
            domain=domain,
            disp=disp,
            shear=shear,
            strain=strain,
            stress=stress,
            acceleration=acceleration,
            v=prob.v_half,
            ke = ke,
            pe = pe,
        )


def save_implicit_vtk(prob: ImplicitSolver, save_path, test_name, smooth_flag, step, mode=None):
    domain = prob.domain
    disp = prob.u
    stress = prob.sig
    strain = prob.eps
    volume_strain = trace(strain)
    dev_strain = symmetric(strain) - volume_strain * kronecker(domain) / domain.getDim()
    shear = sqrt(2 * inner(dev_strain, dev_strain))

    proj = Projector(domain, reduce=False)
    stress = proj(stress)
    strain = proj(strain)
    shear = proj(shear)
    volume_strain = proj(volume_strain)
    saveVTK(
        os.path.join(save_path, "vtk/%s%s_%d.vtu" % (test_name, '_smooth' if smooth_flag else '_rough', step)),
        domain=domain,
        disp=disp,
        shear=shear,
        strain=strain,
        stress=stress,
        vol_eps=volume_strain,
    )


if __name__ == "__main__":
    order = 1
    integration_order=1
    mesh_name = 'biaxial_0.05_548'
    domain = ReadGmsh('../biaxial_msh/%s.msh' % mesh_name, numDim=2,
                        order=order, integrationOrder=integration_order)
    plot_model_gauss_points_num(domain, save_path='/home/shguan/simu/biaxial')
