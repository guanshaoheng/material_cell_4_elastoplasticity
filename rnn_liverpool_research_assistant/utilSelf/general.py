import os
import pickle
import sys

import numpy as np


def check_mkdir(*args):
    for path in args:
        if not os.path.exists(path):
            os.mkdir(path)
            print('\t\tDirectory made as %s' % path)


def get_pool(mpi=False, threads=1):
    """ function to return pool for parallelization
        supporting both MPI (experimental) on distributed
        memory and multiprocessing on shared memory.
    """
    if mpi:  # using MPI
        from FEMxDEM.mpipool import MPIPool
        pool = MPIPool()
        pool.start()
        if not pool.is_master():
            sys.exit(0)
    elif threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads)
    else:
        pool = None
    return pool


def writeLine(fname, s, mode='w'):
    f = open(fname, mode=mode)
    f.write(s)
    f.close()


def echo(*args):
    print('\n' + '=' * 80)
    for i in args:
        print('\t%s' % i)


def getCons(mode, ndim=3, nump=1, explicitFlag=False, numg=None, **kwargs):
    if mode == 'misesideal':
        from FEMxEPxML.vonmisesConsIdeal import vonmisesIdealConstitutive
        cons = vonmisesIdealConstitutive(
            explicitFlag=explicitFlag, numg=numg, rho=kwargs['rho'],
            p0=kwargs['p0'],
            nu=0.2, E=2e6, yield_stress0=1e5,
            verboseFlag=False, ndim=2, save_flag=True, nump=nump,
            save_path=kwargs['save_path'],
        )
    elif mode == 'drucker':
        from FEMxEPxML.DruckerPragerCons import druckerPragerConstitutive
        cons = druckerPragerConstitutive(
            p0=kwargs['p0'], numg=numg, save_path=kwargs['save_path'], rho=kwargs['rho'], nump=nump,
            explicitFlag=explicitFlag)

    elif mode == 'mises_harden':
        from FEMxEPxML.vonMisesConsHarden import MisesHardenConstitutive
        cons = MisesHardenConstitutive(
            p0=kwargs['p0'], numg=numg, save_path=kwargs['save_path'], rho=kwargs['rho'], nump=nump,
            explicitFlag=explicitFlag)
        
    elif mode == 'rnn_misesideal' or mode == 'rnn_drucker' or mode == 'rnn_drucker_fc' or mode=='rnn_mises_harden'\
            or mode == 'rnn_mises_harden_fc' or mode == 'rnn_mises_harden_epsp' or mode == 'rnn_mises_harden_epsp_fc'\
            or mode == 'rnn_misesideal_fc' or mode=='rnn_mises_harden_extract':
        from FEMxEPxML.rnnMisesIdealCons import rnnMisesIdealConstitutive
        cons = rnnMisesIdealConstitutive(
            p0=kwargs['p0'], numg=numg, save_path=kwargs['save_path'], rho=kwargs['rho'],
            step_scalar=kwargs['scalar'], mode=mode)
    else:
        raise ValueError('Mode %s not involved yet.' % mode)
    return cons


def pickle_dump(**kwargs):
    root_path = kwargs['root_path']
    savePath = os.path.join(root_path, 'scalar')
    check_mkdir(savePath)
    for k in kwargs:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'wb')
            pickle.dump(kwargs[k], f, 0)
            f.close()
    print('\tScalar saved in %s' % savePath)


def pickle_load(*args, root_path):
    cwd = os.getcwd()
    # if 'FEMxML' not in cwd:
    #     root_path = os.path.join(cwd, 'FEMxML')
    if 'sciptes4figures' in cwd:
        root_path = os.getcwd()
    savePath = os.path.join(root_path, 'scalar')
    # if not os.path.exists(savePath):
    #     os.mkdir(savePath)
    if 'epoch' in root_path:
        root_path = os.path.split(root_path)[0]
        savePath = os.path.join(root_path, 'scalar')
    print()
    print('-' * 80)
    print('Note: Scalar restored from %s' % savePath)
    for k in args:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'rb')
            # eval('%s = pickle.load(f)' % k)
            yield eval('pickle.load(f)')
            f.close()


def mapMask(param):
    return param[0](param[1])


def get_load_information(
        out_directory, test_name, mode, explicit_flag, order, numg,
        nx=None, ny=None, smooth_flag=None, mesh_name=None, rate_vel=None, safety_coefficient=None, vel=None,
        damp=None, gama=None,
        **kwargs):
    temp = test_name
    if smooth_flag is not None:
        temp += '_%s' % ('smooth' if smooth_flag else 'rough')
    temp += '_%s_%s_intorder%d_numg%d' % (
        'explicit' if explicit_flag else 'implicit', mode, order, numg)
    if mesh_name:
        temp += '_%s' % mesh_name
    else:
        temp += '_x%d_y%d' % (nx, ny)
    if explicit_flag:
        if rate_vel is not None:
            temp += '_rate%.2f' % np.abs(rate_vel)
        else:
            temp += '_vel%.2f' % np.abs(vel)
        if damp is not None and damp != 0.:
            if gama is not None:
                temp += '_gama%.2f' % (gama)
            else:
                temp += '_damp%.1e' % (damp)
        temp += '_safe%.1f' % (safety_coefficient)
    if mode == 'uh':
        temp += '_p%dkPa_ocr%.1f' % (kwargs['p0'] / 1e3, kwargs['ocr'])
    elif mode == 'csuh':
        temp += '_p%dkPa_ocr_%.1f_bflag%d' % (kwargs['p0'] / 1e3, kwargs['ocr'], 1 if kwargs['b_flag'] else 0)

    elif mode == 'norsand':
        temp += '_p%dkPa_e%.3f' % (kwargs['p0'] / 1e3, kwargs['e0'])
    elif mode == 'mldem' or mode == 'mixed':
        if 'active' in kwargs['NN_sig_path']:
            temp += '_active'
        temp += '_NN%s' % kwargs['nn_name']
    elif mode == 'eb':
        temp += '_fric_%.1f' % kwargs['fric']
    elif mode == '2ml':
        temp += '_%s' % kwargs['input_features']

    if explicit_flag:
        temp += "_timestep%.1e" % kwargs['time_step']

    if 'rnn' in mode:
        if kwargs['scalar']:
            temp += '_scalar%d' % kwargs['scalar']
        else:
            temp += '_scalarAdap'

    name = os.path.join(out_directory, temp)
    return name


def get_time_step(rho, lam_2G, element_size, safety_coefficient=0.2):
    time_step = safety_coefficient * np.sqrt(rho / lam_2G) * element_size
    return time_step


def explicit_material_constants(p0=None, nn_name=None, nn_name_D=None, csuh_para_line=None, active_iter=None, ocr=None):
    if p0 is None:
        p0 = 1e5  # confining pressure
    else:
        p0 = p0
    # footing-dem Loss :5.459e-02 	 kappa:5.212e-02 	 lambdaa:1.488e-01 	 N:1.791e+00 	 Z:9.759e-01 	 ocr:3.599e+01 	 theta_degree:2.359e+01
    if csuh_para_line is None:
        # csuh_dic = get_dic_from_string(s='ocr:10. \t theta_degree:30. \t lambdaa:0.135 \t kappa:0.04 \t N:1.973 \t Z:0.933938655')
        # csuh_dic = get_dic_from_string(s='kappa:5.748e-02 	 lambdaa:1.500e-01 	 N:1.804e+00 	 Z:9.415e-01 	 ocr:3.207e+01 	 theta_degree:2.578e+01')  # fine
        csuh_dic = get_dic_from_string(
            s='kappa:1.906e-01 	 lambdaa:2.142e-01 	 N:1.931e+00 	 Z:2.743e-01 	 ocr:3.774e+02 	 theta_degree:1.329e+01')  # optimized from the dataset collected from the dem simulation
        # csuh_dic = get_dic_from_string(s='kappa:1.906e-01 	 lambdaa:2.142e-01 	 N:1.931e+00 	 Z:2.743e-01 	 ocr:3.774e+02 	 theta_degree:8.')  # Parameter used in the paper of exFEM-NN
        # csuh_dic = get_dic_from_string(s='ocr:20. \t theta_degree:30. \t lambdaa:0.1689 \t kappa:0.1 \t N:2.021 \t Z:0.9358')
    else:
        # csuh_dic = get_dic_from_string('kappa:5.111e-02 	 lambdaa:1.485e-01 	 N:1.790e+00 	 Z:9.824e-01 	 ocr:3.833e+01 	 theta_degree:2.314e+01')
        csuh_dic = get_dic_from_string(csuh_para_line)
    if ocr is not None:
        csuh_dic['ocr'] = ocr
    ocr = csuh_dic['ocr']

    poisson = 0.2

    # original

    e0 = 0.6
    E = 2e7
    lam = E * poisson / (1 + poisson) / (1 - 2 * poisson)
    G = E / 2 / (1 + poisson)
    rho = 2650  # kg/m^3

    # nn_name = 'X_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_mix_biaxial_1'
    kwargs = {'p0': p0, 'ocr': ocr, 'e0': e0, 'lam': lam, 'rho': rho, 'G': G,
              'poisson': poisson,
              'E': E,
              'csuh_dic': csuh_dic,
              }
    if nn_name is not None:
        input_features = nn_name.split('X_')[1].split('_')[0]
        if active_iter is None:
            kwargs['NN_sig_path'] = './FEMxML/biax_ml_1e5/%s/entire_model.pt' % nn_name
        else:
            kwargs['NN_sig_path'] = './FEMxML/%s/entire_model_iter%d.pt' % (nn_name, active_iter)
        kwargs['input_features'] = input_features
        if 'active' in nn_name:
            temp = nn_name.split('/')[2]
            if active_iter is not None:
                temp += '_iter%d' % active_iter
            kwargs['nn_name'] = temp
        else:
            kwargs['nn_name'] = os.path.split(nn_name)[-1]
    if nn_name_D is not None:
        if active_iter is None:
            kwargs['NN_D_path'] = './FEMxML/biax_ml_1e5/%s/entire_model.pt' % nn_name_D
        else:
            kwargs['NN_D_path'] = './FEMxML/%s/entire_model_iter%d.pt' % (nn_name_D, active_iter)
    return p0, e0, ocr, E, poisson, lam, G, rho, nn_name, kwargs


def get_dic_from_string(s: str):
    dic = {}
    s = s.replace(' ', '')
    line_list = s.split('\t')
    for i in line_list:
        temp = i.split(':')
        if temp[
            0] in "kappa,lambdaa,N,Z,ocr,theta_degree,M,nu,E,A,B,epsilon0,yield_stress0,dilation_coefficient,yield_p_c,C,D,epsilon0_p,harden_E":
            dic[temp[0]] = float(temp[1])
    return dic


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True
