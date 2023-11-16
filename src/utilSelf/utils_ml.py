import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from utilSelf.general import echo, check_mkdir, writeLine


def getFilesPathList(root_path_list, maxTime, explicit_flag=False, add_flag=False, series_flag=False):
    file_list = []
    num = 0
    for pathTemp in root_path_list:
        if add_flag:
            file_path = pathTemp
        else:
            file_path = os.path.join(pathTemp, 'iteration_gauss')
        # file_name_list = ['time_%d.dat' % i for i in range(1, 101)]
        file_name_list = os.listdir(file_path)
        if "added_points" == os.path.split(file_path)[-1]:
            step_list = [int(name.split('_')[1]) for name in file_name_list]
            step_argsort = np.argsort(np.array(step_list))
            file_list += [os.path.join(file_path, file_name_list[index]) for index in step_argsort]
        elif explicit_flag:
            step_list = [int(name.split('_')[1].split('.')[0]) for name in file_name_list]
            step_argsort = np.argsort(np.array(step_list))
            file_list += [os.path.join(file_path, file_name_list[index]) for index in step_argsort]
        else:
        # sort the according to time sequence
            for i in range(0, maxTime + 1):
                temp1 = 'time_%d.dat' % i
                if temp1 in file_name_list and (i == 0 or explicit_flag):
                    file_list.append(os.path.join(file_path, temp1))
                if not explicit_flag:
                    temp1_0 = 'time_%d_iter_0.dat' % i
                    if temp1_0 in file_name_list:
                        for j in range(0, 101):
                            temp2 = 'time_%d_iter_%d.dat' % (i, j)
                            if temp2 in file_name_list:
                                if series_flag and j > 0:
                                    file_list[-1] = os.path.join(file_path, temp2)
                                else:
                                    file_list.append(os.path.join(file_path, temp2))
                            else:
                                break

        echo('Path: %s Num: %d' % (file_path, len(file_list)-num))
        num = len(file_list)
    return file_list


def get_data(root_path_list,  maxTime=100, mixflag=False, explicit_flag=False, add_flag=False, series_flag=False):
    echo('Reading data...')
    file_list = getFilesPathList(root_path_list,
                                 maxTime, explicit_flag=explicit_flag,
                                 add_flag=add_flag, series_flag=series_flag)
    if mixflag:
        indexList = []
        for rootfile in root_path_list:
            indexList.append(getGaussianPointsIndex(filePath=rootfile))

    stress, strain, fabric, strain_increment, tangent, strain_abs, stress_last = [], [], [], [], [], [], []
    H_0, H_1 = [], []
    for file in file_list:
        f = open(file, 'r')
        datas = f.readlines()
        f.close()
        n = int(datas[0].split(' ')[-1])  # n is the number of the RVEs
        if mixflag:
            if n == 32:
                listTemp = np.concatenate((indexList[0]['lower'], indexList[0]['upper']),  axis=0)+1
            elif n == 512:
                listTemp = indexList[1]['shear']+1
        i, tol = 0, len(datas)
        while i < tol:
            '''
                NAME                # 
            strain_increment        0
            strain_total            1
            stress_increment        2
            stress_toatal           3 
            tangent                 4
            fabric                  5
            vR                      6
            strain_abs              7
            '''
            temp = datas[i].split(' ')[0]
            if 'strain_increment' in temp:
                # strain_increment.append(blockDataReader(datas[i + 1:i + n + 1]))
                strain_increment+=(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'strain_toatal' in temp or temp == 'eps':
                # strain.append(blockDataReader(datas[i + 1:i + n + 1]))
                strain+=(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'stress_toatal' in temp or temp == 'sig':
                # stress.append(blockDataReader(datas[i + 1:i + n + 1]))
                stress+=(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'fabric' in temp:
                # fabric.append(blockDataReader(datas[i + 1:i + n + 1]))
                fabric+=(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'tangent' in temp or temp == 'D':
                # tangent.append(blockDataReader(datas[i + 1:i + n + 1]))
                tangent+=(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'strain_abs' in temp or temp == 'eps_abs':
                # strain_abs.append(blockDataReader(datas[i + 1:i + n + 1]))
                strain_abs += (blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif temp == 'sig_last':
                # stress_last.append(blockDataReader(datas[i + 1:i + n + 1]))
                stress_last+=(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif temp == 'H_0':
                H_0 += (blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif temp == 'H_1':
                H_1 += (blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)

            else:
                i += 1

    # delete stress subject to the symmetricity \sigma_xy = \sigma_yx
    stress = np.array(stress).reshape([-1, 4])
    stress = np.delete(stress, [1], axis=1)
    strain = np.array(strain).reshape([-1, 4])
    strain = np.concatenate(
        (strain[:, 0:1], .5*(strain[:, 1:2]+strain[:, 2:3]), strain[:, 3:4]), axis=1)
    strain_abs = np.array(strain_abs).reshape([-1, 4])
    strain_abs = np.concatenate(
        (strain_abs[:, 0:1], .5*(strain_abs[:, 1:2]+strain_abs[:, 2:3]), strain_abs[:, 3:4]), axis=1)
    tangent = np.array(tangent).reshape([-1, 6])
    stress_last = np.array(stress_last).reshape([-1, 4])
    stress_last = np.delete(stress_last, [1], axis=1)
    H_0, H_1 = np.array(H_0), np.array(H_1)

    # add the initial state to the dataset
    # strain = np.concatenate((strain, np.array([[0.,0.,0.] for _ in range(n)])), axis=0)
    # stress = np.concatenate((stress, np.array([[-1e5,0,-1e5] for _ in range(n)])), axis=0)
    # tangent = np.concatenate((tangent, tangent[:n]), axis=0)
    # stress_last = np.concatenate((stress_last, np.array([[-1e5,0,-1e5] for _ in range(n)])), axis=0)
    returned_dict = {
        'sig': stress, 'eps': strain,
        'eps_abs':strain_abs,
    }
    if len(stress_last)!=0:
        returned_dict['sig_last'] = stress_last
    if len(H_0)!=0:
        returned_dict['H_0'] = H_0
        returned_dict['H_1'] = H_1
    if len(tangent)!=0:
        returned_dict['tangent'] = tangent
    return returned_dict


def reconstruct_x_y(
        input_features, output_features,
        eps, sig,
        tangent=None, eps_abs=None, sig_last=None, rotate_flag=False, H_0=None, H_1=None):
    if rotate_flag:
        eps = xy_exchange_glue(eps)
        sig = xy_exchange_glue(sig)
        eps_abs = xy_exchange_glue(eps_abs)
    # reconstruct the input and output
    if input_features == 'epsANDabsy':
        input_value = np.concatenate((eps, eps_abs[..., 2:3]), axis=-1)
    elif input_features == 'epsANDabsxy':
        input_value = np.concatenate((eps, eps_abs[..., 0:1], eps_abs[..., 2:3]), axis=-1)
    elif input_features == 'epsANDabsxyq':
        input_value = np.concatenate((eps, eps_abs[..., 0:1], eps_abs[..., 2:3], get_q_2d(sig_last)), axis=-1)
    elif input_features == 'epsANDabs':
        input_value = np.concatenate((eps, eps_abs), axis=-1)
    elif input_features == 'epsANDsiglast':
        input_value = np.concatenate((eps, sig_last), axis=-1)
    elif input_features == 'epsANDplast':
        input_value = np.concatenate((
            eps,
            get_p_2d(sig_last)), axis=1)
    elif input_features == 'epsANDpqlast':
        input_value = np.concatenate((
            eps,
            get_p_2d(sig_last), get_q_2d(sig_last)), axis=1)
    elif input_features == 'eps':
        input_value = eps
    elif input_features == 'epsANDH':
        input_value = np.concatenate((eps, H_0), axis=1)
    elif input_features == 'epsANDqH':
        input_value = np.concatenate((eps, get_q_2d(sig_last), H_0), axis=1)
    elif input_features == 'epsANDpqH':
        input_value = np.concatenate((eps, get_p_2d(sig_last), get_q_2d(sig_last), H_0), axis=1)
    else:
        raise ValueError('No %s input mode' % input_features)
    if output_features == 'sig':
        output_value = sig
    elif output_features == 'D':
        output_value = tangent
    elif output_features == 'sigANDH':
        output_value = np.concatenate((sig, H_1), axis=1)
    else:
        raise ValueError('No %s output mode' % output_features)
    return input_value, output_value


def save_scalar(scalarPath, input_value, output_value):
    cwd = os.getcwd()
    input_mean, input_std = getMeanStd(data=input_value)
    output_mean, output_std = getMeanStd(data=output_value)
    kwargs = {'input_mean': input_mean, 'input_std': input_std,
              'output_mean': output_mean, 'output_std': output_std}
    pickle_dump(
        root_path=os.path.join(cwd, scalarPath) if scalarPath else cwd,
        **kwargs)


def blockDataReader(datas):
    blockData = []
    for line in datas:
        if '[144685.11089822' in line:
            print()
        blockData.append(np.array([float(j.replace('\n', '')) for j in line.split(' ')], dtype=float))
    return blockData


def getMeanStd(data: np.array):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std


def pickle_dump(**kwargs):
    root_path = kwargs['root_path']
    savePath = os.path.join(root_path, 'scalar')
    check_mkdir(savePath)
    for k in kwargs:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'wb')
            pickle.dump(kwargs[k], f, 0)
            f.close()
    echo('\tScalar saved in %s' % savePath)

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

    echo('Note: Scalar restored from %s' % savePath)
    for k in args:
        if k != 'root_path':
            f = open(os.path.join(savePath, '%s' % k), 'rb')
            # eval('%s = pickle.load(f)' % k)
            yield eval('pickle.load(f)')
            f.close()


def xy_exchange_glue(x_data):
    '''
      0  1  2
     xx xy yy

     0      1      2      3      4     5
    xxxx  xxxy   xxyy   xyxy   yy_xy  yyyy

     5      4      2      3      1     0
    yyyy  yy_xy  xxyy   xyxy   xx_xy  xxxx
    '''
    x_data_new = np.concatenate((x_data[:, 2:3],x_data[:, 1:2],x_data[:, 0:1]), axis=1)
    x_data_glued = np.concatenate((x_data_new, x_data), axis=0)
    return x_data_glued


def getGaussianPointsIndex(filePath):
    indexDic = {}
    f = open(os.path.join(filePath, 'gaussianPointsList.txt'), 'r')
    data = f.readlines()
    f.close()
    i = 0
    while i < len(data):
        if 'Upper' in data[i]:
            i += 1
            indexDic['upper'] = line2IntArray(data[i])
            i += 1
        elif 'Lower' in data[i]:
            i += 1
            indexDic['lower'] = line2IntArray(data[i])
            i += 1
        elif 'Shear' in data[i]:
            i += 1
            indexDic['shear'] = line2IntArray(data[i])
            i += 1
        else:
            i += 1

    return indexDic


def line2IntArray(data):
    indexTemp = data[:-1].split(sep=' ')
    indexTemp[-1] = indexTemp[-1].replace('\n', '')
    indexTemp = np.array([int(i) for i in indexTemp], dtype=np.int)
    return indexTemp


# plot the prediction data on $y=x$ line
def plot_prection(y_origin, y_predict, s='prediction & DEM simulation', root_path='./'):
    plot_path = os.path.join(root_path, 'plot_prediction')
    check_mkdir(plot_path)
    import matplotlib.ticker as mtick
    y_origin = y_origin.flatten()
    y_predict = y_predict.flatten()
    fig = plt.figure()
    axes = fig.gca()
    plt.plot([min(y_origin), max(y_origin)], [min(y_origin), max(y_origin)], 'r--')
    # axes.scatter(y_origin[:1000000], y_predict[:1000000])
    axes.scatter(y_origin, y_predict, s=2)
    plt.axis('equal')
    # plt.title(s, fontsize=15)
    plot_name = '%s_prediction.png' % s
    temp = os.path.join(plot_path, plot_name)
    plt.xlabel('DEM simulation', fontdict={'weight': 'normal', 'size': 16})
    plt.ylabel('ML prediction', fontdict={'weight': 'normal', 'size': 16})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # axes.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    # axes.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.tight_layout()
    # plt.show()
    plt.savefig(temp, dpi=400)
    print('figure saved as \n \t %s' % temp)


def findDevice(useGPU=True):
    print()
    print('-' * 80)
    if useGPU and torch.cuda.is_available():
        device = torch.device('cuda')
        echo('\t%s is used in this calculation' % torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        echo('\tOnly CPU is used in this calculation')
    return device


def splitTrainValidation(inputs, outputs, d_outouts=None, valRatio=0.2):
    train_size = int(len(inputs) * (1 - valRatio))
    rnd_idx = np.random.permutation(len(inputs))

    train_index = rnd_idx[:train_size]
    val_index = rnd_idx[train_size:]

    x_Train = inputs[train_index]
    y_Train = outputs[train_index]
    x_Val = inputs[val_index]
    y_Val = outputs[val_index]

    if d_outouts is not None:
        d_y_train = d_outouts[train_index]
        d_y_val = d_outouts[val_index]
        return x_Train, x_Val, y_Train, y_Val, d_y_train, d_y_val
    else:
        return x_Train, x_Val, y_Train, y_Val


def normalization(x, xmean, xstd, reverse=False):
    if reverse:
        normed = x * xstd + xmean
    else:
        normed = (x - xmean) / xstd
    return normed


def sampling_index(x_data, sample_num=None, ratio=None):
    n = len(x_data)
    if sample_num:
        num = sample_num
    elif ratio and 0. < ratio <= 1.:
        num = int(ratio*n)
    else:
        echo('sample_num: %s ratio: %s' % (sample_num, ratio))
        raise
    echo('Sampled/Total:  %d/%d' % (num, n))
    return np.random.permutation(range(n))[:num]


def calVariance(data, num=6):
    '''
    data = [num_NN, num_sample, num_feature]
    '''
    std = np.std(data, axis=0)
    stdAverage = np.average(std)
    return stdAverage


def remove_from_list(a: list, b: list):
    for i in b:
        a.remove(i)
    return a


def get_p_2d(stress_voigt):
    p = 0.5*(stress_voigt[:, 0:1]+stress_voigt[:, 2:3])
    return p


def get_q_2d(stress_voigt):
    sig_vonmises = np.sqrt(stress_voigt[:, 0]**2.-
                           stress_voigt[:, 0]*stress_voigt[:, 2]+
                           stress_voigt[:, 2]**2.+3.*stress_voigt[:, 1]**2.)
    return sig_vonmises.reshape([-1, 1])


def get_combination_num(input_num):
    if input_num == 3:
        combination_num = 4
    elif input_num == 6:
        combination_num = 9
    else:
        echo('Input_num %d has not been added yet!' % input_num)
        raise
    return combination_num


def writeDown(info, savePath, appendFlag=False):
    fname = os.path.join(savePath, 'history.dat')
    if appendFlag:
        # print('-'*100)
        # print('Append the history file and retain!')
        writeLine(fname, s=info, mode='a')
    else:
        # print('-'*100)
        # print('Delete the history file and begin training!')
        writeLine(fname, s=info, mode='w')


def transform_data_in_blocks_2_series(datas: np.ndarray, numg: int):
    '''
        numg is the number gaussian points
        datas is the ndarray to transform
    '''
    n = len(datas)
    n_features = len(datas[0])
    num_steps = n//numg
    data_series = np.zeros(shape=[numg, num_steps, n_features])
    for numg_temp in range(numg):
        for step in range(num_steps):
            data_series[numg_temp, step] = datas[numg_temp+step*numg]
    return data_series


def get_data_series(
        root_path_list, numg, maxTime=100, mixflag=False, series_flag=False, explicit_flag=False, add_flag=False):
    '''
        series_flag used to ONLY read the converged results in the implicit datasets
    '''
    data_dic = get_data(
        root_path_list, maxTime=maxTime, mixflag=mixflag,
        explicit_flag=explicit_flag, series_flag=series_flag, add_flag=add_flag)
    for key in data_dic.keys():
        if len(data_dic[key]) > 0:
            data_dic[key] = transform_data_in_blocks_2_series(datas=data_dic[key], numg=numg)
    return data_dic


def error_evaluate(t_true, t_pre, ndim=2):
    """
        Evaluating the average relative error between 2 tensor
    """
    relative_error = []
    if ndim == 2:
        check_list = [(0, 0), (1, 1)]
    else:
        raise
    for i, j in check_list:
        relative_error.append(0. if abs(t_true[i, j]) < 1e2 else np.abs(t_pre[i, j]/t_true[i, j]-1.))
    # relative_error = np.abs(t_true-t_pre)/t_true
    return relative_error


def data_clean(data_dic: dict):
    '''
        data_dic['sig'] = [:, n_features]
    '''
    sig = data_dic['sig']
    not_use_index = []
    for i in range(len(sig)):
        if np.sum(sig[i]*sig[i]) == 0.:
            not_use_index.append(i)
    not_use_index = np.array(not_use_index)
    echo('Number of the deleted datasets is: (%d)' % len(not_use_index),
         # '\t\t\t\t\t\t'+' '.join(['%d' % i for i in not_use_index])
         )
    if len(not_use_index)>=1:
        for key in data_dic.keys():
            data_dic[key] = np.delete(data_dic[key], obj=not_use_index, axis=0)
    return data_dic


def data_clean_numg(data_dic: dict):
    '''
        data_dic['sig'] = [numg, step, n_features]
    '''
    sig = data_dic['sig']
    not_use_numg = []
    for numg in range(len(sig)):
        for sig_step in sig[numg]:
            if np.sum(sig_step * sig_step) == 0.:
                not_use_numg.append(numg)
                break
    not_use_numg = np.array(not_use_numg)

    echo('Number of the deleted gauss points is: (%d)' % len(not_use_numg),
         '\tNumber of the points: '+' '.join(['%d' % i for i in not_use_numg])
         )

    if len(not_use_numg) >= 1:
        for key in data_dic.keys():
            data_dic[key] = np.delete(data_dic[key], obj=not_use_numg, axis=0)
    return data_dic
        
