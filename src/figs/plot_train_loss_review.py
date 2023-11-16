import os

import matplotlib.pyplot as plt
import numpy as np
from sciptes4figures.utils_plot import plot_training_loss, get_color_list, configurations
from ml_cons_train.plot_training_loss_parameters_evolution import plot_data_frame
from utilSelf.general import echo
from FEMxML.rnn_liverpool_research_assistant.rnn_model_trainer_j2 import get_save_dir_name



font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 200
# fix random seeds
axes = {'labelsize': 'large'}
font = {'family': 'serif',
        'weight': 'normal',
        'size': 17}
legend = {'fontsize': 'large'}
lines = {'linewidth': 3,
         'markersize': 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)

colorlist = ['#008080', '#FF7F50', '#4169E1', '#DA70D6', '#808000', '#4B0082',
             '#FF8C00', '#FF1493', '#4682B4', '#DAA520', '#9370DB',
             '#2E8B57', '#483D8B', '#FF6347', '#008B8B', '#BA55D3',
             '#B8860B', '#1E90FF', '#3CB371'
             ]


def main(nn_architecture='sig', extra_description=None, lc=None, len_sequence=None, mode='j2', l_epsp_c=None, num_deep_layer=None):

    name_characters = '%s%s' % (nn_architecture, extra_description)
    if lc:
        name_characters += '_lc%.1f' % lc
    if l_epsp_c and 'epsp' in extra_description:
        name_characters += '_lepsp%.1f' % l_epsp_c
    if len_sequence:
        name_characters += '_len%d' % len_sequence
    if num_deep_layer:
        name_characters += '_deep%d' % num_deep_layer
    echo(name_characters)

    filename_list = os.listdir("../")
    loss_dic = {}
    path_list = []
    num_samples_list = []
    for filename in filename_list:
        if ('rnn_%s' % mode) not in filename:
            continue

        if name_characters not in filename.split('NN')[-1]:
            continue
        num_samples = int(filename.split('Samples')[-1].split('_')[0])
        path_list.append(filename)
        num_samples_list.append(num_samples)
    index = np.argsort(np.array(num_samples_list))
    for i in index:
        num_samples = num_samples_list[i]
        filename = path_list[i]

        echo('Loss evolution data collected from %s' % filename)
        history_file_path = os.path.join("../", os.path.join(filename, 'history.txt'))
        with open(history_file_path, mode='r') as f:
            history_str = f.readlines()
        epoch_list, train_loss, val_loss, K, G, q_yield = \
            [], [], [], [], [], []
        for line in history_str:
            if line[:5] != 'Epoch':
                continue
            line_list = line.split('/')
            epoch_num = int(line_list[0].split('[')[-1])
            line_list2 = line_list[-1].split('\t')
            epoch_list.append(epoch_num)
            train_loss.append(float(line_list2[0].split(' ')[-1]))
            val_loss.append(float(line_list2[1].split(' ')[-1]))
            if extra_description == '_cal':
                line_list3 = line_list2[5].split(' ')
                K.append(float(line_list3[0].split('=')[-1]))
                G.append(float(line_list3[1].split('=')[-1]))
                q_yield.append(float(line_list3[2].split('=')[-1].split('\n')[0]))
        if extra_description == '_cal':
            plot_data_frame(
                label_list=['Epoch', 'K', 'G', r'$\sigma_y$'],
                data_list=np.array([epoch_list, K, G, q_yield]).T)
        loss_dic['%s_%d' % (mode, num_samples)] = [
            np.array(epoch_list), np.array(train_loss), np.array(val_loss)]

    for key in loss_dic.keys():
        echo('%s best loss: train_loss: %.2e\tval_loss: %.2e' %
             (key, np.min(loss_dic[key][1]), np.min(loss_dic[key][2])))

    # plot_training_loss(loss_dic=loss_dic, train_plot_flag=True, validation_plot_flag=True)

    # 作图
    for key in loss_dic:
        epoch = loss_dic[key][0]
        train = loss_dic[key][1]
        valid = loss_dic[key][2]
        key = key.replace('201', '200')
        key = key.replace('999', '1000')
        key = key.replace('j2_', '')
        if key == "20":
            internal = 25
        else:
            internal = 100
        plt.scatter(epoch[::internal]/1e3, valid[::internal], marker='o', s=20,
                # label='%s_validation' % key,
                edgecolors='k', zorder=10, alpha=0.7)
        plt.semilogy(epoch/1e3, train, label='%s' % key, zorder=9, alpha=0.7)
    plt.xlabel('Epoch/1e3')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./training_loss_j2_%s_extensions.png" % nn_architecture, dpi=200)
    # plt.show()
    return


if __name__ == '__main__':
    '''
        这个文件用来作J2 physics_extended 模型的训练误差，  nn_architecture = 'classic'  len_sequence=200, 
        用于跟数据驱动的模型进行对比，体现physics_extended在缓解模型对数据依赖上的作用  
        
    '''
    mode = 'j2'  # 'j2' 'drucker'  j2_harden

    # =============================
    # 用于比较 J2 with physical extensions
    nn_architecture = 'sig'      # 'classic', 'pq', 'pqh', 'sig'
    len_sequence = 20

    # =============================
    # 用于比较 J2 data-driven
    # nn_architecture = 'classic'      # 'classic', 'pq', 'pqh', 'sig'
    # len_sequence = 200

    extra_description = '_deps_sig'   # _deps_sig  _deps_sig_fc _deps_sig_epsp _deps_sig_epsp_fc _deps_sig_epsp_split
                                              # _deps_sig_epsp_split_fc  _deps_sig_epsp_mc  _deps_sig_epsp_split_classify
                                              # _deps_sig_epspvec_split_classify
                                              # _deps_sig_simple
                                              # _deps_sig_muLtiGRU
    lc = 1.0
    l_epsp_c = 1.0
    num_deep_layer = None
    main(
        nn_architecture=nn_architecture, extra_description=extra_description, lc=lc, len_sequence=len_sequence,
        mode=mode, l_epsp_c=l_epsp_c, num_deep_layer=num_deep_layer)

