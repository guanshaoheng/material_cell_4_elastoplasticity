import os
import time

import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cons.utils_constitutive import tensor2voigt
from rnn_model import rnn_net_4_constitutive, \
    rnn_net_4_constitutive_fixed_sig_general
from utilSelf.utils_rnn_cons import get_q_2d_arr
from utilSelf.utils_ml import findDevice, splitTrainValidation
from utilSelf.general import check_mkdir, writeLine, echo, load_npy


def main(
        num_epochs=int(1e7), num_inputs=3, num_outputs=3, num_layers=2, hidden_size=30,
        print_interval=10, patience_num=1000, num_samples=1000, batch_size=20, n_batch_max=10,
        nn_architecture='classic', extra_description=None, lc=None, len_sequence=50, mode='j2',
        l_epsp_c=None, num_deep_layer=None,
        extract_number: int = None,
):
    """

    :param num_epochs:
    :param num_inputs:
    :param num_outputs:
    :param num_layers:
    :param hidden_size:
    :param print_interval:
    :param patience_num:
    :param num_samples:
    :param batch_size:
    :param n_batch_max:
    :param nn_architecture:
    :param extra_description:
    :param lc:   loss coefficient for \sigma_{xy}, the larger the influence should be stronger.
    :return:
    """
    # working directory
    cwd = os.getcwd()

    save_dir = get_save_dir_name(
        cwd=cwd, mode=mode, num_samples=num_samples, nn_architecture=nn_architecture,
        extra_description=extra_description, lc=lc, len_sequence=len_sequence, l_epsp_c=l_epsp_c,
        num_deep_layer=num_deep_layer, extract_number=extract_number,
    )
    # save_dir += '_test'
    check_mkdir(save_dir)
    history_file_full_path = os.path.join(save_dir, 'history.txt')

    # define the general things
    start_time = time.time()

    # datasets reading
    strain_paths_vector, sig_numg_vector, eps_p_norm, eps_p_numg_vector = data_reading(
        num_samples=num_samples, mode=mode, voigt_flag=True)
    num_samples, num_steps = len(strain_paths_vector), len(strain_paths_vector[0])

    # extracting a certain number of points from the whole sequence
    if extract_number is not None:
        extract_index = np.arange(0, num_steps, num_steps // extract_number)
        strain_paths_vector = strain_paths_vector[:, extract_index]
        sig_numg_vector = sig_numg_vector[:, extract_index]
        eps_p_norm = eps_p_norm[:, extract_index]
        eps_p_numg_vector = eps_p_numg_vector[:, extract_index]

    # add the plastic strain as internal variables into the stress
    if 'epsp' in extra_description:
        sig_numg_vector = np.concatenate((sig_numg_vector, eps_p_norm[:, :, np.newaxis]), axis=-1)
    if 'mc' in extra_description or 'epspvec' in extra_description:
        sig_numg_vector = np.concatenate((sig_numg_vector, eps_p_numg_vector), axis=-1)

    # dataset pre-processing get the scalar of the datasets
    scalar_x, scalar_y, train_x_norm_median = data_preprocessing(
        strain_paths_vector=strain_paths_vector, sig_numg_vector=sig_numg_vector, save_dir=save_dir,
        nn_architecture=nn_architecture)

    # s plit the datasets into
    if extract_number is None:
        strain_paths_vector, sig_numg_vector = truncated_sequence_via_time(
            x_sequence=strain_paths_vector, y_sequence=sig_numg_vector, len_sequence=len_sequence)

    # del strain_paths_vector, sig_numg_vector

    # initializing the network
    device = findDevice(useGPU=True)
    loss_operator = torch.nn.MSELoss()
    if nn_architecture == 'classic':  # classic gru with data-driven hidden
        model = rnn_net_4_constitutive(
            num_inputs=num_inputs, num_outputs=num_outputs,
            num_layers=num_layers, hidden_size=hidden_size, device=device, save_dir=save_dir)
    elif nn_architecture == 'sig':  # change model with explicit hidden i. e. sig, p, q,...
        model = rnn_net_4_constitutive_fixed_sig_general(
            num_inputs=num_inputs, num_outputs=num_outputs, device=device, hidden_size=hidden_size,
            extra_description=extra_description, save_dir=save_dir, num_deep_layer=num_deep_layer,
            train_x_norm_median=train_x_norm_median)
    else:
        raise ValueError('Please check if the nn_architecture (%s) is included' % nn_architecture)
    if extra_description == '_cal':
        optimizer = torch.optim.Adam([model.K_norm, model.G_norm, model.q_yield_norm])
    else:
        optimizer = torch.optim.Adam(model.parameters())

    # echo before training
    line = '---------------------------------------------------- \n' \
           f'\t num of samples:                    {num_samples} \n' \
           f'\t num of steps:                      {num_steps} \n' \
           f'\t num of inputs:                     {num_inputs} \n' \
           f'\t num of outputs:                    {num_outputs} \n' \
           f'\t num of GRU hidden layers:          {num_layers} \n' \
           f'\t num of GRU hidden size:            {hidden_size} \n' \
           f'\t bacth size:                        {batch_size} \n' \
           f'\t max number of bacthes per epoch:   {n_batch_max} \n' \
           f'\t save dir:                          {save_dir} \n' \
           '\n'
    echo(line)
    line_model_capacity = model.model_capacity() + '\n\n'
    echo(line_model_capacity)
    # summary(model, input_size=(num_steps, num_inputs))
    writeLine(fname=history_file_full_path, s=line, mode='w')
    writeLine(fname=history_file_full_path, s=line_model_capacity, mode='a')

    # model training implemented in the loop structure
    model_save_full_path = os.path.join(save_dir, 'rnn.pt')
    min_loss = 1e5
    trial_num = 0
    x_train, x_val, y_train, y_val = splitTrainValidation(
        inputs=strain_paths_vector, outputs=sig_numg_vector)
    num_train_samples = len(x_train)
    index = np.arange(num_train_samples)
    index_val = np.arange(len(x_val))
    x_tensor, y_tensor = torch.from_numpy(x_train).float().to(device=device), torch.from_numpy(y_train).float().to(
        device=device)
    x_val_tensor, y_val_tensor = torch.from_numpy(x_val).float().to(device=device), torch.from_numpy(y_val).float().to(
        device=device)
    y_std_tensor = torch.from_numpy(scalar_y.scale_).float().to(device=device)
    for epoch in range(num_epochs):
        np.random.shuffle(index)
        loss_train = 0.
        round_num = min(max(num_train_samples // batch_size, 1), n_batch_max)
        for i in range(round_num):
            index_temp = index[i * batch_size:(i + 1) * batch_size]
            h0 = get_h0(nn_architecture=nn_architecture, y0=y_tensor[index_temp, 0:1])
            output_train = model.forward(x_tensor[index_temp], h0=h0)
            loss = get_loss(
                loss_operator=loss_operator, y=y_tensor[index_temp], y_pre=output_train,
                extra_description=extra_description, y_std=y_std_tensor, lc=lc, l_epsp_c=l_epsp_c, device=device)
            optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()
            loss_train += loss.item() / round_num

        if (epoch + 1) % print_interval == 0:
            np.random.shuffle(index_val)
            index_val_temp = index_val[:batch_size * 50]
            improved_flag = False
            # with torch.no_grad():  # validation loss calculation
            h0 = get_h0(nn_architecture=nn_architecture, y0=y_val_tensor[index_val_temp, 0:1])
            output_val = model.forward(x_val_tensor[index_val_temp], h0=h0)
            loss_val = get_loss(
                loss_operator=loss_operator, y=y_val_tensor[index_val_temp],
                y_pre=output_val, extra_description=extra_description,
                y_std=y_std_tensor, lc=lc, l_epsp_c=l_epsp_c, device=device).item()
            if loss_val < min_loss:
                min_loss = loss_val
                improved_flag = True
                trial_num = 0
                # save the trained model
                torch.save(model, model_save_full_path)
            time_used = (time.time() - start_time) / 60.
            line = f"Epoch [{epoch + 1}/{num_epochs}], " \
                   f"Step [{i + 1}/{num_train_samples // batch_size}], " \
                   f"train_loss: {loss_train:.4e}" + '\t' + \
                   f"valid_loss: {loss_val:.4e}" + '\t' + \
                   f"min_Loss: {min_loss:.4e}" + '\t' + \
                   f"consumued_time: {time_used:.4e}(mins)" + '\t' + \
                   ('Improved!' if improved_flag else 'No imp. in %d epochs' % trial_num)
            if extra_description == '_cal':
                line += '\tK=%.3e G=%.3e q_yield=%.3e' % \
                        (model.K_norm * model.K_origin, model.G_norm * model.G_origin,
                         model.q_yield_norm * model.q_yield_origin)
            elif extra_description == '_deps_sig_epsp_mc':
                line += '\tK=%.3e G=%.3e A=%.3e B=%.3e epsilon_0=%.3e' % \
                        (model.mc.K_norm * model.mc.K_origin, model.mc.G_norm * model.mc.G_origin,
                         model.mc.A_norm * model.mc.A_origin, model.mc.B_norm * model.mc.B_origin,
                         model.mc.epsilon_norm * model.mc.epsilon_origin)
            echo(line)
            writeLine(fname=history_file_full_path, s=line + '\n', mode='a')

            # plot the prediction results for checking
            # if (epoch + 1) % (print_interval*10) == 0:
            #     for i in random.choices(list(range(len(index_val_temp))), k=2):
            #         plt.plot(output_val[i, :].detach().numpy())
            #         indexx = np.array(range(0, 200, 200//35))
            #         plt.scatter(indexx, y_val_tensor[index_val_temp][i, :, 0].detach().numpy()[indexx])
            #         plt.scatter(indexx, y_val_tensor[index_val_temp][i, :, 1].detach().numpy()[indexx])
            #         plt.scatter(indexx, y_val_tensor[index_val_temp][i, :, 2].detach().numpy()[indexx])
            #         plt.show()

        trial_num += 1
        if trial_num > patience_num:
            break


def get_h0(nn_architecture: str, y0: torch.Tensor):
    """

    :param nn_architecture:  str
    :param y0: the hidden before the first step    in shape of (num_samples, num_outputs)
    :return:
    """
    if nn_architecture == 'classic':
        # h0 = get_initial_hidden(output=y_train[index_temp], num_layers=num_layers)
        h0 = None
    elif nn_architecture == 'sig':
        h0 = y0
    else:
        raise ValueError('Please check if the nn_architecture (%s) is included' % nn_architecture)
    return h0


def get_loss(loss_operator, y_pre, y, extra_description, y_std, lc=1.0, l_epsp_c=1.0, device=torch.device('cpu')):
    """

    :param loss_operator:
    :param y_pre:  prediction
    :param y:      ground truth
    :param nn_architecture:
    :param y_std:
    :param lc:  is the loss coefficient of \sigma_{xy}, the larger the influence should be stronger
    :return:
    """
    if 'epsp' in extra_description:
        scalar = torch.tensor([1., lc, 1., l_epsp_c]).float().to(device=device)
    else:
        scalar = torch.tensor([1., lc, 1.]).float().to(device=device)
    if 'mc' in extra_description:
        scalar = torch.tensor([1., lc, 1.]).float().to(device=device)
        loss = loss_operator(y_pre[:, 1:, :3] / y_std[:3] * scalar, y[:, 1:, :3] / y_std[:3] * scalar)
        return loss
    elif 'epspvec' in extra_description:
        scalar = torch.tensor([1., lc, 1., l_epsp_c, l_epsp_c, l_epsp_c, l_epsp_c]).float().to(device=device)
        loss = loss_operator(y_pre[:, 1:] / y_std * scalar, y[:, 1:] / y_std * scalar)
        return loss

    loss = loss_operator(y_pre[:, 1:, :4] / y_std[:4] * scalar, y[:, 1:, :4] / y_std[:4] * scalar)
    return loss


def data_reading(num_samples, mode=None, voigt_flag=True,
                 cwd='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant'):
    if mode is None or mode == 'j2':
        fname = 'loading_results/mises_results_numSamples%d_5.npy' % num_samples
    elif mode == 'drucker':
        fname = 'loading_results/drucker/mises_results_numSamples%d_5.npy' % num_samples
    elif mode == 'j2_harden':
        fname = 'loading_results/mises_harden/mises_results_numSamples%d_6.npy' % num_samples
    else:
        raise
    fname = os.path.join(cwd, fname)
    if 'harden' not in mode:
        strain_paths_tensor, sig_numg, q_numg, eps_p_numg, eps_e_numg = load_npy(fname)
    else:
        strain_paths_tensor, sig_numg, q_numg, eps_p_numg, eps_e_numg, H_numg = load_npy(fname)
    echo('Data collected from %s' % fname)

    eps_p_norm_numg = np.linalg.norm(eps_p_numg, axis=(2, 3))

    """
    # code to check the eps_p_norm_numg is calculated in right way
    A = 0.2 * 2e6
    B = 0.5
    epsilon_0 = np.power(1e5 / A, 1 / B)
    H = A * (eps_p_norm_numg + epsilon_0)**B
    f = q_numg-H
    plt.plot(f[150], label='cal')
    plt.plot(q_numg[150] - H_numg[150], label='H', marker = '+')
    plt.legend();plt.tight_layout()
    plt.show()
    """

    if voigt_flag:
        strain_paths_vector = tensor2voigt(tensor=strain_paths_tensor)
        sig_numg_vector = tensor2voigt(tensor=sig_numg)
        eps_p_numg_vector = tensor2voigt(tensor=eps_p_numg)
        del strain_paths_tensor, sig_numg
        return strain_paths_vector, sig_numg_vector, eps_p_norm_numg, eps_p_numg_vector
    else:
        return strain_paths_tensor, sig_numg, q_numg, eps_p_numg, eps_e_numg


def get_save_dir_name(
        cwd, mode, num_samples, nn_architecture, extra_description, lc, len_sequence, l_epsp_c=1.0, num_deep_layer=None,
        extract_number: int = None
):
    save_dir = os.path.join(cwd, 'rnn_%s_numSamples%d_NN%s' %
                            (mode, num_samples, nn_architecture))
    if extra_description:
        save_dir += extra_description
    if lc:
        save_dir += '_lc%.1f' % lc
    if l_epsp_c and 'epsp' in extra_description:
        save_dir += '_lepsp%.1f' % l_epsp_c
    if extract_number is not None:
        save_dir += '_extractLen%d' % extract_number
    elif len_sequence:
        save_dir += '_len%d' % len_sequence
    if num_deep_layer and \
            ('fc' in extra_description or '_deps_sig_epspvec_split_classify' in extra_description):
        save_dir += '_deep%d' % num_deep_layer
    return save_dir


def data_preprocessing(strain_paths_vector, sig_numg_vector, save_dir, nn_architecture):
    deps_norm_median: float = 0.0
    if nn_architecture == 'classic':
        scalar_x = fit_save_scalar(strain_paths_vector, save_dir, name='x')
        scalar_y = fit_save_scalar(sig_numg_vector, save_dir, name='y')
    elif nn_architecture == 'sig':
        deps_vec = strain_paths_vector[:, 1:] - strain_paths_vector[:, :-1]

        # cal the  median norm of dstrain
        deps_norm = np.sqrt(deps_vec[:, :, 0] ** 2 + 2. * deps_vec[:, :, 1] ** 2 + deps_vec[:, :, 2] ** 2).reshape(-1)
        deps_norm_median, deps_norm_average = np.median(deps_norm), np.average(deps_norm)
        '''
        # cal the  median norm of dstrain
        deps_norm = np.sqrt(deps_vec[:, :, 0]**2 + 2.* deps_vec[:, :, 1]**2 + deps_vec[:, :, 2]**2 ).reshape(-1)
        deps_norm_median, deps_norm_average = np.median(deps_norm) , np.average(deps_norm)
        0.0013429818328662532, 0.00178003876731612
        '''
        # dsig_vec = sig_numg_vector[:, 1:] - sig_numg_vector[:, :-1]
        scalar_x = fit_save_scalar(deps_vec, save_dir, name='x')
        # scalar_y = fit_save_scalar(dsig_vec, os.path.join(save_dir, 'y.joblib'))
        scalar_y = fit_save_scalar(sig_numg_vector, save_dir, name='y')
    else:
        raise ValueError('Please check the nn_architecture (%s) is not involved now.' % nn_architecture)
    return scalar_x, scalar_y, deps_norm_median


def get_initial_hidden(output: np.ndarray, num_layers: int):
    """

    :param output:      in shape of  (num_samples, num_steps, num_outputs)
    :param num_layers:
    :return:
               total shape should be in shape of  (num_layers, num_samples, num_output)
               first step of the output      in shape of  (num_samples, num_ouput)
            'if 2 layers GRU,
            with the second GRU taking in outputs of the first GRU and computing the final results.'
    """
    num_samples, num_steps, num_outputs = output.shape
    h0 = torch.zeros([num_layers, num_samples, num_outputs])
    h0[0] = torch.from_numpy(output[:, 0, :])  # h0_1st
    return h0


def fit_save_scalar(dataset: np.ndarray, save_dir: str, name: str):
    """

    :param dataset: in shape of  (num_samples, num_step, num_features)
    :return:
    """
    num_samples = len(dataset)
    dataset_ = np.concatenate([dataset[i] for i in range(num_samples)], axis=0)

    scalar = StandardScaler()
    scalar.fit(dataset_)
    joblib.dump(scalar, filename=os.path.join(save_dir, '%s.joblib' % name))

    scalar_minmax = MinMaxScaler()
    scalar_minmax.fit(dataset_)
    joblib.dump(scalar_minmax, filename=os.path.join(save_dir, '%s_minmax.joblib' % name))

    # q's scalar
    if name == 'y':
        q = get_q_2d_arr(dataset_)
        scalar_q = StandardScaler()
        scalar_q.fit(q)
        joblib.dump(scalar_q, filename=os.path.join(save_dir, 'q.joblib'))

        scalar_q_minmax = MinMaxScaler()
        scalar_q_minmax.fit(q)
        joblib.dump(scalar_q_minmax, filename=os.path.join(save_dir, 'q_minmax.joblib'))

    return scalar


def normlization(dataset: np.ndarray, scalar: StandardScaler, reverse: bool):
    """

    :param dataset:  in shape of  (num_samples, num_step, num_features)
    :param scalar:
    :return:
    """
    num_samples = len(dataset)
    if not reverse:
        dataset_normed = np.array([scalar.transform(dataset[i]) for i in range(num_samples)])
    else:
        dataset_normed = np.array([scalar.inverse_transform(dataset[i]) for i in range(num_samples)])
    return dataset_normed


def restore_model(model_full_path, device=torch.device('cpu')):
    echo('Model restored from %s' % model_full_path)
    model = torch.load(model_full_path).to(device=device)
    return model


def call_and_prediction(
        num_samples, nn_architecture, len_sequence: int, train_sample_flag=False,
        extra_description=None, lc=None, mode='j2', l_epsp_c=None, num_deep_layer=None,
        extract_number=None, hidden_size:int=None, num_samples_used=None):
    """
            x in shape of (numg, step, features)
            y in shape of (numg, step, features)

    :param model_full_path:
    :return:
    """
    if extra_description == '_deps_sig_epsp_split':
        gpu_flag = False
    else:
        gpu_flag = True
    device = findDevice(useGPU=gpu_flag)
    cwd = os.getcwd()
    save_dir = get_save_dir_name(
        cwd=cwd, mode=mode,
        num_samples=num_samples if num_samples_used is None else num_samples_used,
        nn_architecture=nn_architecture,
        extra_description=extra_description, lc=lc, len_sequence=len_sequence, l_epsp_c=l_epsp_c,
        num_deep_layer=num_deep_layer, extract_number=extract_number, )

    # save_dir += '_perfect'

    if 'mc' not in extra_description:
        model = restore_model(model_full_path=os.path.join(save_dir, 'rnn.pt'), device=device)
        try:
            model.x_std = model.x_std.to(device)
            model.y_std = model.y_std.to(device)
            model.x_mean = model.x_mean.to(device)
            model.y_mean = model.y_mean.to(device)
        except:
            pass
    else:
        model = rnn_net_4_constitutive_fixed_sig_general(
            num_inputs=3, num_outputs=3, hidden_size=20,
            extra_description=extra_description, save_dir=save_dir, device=device)
    x, y, eps_p_norm, eps_p_vec = data_reading(
        num_samples=201 if not train_sample_flag else num_samples,
        mode=mode, voigt_flag=True)  # make sure use the samples not involved in the training
    num_steps = len(x[0])
    if 'epsp' in extra_description:
        y = np.concatenate((y, eps_p_norm[:, :, np.newaxis]), axis=-1)
    if 'mc' in extra_description or 'epspvec' in extra_description:
        y = np.concatenate((y, eps_p_vec), axis=-1)
    scalar_x = joblib.load(os.path.join(save_dir, 'x.joblib'))
    scalar_y = joblib.load(os.path.join(save_dir, 'y.joblib'))
    # x_normed = normlization(x, scalar=scalar_x, reverse=False)
    # y_normed = normlization(y, scalar=scalar_y, reverse=False)
    h0 = get_h0(nn_architecture=nn_architecture, y0=torch.from_numpy(y[:, 0:1]).float()).to(device)
    if h0 is not None:
        h0 = h0.float()
    x_tensor = torch.from_numpy(x).float().to(device)

    if nn_architecture == 'classic' or nn_architecture == 'sig':
        if 'extract' in extra_description:
            y_pre_list = [h0[:, 0]]
            h0_1 = torch.zeros(size=(201, hidden_size), device=device)
            dx_tensor = x_tensor[:, 1:] - x_tensor[:, :-1]
            for i in range(num_steps - 1):
                deps = dx_tensor[:, i]
                y_pre_tensor_single, h0_1 = model.get_sig_extract(deps, h0_1=h0_1)
                y_pre_list.append(y_pre_tensor_single)
            y_pre_tensor = torch.stack(y_pre_list)

            y_pre_tensor = torch.transpose(y_pre_tensor, 0,
                                           1)  # in shape of (num_samples, num_steps, hidden_size)      q
        else:
            with torch.no_grad():
                y_pre_tensor = model.forward(x_tensor, h0=h0)
    else:
        raise ValueError('Please check if the nn_architecture (%s) is included' % nn_architecture)
    loss_numg = np.mean(
        np.mean(((y_pre_tensor.cpu().detach().numpy()[:, :, :3] - y[:, :, :3]) / scalar_y.scale_[:3]) ** 2, axis=1),
        axis=1)
    y_pre = y_pre_tensor.cpu().detach().numpy()

    # # plot
    # for numg_plot in np.random.randint(0, len(x) + 1, 3 if 'mc' not in extra_description else 20):
    #     plot_prediction(y[numg_plot, :, :4], y_pre[numg_plot], numg_plot, mode=mode)

    # plot the worst one
    worst_index = np.argmax(loss_numg)
    plot_prediction(
        y[worst_index, :, :4], y_pre[worst_index], numg_plot=worst_index, s='worst', mode=mode,
        nn_architecture=nn_architecture, extra_description=extra_description)
    # plot the best one
    best_index = np.argmin(loss_numg)
    plot_prediction(
        y[best_index, :, :4], y_pre[best_index], numg_plot=best_index, s='best', mode=mode,
        nn_architecture=nn_architecture, extra_description=extra_description)

    # echo the overall loss
    mean_loss, worst_loss = np.mean(loss_numg), loss_numg[worst_index]
    echo('The overall test MSE is %.2e and the worst is %.2e' % (mean_loss, worst_loss))

    # plot the loss distribution
    sns.displot(loss_numg, kind="kde")
    # plt.plot([0, 1], [0, 1])
    xlim_min, xlim_max = plt.xlim()
    ylim_min, ylim_max = plt.ylim()
    posi_x = xlim_min + 0.95 * (xlim_max - xlim_min)
    posi_y = ylim_min + 0.7 * (ylim_max - ylim_min)
    s = 'Overall MSE: %.1e \nthe worst: %.1e' % (mean_loss, worst_loss)
    plt.text(
        posi_x, posi_y, s,
        verticalalignment='top', horizontalalignment='right', size=22, color='blue',
        bbox=dict(
            boxstyle="round",
            ec=(1., 0.5, 0.5),
            fc=(1., 0.8, 0.8), alpha=0.5,
        ))
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.tight_layout()
    # plt.show()
    fig_name = "./figs/%s_%s_%s_distribution.png" % (mode, nn_architecture, extra_description)
    plt.savefig(fig_name, dpi=200)
    print("fig saved as %s" % fig_name)


def truncated_sequence_via_time(
        x_sequence: np.ndarray, y_sequence: np.ndarray, len_sequence: int):
    """

    :param x_sequence:  (num_samples, num_stepm, num_features)
    :param y_sequence:
    :param len_sequence:
    :return:
    """
    num_samples, num_steps, _ = x_sequence.shape
    x_truncated, y_truncated = [], []
    num_sequences = num_steps // len_sequence
    for numg in range(num_samples):
        for j in range(num_sequences):
            x_truncated.append(x_sequence[numg, j * len_sequence:(j + 1) * len_sequence])
            y_truncated.append(y_sequence[numg, j * len_sequence:(j + 1) * len_sequence])
    x_truncated_array = np.array(x_truncated)
    y_truncated_array = np.array(y_truncated)
    return x_truncated_array, y_truncated_array


def plot_prediction(y, y_pre, numg_plot=0, s=None, mode=None, nn_architecture=None, extra_description=None):
    num_step, num_features = y.shape
    y[:, :3], y_pre[:, :3] = y[:, :3] / 1e6, y_pre[:, :3] / 1e6
    num_points = 20
    index_scatter = range(0, num_step, num_step // num_points)
    fig = plt.figure(figsize=[6.4, 4.8])
    fig.add_subplot(211)
    legend_temp = mode
    if mode == "j2":
        legend_temp = "$J_2$"
    elif mode == "drucker":
        legend_temp = "Drucker"
    elif mode == "j2_harden":
        legend_temp = "$J_2$-harden"

    plt.scatter(index_scatter, y[::(num_step // num_points), 0], label='$\sigma_{00}$')
    plt.scatter(index_scatter, y[::(num_step // num_points), 1], label='$\sigma_{01}$')
    plt.scatter(index_scatter, y[::(num_step // num_points), 2], label='$\sigma_{11}$')
    plt.plot(range(num_step), y_pre[:, 0]) #, label='$\sigma_{00}$ Pr.')
    plt.plot(range(num_step), y_pre[:, 1]) #, label='$\sigma_{01}$ Pr.')
    plt.plot(range(num_step), y_pre[:, 2]) #, label='$\sigma_{11}$ Pr.')
    # plt.xlabel('Load step')
    plt.ylabel('MPa')
    plt.xticks([])
    plt.legend()

    # plot the yield stress
    color_ax = 'purple'
    ax = fig.add_subplot(212)
    q, q_pre = get_q_2d_arr(y[:, :3]), get_q_2d_arr(y_pre[:, :3])
    plt.scatter(index_scatter, q[::(num_step // num_points)], label='$q$', color=color_ax)
    plt.plot(range(num_step), q_pre,  color=color_ax)
    plt.xlabel('Load step')
    plt.ylabel('MPa')
    ax.yaxis.label.set_color(color_ax)
    ax.tick_params(axis='y', colors=color_ax)
    plt.legend(loc='upper left')
    if num_features == 4:  # 绘制塑性应变的大小
        ax2 = plt.twinx(ax)
        plt.scatter(
            index_scatter, y[::(num_step // num_points), 3],
            marker='+', color='b', s=60,
            # label=r'$\| \epsilon^p \|$ %s' % mode
        )
        plt.plot(range(num_step), y_pre[:, 3], label=r'$\| \epsilon^p \|$', color='b', linestyle='--')
        ylabel = r'$\| \epsilon^p \|$'
        ax2.yaxis.label.set_color('b')
        ax2.tick_params(axis='y', colors='b')
        plt.legend(loc='lower right')

    title = 'Gauss point %d' % numg_plot
    # if "best" in s:
    #     s = "Best prediction"
    # elif "worst" in s:
    #     s = "Worst prediction"
    if not s:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if nn_architecture:
        fig_name = "./figs/%s_%s_%s_%s.png" % (mode, nn_architecture, extra_description, s)
        plt.savefig(fig_name, dpi=200)
        print("fig saved as %s" % fig_name)
    else:
        plt.show()

    return


if __name__ == '__main__':

    """
        j2 physics extended             rnn_j2_harden_numSamples201_NNsig_deps_sig_epsp_split_lc0.8_lepsp1.5_len20
        j2_harden  extract data_driven  rnn_j2_harden_numSamples999_NNsig_deps_sig_extract_lc1.0_extractLen50   
        j2_harden  DNN                  rnn_j2_harden_numSamples999_NNsig_deps_sig_epspvec_split_classify_lc1.0_lepsp1.0_len2_deep10  # perfect

        """
    axes = {'labelsize': 'medium'}
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
    # legend = {'fontsize': 'medium'}
    # lines = {'linewidth': 3,
    #          'markersize': 7}
    mpl.rc('font', **font)
    mpl.rc('axes', **axes)
    # mpl.rc('legend', **legend)
    # mpl.rc('lines', **lines)

    mode = 'j2_harden'  # 'j2' 'drucker'  j2_harden
    mode_list = ['j2_harden']
    nn_architecture = 'sig'  # 'classic' , 'sig'
    cwd = os.getcwd()
    print('\n' + cwd + '\n')
    # for extra_description in extra_description_list[1:]:
    #     for num_samples in [201]:
    #         main(num_samples=num_samples, nn_architecture=nn_architecture, extra_description=extra_description)
    extra_description = '_deps_sig_epspvec_split_classify'
    # _deps_sig  _deps_sig_fc _deps_sig_epsp _deps_sig_epsp_fc _deps_sig_epsp_split
    # _deps_sig_epsp_split_fc  _deps_sig_epsp_mc  _deps_sig_epsp_split_classify
    # _deps_sig_epspvec_split_classify
    # _deps_sig_simple
    # _deps_sig_muLtiGRU
    # _deps_datadriven_sig_epspvec

    extra_description_list = ['_deps_sig_extract']

    num_deep_layer_list = [2]
    num_deep_layer = 10
    extract_number = None  # 在数据驱动中用50 (从长度为200的序列抽取 extract_number), 普通案例为None
    hidden_size = 30

    lc = 1.0
    l_epsp_c = 1.0
    # lc = 0.8
    # l_epsp_c = 1.5
    len_sequence = 2 if ('fc' not in extra_description) and ('_mc' not in extra_description) else 2
    # train_flag = True  #
    train_flag = False
    if train_flag:
        for i in extra_description_list:
            for j in mode_list:
                for num_deep_layer in num_deep_layer_list:
                    main(num_samples=999, nn_architecture=nn_architecture, extra_description=i,
                         hidden_size=hidden_size,
                         lc=lc, len_sequence=len_sequence,
                         n_batch_max=80 if 'mc' not in extra_description else 10,
                         batch_size=1024, mode=j, l_epsp_c=l_epsp_c, num_deep_layer=num_deep_layer,
                         extract_number=extract_number)
    else:
        call_and_prediction(
            num_samples=999, nn_architecture=nn_architecture,
            extra_description=extra_description, lc=lc, len_sequence=len_sequence, train_sample_flag=False,
            mode=mode, l_epsp_c=l_epsp_c, num_deep_layer=num_deep_layer, extract_number=extract_number,
            hidden_size=hidden_size)
