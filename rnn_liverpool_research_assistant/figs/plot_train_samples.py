import os.path
import numpy as np
import matplotlib.pyplot as plt
from rnn_model_trainer_j2 import data_reading
from utilSelf.general import check_mkdir, echo


def main(num_samples=201, mode='drucker'):
    # datasets reading
    strain_paths_tensor, sig_numg, q_numg, eps_p_numg, eps_e_numg =\
        data_reading(num_samples=num_samples, mode=mode, voigt_flag=False)
    save_path = 'loading_results/drucker/drucker_plot_%d' % num_samples
    check_mkdir(save_path)
    plot_samples(strain_paths_tensor, sig_numg, q_numg, eps_p_numg, eps_e_numg, mode=mode, save_path=save_path)


def plot_samples(strain_paths_tensor, sig_numg, q_numg, eps_p_numg, eps_e_numg, mode, H_numg=None, save_path=None):
    for i in range(len(sig_numg)):
        fig = plt.figure(figsize=(6.4 * 1.5, 4.8))
        # plt.title('Gaussian point %d' % i)
        # eps
        fig.add_subplot(221)
        plt.plot((eps_p_numg + eps_e_numg)[i, :, 0, 0], label=r'$\epsilon_{00}$')
        plt.plot((eps_p_numg + eps_e_numg)[i, :, 0, 1], label=r'$\epsilon_{01}$')
        plt.plot((eps_p_numg + eps_e_numg)[i, :, 1, 1], label=r'$\epsilon_{11}$')
        plt.plot(strain_paths_tensor[i, :, 0, 0], label=r'$\epsilon_{00}$ origin')
        plt.plot(strain_paths_tensor[i, :, 0, 1], label=r'$\epsilon_{01}$ origin')
        plt.plot(strain_paths_tensor[i, :, 1, 1], label=r'$\epsilon_{11}$ origin')
        plt.legend()
        plt.xlabel('Loading step')

        # sig
        fig.add_subplot(222)
        plt.plot(sig_numg[i, :, 0, 0] / 1e6, label=r'$\sigma_{00}$')
        plt.plot(sig_numg[i, :, 0, 1] / 1e6, label=r'$\sigma_{01}$')
        plt.plot(sig_numg[i, :, 1, 1] / 1e6, label=r'$\sigma_{11}$')
        plt.xlabel('Loading step')
        plt.ylabel('(MPa)')
        plt.legend()

        # q
        ax = fig.add_subplot(223)
        if mode == 'misesideal':
            ax.plot(q_numg[i] / 1e6)
            ax.set_ylabel('$q$ (MPa)')
        elif mode == 'drucker':
            ax.plot(q_numg[i] / 1e6)
            ax.set_ylabel('$q$ (MPa)')
            ax2 = ax.twinx()
            ax2.plot((q_numg[i] + 2.*np.sin(np.pi*15./180.) * np.trace(sig_numg[i], axis1=1, axis2=2) / 2.) / 1e6, 'b-')
            ax2.set_ylabel('yield value (MPa)', color='b')
        elif mode == 'mises_harden':
            ax.plot((q_numg[i]-H_numg[i]) / 1e6)
            ax.set_ylabel('$q$ (MPa)')
        else:
            print("%s is not included" % mode)
            raise
        plt.xlabel('Loading step')

        # eps_p
        fig.add_subplot(224)
        plt.plot(eps_p_numg[i, :, 0, 0], label=r'$\epsilon^p_{00}$')
        plt.plot(eps_p_numg[i, :, 0, 1], label=r'$\epsilon^p_{01}$')
        plt.plot(eps_p_numg[i, :, 1, 1], label=r'$\epsilon^p_{11}$')
        plt.xlabel('Loading step')
        plt.legend()
        fig.suptitle('Guass point %d' % i)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            fname = os.path.join(save_path, '%s_%d.png' % (mode, i))
            plt.savefig(fname)
            echo('file saved as %s' % fname)
        plt.close()


if __name__ == '__main__':
    main(num_samples=201, mode='drucker')
