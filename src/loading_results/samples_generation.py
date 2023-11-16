
import os
import matplotlib.pyplot as plt
import numpy as np
from cons.utils_constitutive import voigt_2_tensor_high
from utilSelf.general import check_mkdir, echo
from figs.plot_train_samples import plot_samples


class samplesGenerator:
    def __init__(self, strain_file_path, mode):
        self.mode = mode
        if mode == 'misesideal':
            from cons.vonmisesConsIdeal import vonmisesIdealSingle
            self.cons = vonmisesIdealSingle(p0=0., nu=0.2, E=2e6, yield_stress0=1e5)
        elif mode == 'drucker':
            from cons.DruckerPragerCons import druckerPragerSingle
            self.cons = druckerPragerSingle()
        elif mode == 'mises_harden':
            from cons.vonMisesConsHarden import MisesHardenSingle
            self.cons = MisesHardenSingle(p0=0., nu=0.2, E=2e6)
        else:
            print('%s is not included' % mode)
            raise
        # load strain paths
        self.strain_paths_tensor = voigt_2_tensor_high(self.get_strain_paths(strain_file_path))
        self.num_samples, self.load_step = len(self.strain_paths_tensor), len(self.strain_paths_tensor[0])

    def get_strain_paths(self, path):
        with open(path, mode='rb') as f:
            paths = np.load(f)
        return paths

    def load(self):
        deps_path = np.zeros_like(self.strain_paths_tensor)
        deps_path[:, 1:, :, :] = self.strain_paths_tensor[:, 1:] - self.strain_paths_tensor[:, :-1]

        # test: plot the first 6 before loadings
        if 'harden' not in self.mode:
            sig_numg, q_numg, eps_p_numg, eps_e_numg = self.cons.prediction_numg(
                deps_numg=deps_path[:6])
            H_numg, eps_p_norm = None, None
        else:
            sig_numg, q_numg, eps_p_numg, eps_e_numg, H_numg = self.cons.prediction_numg(
                deps_numg=deps_path[:6])
        '''
        
        eps = self.strain_paths_tensor[0];
        sig = sig_numg[0];
        val, vec = np.linalg.eig(eps)
        val_sig, vec_sig = np.linalg.eig(sig)
        '''

        plot_samples(
            strain_paths_tensor=self.strain_paths_tensor[:6],
            sig_numg=sig_numg, q_numg=q_numg, eps_p_numg=eps_p_numg,
            eps_e_numg=eps_e_numg, mode=self.mode,
            H_numg=H_numg,
            save_path=None)

        # load simulation
        # sig_numg, q_numg, eps_p_numg, eps_e_numg = self.cons.prediction_multiprocess(deps_numg=deps_path)
        datasets = self.cons.prediction_multiprocess(deps_numg=deps_path)
        return datasets

    def save2npy(self, num_samples, *args):
        '''
            save the the loading results as npy
        :param args: every of them should be in shape of  (numg, step, features)
        :return:
        '''
        save_path = self.mode
        check_mkdir(save_path)
        echo("file saved in %s" % save_path)
        file_name = os.path.join(save_path, 'numSamples%d_multiprocess_%d.npy' % (num_samples, len(args)))
        with open(file_name, mode='wb') as f:
            for i in args:
                np.save(f, i)
        echo('\nLoding results saved as %s' % file_name)


if __name__ == '__main__':
    num_samples = 201  # 999
    mode = 'misesideal'   # misesideal drucker mises_harden   three constitutives
    samples_gen = samplesGenerator(
        strain_file_path='random_strain_path/paths_%d.npy' % num_samples,
        mode=mode)
    datasets = samples_gen.load()  # Loading Sample Preparation
    samples_gen.save2npy(num_samples, samples_gen.strain_paths_tensor, *datasets)  # save data
    print()
