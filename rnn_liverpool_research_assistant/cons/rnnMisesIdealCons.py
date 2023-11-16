import numpy as np
import torch
from constitutive import ConstitutiveMask
from rnn_model import rnn_net_4_constitutive_fixed_sig_general
from rnn_model_trainer_j2 import restore_model
import joblib, os
from utilSelf.general import echo
from utils_constitutive import voigt_2_tensor


class rnnMisesIdealConstitutive(ConstitutiveMask):
    """
            Compression is positive
    """
    def __init__(
            self, p0, numg, save_path, rho, explicitFlag=True, ndim=2, hidden_size=30, step_scalar=0, mode='rnn_misesideal',
            # NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_j2_numSamples201_NNsig_deps_sig_lc1.0_len20',        # mises
            # NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_j2_numSamples999_NNsig_deps_sig_fc_lc1.0_len2',         # mises_fc deep_network
            # NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_drucker_numSamples201_NNsig_deps_sig_p_lc1.0_len10',  # drucker with p
            # NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_drucker_numSamples999_NNsig_deps_sig_fc_lc1.0_len2',  # drucker-fc deep network
            # NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_j2_harden_numSamples999_NNsig_deps_sig_epspvec_split_classify_lc1.0_lepsp1.0_len2_deep10_perfect',    # j2_harden  deep network
            NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_j2_harden_numSamples999_NNsig_deps_sig_extract_lc1.0_extractLen50',    # j2_harden  extract data_driven
            # NN_dir='/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/rnn_j2_numSamples201_NNsig_cal',
    ):
        ConstitutiveMask.__init__(
            self, save_path=save_path, p0=1.0e5, ndim=2, explicitFlag=True,
            name=mode, numg=numg, cons=None, rho=rho, nump=1)
        self.numg = numg
        self.mode = mode
        self.voigt_len = 3 if self.ndim == 2 else 6
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.hidden_size = hidden_size
        self.device = torch.device('cuda') if \
            (self.mode == 'rnn_misesideal' or self.mode == 'rnn_drucker' or
                    self.mode == 'rnn_mises_harden_epsp' or self.mode=='rnn_drucker_fc' or self.mode == 'rnn_misesideal_fc'
             or self.mode == 'rnn_mises_harden_extract') \
            else torch.device('cpu')
        self.NN_sig = restore_model(os.path.join(NN_dir, 'rnn.pt'), device=self.device)
        self.extract_flag = False
        if "extract" in NN_dir:
            self.extract_flag = True
            self.deps_norm_median = self.NN_sig.train_x_norm_median
        if self.mode == 'rnn_mises' or self.mode == 'rnn_misesideal' or self.mode == 'rnn_mises_harden':
            self.NN_sig_cal = self.NN_sig.get_sig
        elif self.mode == 'rnn_drucker':  # as drucker-prager is mean stress dependent
            self.NN_sig_cal = self.NN_sig.get_sig_p if '_deps_sig_p' in NN_dir else self.NN_sig.get_sig
            if '_deps_sig_fc' in NN_dir:
                self.NN_sig_cal = self.NN_sig.get_sig_fc
        elif self.mode == 'rnn_drucker_fc' or self.mode == 'rnn_misesideal_fc' or \
                self.mode == 'rnn_mises_harden_fc':     #
            self.NN_sig_cal = self.NN_sig.get_sig_fc
        elif self.mode == 'rnn_mises_harden_epsp':
            self.NN_sig_cal = self.NN_sig.get_sig_epspvec_split_classify
        elif self.mode == 'rnn_mises_harden_epsp_fc':
            self.NN_sig_cal = self.NN_sig.get_sig_epsp_split_fc
        elif self.mode == 'rnn_mises_harden_extract':
            self.extract_flag = True
        else:
            raise ValueError('%s is not included here, please check' % self.mode)

        self.scalar_x = joblib.load(os.path.join(NN_dir, 'x.joblib'))
        self.scalar_y = joblib.load(os.path.join(NN_dir, 'y.joblib'))
        self.x_mean = self.scalar_x.mean_
        self.x_std = self.scalar_x.scale_
        self.sig_vec = np.array([[p0, 0., p0] for _ in range(self.numg)])
        self.sig = voigt_2_tensor(self.sig_vec)
        self.step_scalar = step_scalar
        # self.hidden_torch = torch.zeros(self.numg, self.hidden_size)
        # self.hidden_torch = torch.zeros(self.numg, self.hidden_size)
        self.sig_torch = torch.from_numpy(
            self.sig_vec if 'epsp' not in self.mode else
            np.concatenate((self.sig_vec, np.zeros(shape=[self.numg, 4])), axis=1)).float()

        self.h0_1 = torch.zeros(size=[self.numg, 30], device=self.device)

        # material parameters
        self.K, self.G, self.q_yield = 1250000., 833333., 1e5

    def solver(self, deps):
        '''

            deps_s: in shape of [numg, 2, 2]   in the input strain tensor, the compression is positive
            return: sig_geo
        '''
        # NOTE: compression is positive in the rnn model
        deps_s_voigt = np.delete(deps.reshape([self.numg, 4]), [2], axis=1)
        eps_s = self.eps - deps_s_voigt
        num_sub_step = 1
        if not self.extract_flag:
            if self.step_scalar:
                d_adapt = torch.ones(self.numg).to(self.device) * self.step_scalar
            else:
                deps_norm = np.sqrt(deps_s_voigt[:, 0] ** 2 + 2. * deps_s_voigt[:, 1] ** 2 + deps_s_voigt[:, 2] ** 2)
                d_adapt = torch.from_numpy(self.deps_norm_median / deps_norm).float().to(self.device)
                d_adapt = torch.nan_to_num(d_adapt, nan=1., posinf=1., neginf=1.)
            with torch.no_grad():
                sig_torch = self.sig_torch.to(self.device) if 'mises' in self.mode else -self.sig_torch.to(self.device)
                deps_torch = torch.from_numpy(deps_s_voigt).float().to(self.device) \
                    if 'mises' in self.mode else -torch.from_numpy(deps_s_voigt).float().to(self.device)
                for i in range(num_sub_step):
                    # sig_torch = (self.NN_sig_cal(deps=deps_torch * d_adapt, h0=sig_torch)-sig_torch)/d_adapt+sig_torch
                    d_temp = self.NN_sig_cal(deps=torch.einsum('ij, i->ij', deps_torch, d_adapt), h0=sig_torch)-sig_torch
                    sig_torch = torch.einsum('ij, i->ij', d_temp, 1/d_adapt) + sig_torch
                sig_vec = sig_torch[:, :3].cpu().detach().numpy()

        else:
            with torch.no_grad():
                deps_torch = -torch.from_numpy(deps_s_voigt).float().to(self.device)  # compression is negative
                sig, self.h0_1 = self.NN_sig.get_sig_extract(deps=deps_torch, h0_1=self.h0_1)
                sig_vec = -sig[:, :3].cpu().detach().numpy()
                sig_torch = - sig

        sences = [eps_s, self.eps_abs + np.abs(deps_s_voigt),
                  sig_vec if 'mises' in self.mode else -sig_vec,
                  sig_torch if 'mises' in self.mode else -sig_torch,
                  0.,
                  # 0., 0.,
                  ]

        sig_geo = self.assemble_sig_ml(sig_vector=sig_vec if 'mises' in self.mode else -sig_vec)
        if self.explicitFlag:
            self.update(scenes=sences)
            return sig_geo
        else:
            raise ValueError('The implicit is not included currently')

    def assemble_sig_ml(self, sig_vector):
        """

        :param sig_vector:  in shape of (numg, (00, 01, 11))
        :return:
        """
        sig_tensor = np.zeros([self.numg, 2, 2])
        for i in range(self.numg):
            sig_tensor[i, 0, 0] = sig_vector[i, 0]
            sig_tensor[i, 0, 1] = sig_tensor[i, 1, 0] = sig_vector[i, 1]
            sig_tensor[i, 1, 1] = sig_vector[i, 2]
        return sig_tensor

    def update(self, scenes):
        """

        :param scenes: eps_s, self.eps_abs + np.abs(deps_s_voigt), sig_vec, sig_normed_torch, hidden_torch
        :return:
        """
        self.eps = scenes[0]
        self.eps_abs = scenes[1]
        self.sig_vec = scenes[2]
        self.sig_torch = scenes[3]
        self.hidden_torch = scenes[4]

    def return2initial(self):
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.sig_vector = np.array([[-self.p0, 0, -self.p0] for _ in range(self.numg)])