import joblib
import os
import torch
import torch.nn as nn

from FEMxML.rnn_liverpool_research_assistant.utils_rnn_cons import get_q_2d


class rnn_cons_base(nn.Module):
    def __init__(self, save_dir, device=torch.device('cpu')):
        super(rnn_cons_base, self).__init__()
        """

        :param num_inputs:
        :param num_outputs:
        :param device:
        :param num_layers_fc:   if num_layers_fc=0, means there are 1 layer used to
                                shift (..., num_inputs+hidden_size) to (..., num_node_fc) and 1 layer used to
                                shift (..., num_node_fc)            to (..., num_outputs)
                                no mid layers with shape of (num_node_fc, num_node_fc)
        :param num_node_fc:
        :param hidden_size:
        """
        self.device = device
        self.save_dir = save_dir
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.x_mean, self.x_std, self.y_mean, self.y_std, self.y_min, self.y_max, \
        self.q_mean, self.q_std, self.q_min, self.q_max = self.get_scalar()

    def get_scalar(self):
        scalar_x = joblib.load(os.path.join(self.save_dir, 'x.joblib'))
        scalar_y = joblib.load(os.path.join(self.save_dir, 'y.joblib'))
        scalar_y_minmax = joblib.load(os.path.join(self.save_dir, 'y_minmax.joblib'))
        scalar_q = joblib.load(os.path.join(self.save_dir, 'q.joblib'))
        scalar_q_minmax = joblib.load(os.path.join(self.save_dir, 'q_minmax.joblib'))
        x_mean = scalar_x.mean_
        # x_var = scalar_x.var_   np.sqrt(x_var)-x_std
        # NOTE: add the extra coefficients to the std to make the magnitudes of the input and output more sensible
        x_std = scalar_x.scale_
        y_mean = scalar_y.mean_
        y_std = scalar_y.scale_ * 6.
        y_min = scalar_y_minmax.data_min_
        y_max = scalar_y_minmax.data_max_
        q_mean = scalar_q.mean_
        q_std = scalar_q.scale_ * 6.
        q_min = scalar_q_minmax.data_min_
        q_max = scalar_q_minmax.data_max_
        return torch.from_numpy(x_mean).float().to(self.device), torch.from_numpy(x_std).float().to(self.device), \
               torch.from_numpy(y_mean).float().to(self.device), torch.from_numpy(y_std).float().to(self.device), \
               torch.from_numpy(y_min).float().to(self.device), torch.from_numpy(y_max).float().to(self.device), \
               torch.from_numpy(q_mean).float().to(self.device), torch.from_numpy(q_std).float().to(self.device), \
               torch.from_numpy(q_min).float().to(self.device), torch.from_numpy(q_max).float().to(self.device),

    def normalize(self, x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor, reverse_flag):
        if not reverse_flag:
            x_normed = (x - x_mean) / x_std
        else:
            x_normed = x * x_std + x_mean
        return x_normed

    def get_initial_layers(self, num_inputs, num_outputs, num_node_fc, num_layers_fc, hidden_size=0):
        layers = []
        layers.append(nn.Linear(num_inputs + hidden_size, num_node_fc))
        for i in range(num_layers_fc):
            layers.append(nn.Linear(num_node_fc, num_node_fc))
        layers.append(nn.Linear(num_node_fc, num_outputs))
        return layers

    def forward_fc(self, x):
        for i in range(len(self.fc_list) - 1):
            x = self.activation(self.fc_list[i](x))
        x = self.fc_list[-1](x)
        return x

    def forward_fc_1(self, x):
        for i in range(len(self.fc_list_1) - 1):
            x = self.activation(self.fc_list_1[i](x))
        x = self.fc_list_1[-1](x)
        return x

    def model_capacity(self):
        """
        Prints the number of parameters and the number of layers in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        line = "\t\t\tThe number of layers in the model: %d" % num_layers + '\n' + \
               "\n\t\t\tThe number of learnable parameters in the model: %d" % number_of_learnable_params
        return line


class rnn_net_4_constitutive(rnn_cons_base):
    def __init__(self, num_inputs, num_outputs, num_layers, hidden_size, device, save_dir):
        super(rnn_net_4_constitutive, self).__init__(save_dir=save_dir, device=device)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, device=self.device,  # (numg, step, n_features)
        )
        self.fc = nn.Linear(hidden_size, num_outputs, device=self.device)
        # self.pre_process_fc = nn.Linear(num_outputs, hidden_size)

    def forward(self, x, h0=None):
        '''

        :param x:  in format of   (num_sample, steps, num_features)         float
        :param h0: in format of   (num_layers, num_sample, num_features)    float
                        hidden state should be defined outside the function
                        or this will be defined as zero matrix (which will
                        not be proper wrong because of the normalization)
        :return:
        '''
        deps = x / self.x_std
        hidden_state = self.init_hidden(num_sample=len(x))
        output, hidden_state = self.gru(deps, hidden_state)
        output1 = self.fc(output)
        sig = output1 * self.y_std[:3] + self.y_mean[:3]
        return sig

    def init_hidden(self, num_sample):
        """
        
        :param num_sample:  
        :param h0:         in shape of (num_layers, num_samples, num_output)
        :return:           in shape of (num_layers, num_sample, hidden_size)
        """
        hidden = torch.zeros(self.num_layers, num_sample, self.hidden_size).to(self.device)
        return hidden


class rnn_net_4_constitutive_fixed_sig_general(rnn_cons_base):
    def __init__(
            self,
            num_inputs, num_outputs, device, hidden_size, extra_description, save_dir, num_deep_layer=None,
            train_x_norm_median: float = None,
    ):
        super(rnn_net_4_constitutive_fixed_sig_general, self).__init__(save_dir=save_dir, device=device)
        self.num_inputs, self.num_outputs = num_inputs, num_outputs
        self.hidden_size = hidden_size
        self.extra_description = extra_description

        # used for implementing the adaptive loading step
        self.train_x_norm_median = train_x_norm_median

        # define the architecture of the network
        if self.extra_description == '_deps_sig':  # good
            self.grucell = torch.nn.GRUCell(
                input_size=self.num_inputs + 3, hidden_size=self.num_outputs + 4, device=self.device)
        elif self.extra_description == '_deps_sig_extract':
            self.grucell_1 = torch.nn.GRUCell(
                input_size=self.num_inputs, hidden_size=self.hidden_size, device=self.device)
            self.grucell_2 = torch.nn.GRUCell(
                input_size=self.num_inputs, hidden_size=self.hidden_size, device=self.device)
            self.grucell_3 = torch.nn.GRUCell(
                input_size=self.num_inputs, hidden_size=self.hidden_size, device=self.device)
            # self.grucell_4 = torch.nn.GRUCell(
            #     input_size=self.num_inputs, hidden_size=self.hidden_size, device=self.device)
            self.linear = torch.nn.Linear(in_features=self.hidden_size, out_features=self.num_outputs,
                                          device=self.device)
        elif self.extra_description == '_deps_datadriven_sig_epspvec':
            self.grucell = torch.nn.GRUCell(
                input_size=self.num_inputs, hidden_size=self.hidden_size, device=self.device)
            self.grucell_1 = torch.nn.GRUCell(
                input_size=self.hidden_size, hidden_size=self.hidden_size, device=self.device)
            self.grucell_2 = torch.nn.GRUCell(
                input_size=self.hidden_size, hidden_size=self.num_outputs, device=self.device)
        elif self.extra_description == '_deps_sig_simple':  # not good
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs, hidden_size=self.num_outputs,
                                            device=self.device)
        elif self.extra_description == '_deps_sig_muLtiGRU':  # not good
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs + 3, hidden_size=self.num_outputs + 4,
                                            device=self.device)
            self.fc_list = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=self.num_outputs + 4,
                    num_outputs=self.num_outputs, num_layers_fc=4, num_node_fc=20))
        elif self.extra_description == '_deps_sig_p':  #
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs + 3, hidden_size=self.num_inputs + 4 + 1,
                                            device=self.device)
        elif self.extra_description == '_deps_sig_h0':  # good
            self.grucell = torch.nn.GRUCell(input_size=self.hidden_size, hidden_size=self.num_outputs)
            self.grucell_1 = torch.nn.GRUCell(input_size=self.num_inputs, hidden_size=self.hidden_size)
            self.fc1 = nn.Linear(self.num_outputs, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, num_outputs)
        elif self.extra_description == '_deps_sig_q':  # not good
            self.fc1 = nn.Linear(self.num_outputs + 1, self.hidden_size)
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs, hidden_size=self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.num_outputs)
        elif self.extra_description == '_deps_sig_q_h0':  # not good
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs, hidden_size=self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.extra_description == '_cal':
            self.K_origin = 0.8e6  # 1.25e6
            self.G_origin = 7e5  # 8.333e5
            self.q_yield_origin = 2e4  # 1e5
            self.K_norm = nn.Parameter(torch.ones(1)[0], requires_grad=True)  # 1.25e6
            self.G_norm = torch.tensor(torch.ones(1)[0], requires_grad=True)  # 8.333e5
            self.q_yield_norm = torch.tensor(torch.ones(1)[0], requires_grad=True)  # 1e5
            # self.K_norm = torch.tensor(1.0, dtype=torch.float, device=self.device, requires_grad=True)  # 1.25e6
            # self.G_norm = torch.tensor(1.0, dtype=torch.float, device=self.device, requires_grad=True)  # 8.333e5
            # self.q_yield_norm = torch.tensor(1.0, dtype=torch.float, device=self.device, requires_grad=True)  # 1e5
        elif self.extra_description == '_deps_sig_fc':
            if num_deep_layer is not None:
                num_layer = num_deep_layer
            else:
                num_layer = 10
            self.fc_list = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=self.num_inputs + self.num_outputs + 1,
                    num_outputs=self.num_outputs, num_layers_fc=num_layer, num_node_fc=30)).to(device=self.device)
        elif self.extra_description == '_deps_sig_epsp':
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs + 3, hidden_size=self.num_inputs + 4 + 1)
        elif self.extra_description == '_deps_sig_epsp_split':
            self.grucell = torch.nn.GRUCell(input_size=self.num_inputs + 3, hidden_size=self.num_inputs + 4 + 1)
            self.grucell_epsp = torch.nn.GRUCell(input_size=self.num_inputs + 3, hidden_size=self.num_inputs + 4 + 1)
        elif self.extra_description == '_deps_sig_epsp_split_classify':
            # self.grucell = torch.nn.GRUCell(input_size=self.num_inputs+3, hidden_size=self.num_inputs+4+1)
            # self.grucell_epsp = torch.nn.GRUCell(input_size=self.num_inputs+3, hidden_size=self.num_inputs+4+1)
            self.fc_list = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=self.num_inputs + self.num_outputs + 8,
                    num_outputs=3, num_layers_fc=4, num_node_fc=20))
            self.fc_list_1 = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=2,
                    num_outputs=1, num_layers_fc=5, num_node_fc=10))
        elif self.extra_description == '_deps_sig_epspvec_split_classify':
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            if num_deep_layer is not None:
                num_layer = num_deep_layer
            else:
                num_layer = 10
            self.fc_list = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=11,
                    num_outputs=8, num_layers_fc=num_layer, num_node_fc=30)).to(device=self.device)
            # self.fc_list_1 = torch.nn.ModuleList(
            #     self.get_initial_layers(
            #         num_inputs=10,
            #         num_outputs=3, num_layers_fc=5, num_node_fc=10))
            self.threshold = nn.Parameter(torch.ones(1), requires_grad=True).to(device=self.device)
        elif self.extra_description == '_deps_sig_epsp_fc':
            self.fc_list = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=self.num_inputs + self.num_outputs + 8,
                    num_outputs=self.num_outputs + 1, num_layers_fc=5, num_node_fc=20))
        elif self.extra_description == '_deps_sig_epsp_split_fc':
            self.fc_list = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=self.num_inputs + self.num_outputs + 8,
                    num_outputs=self.num_outputs, num_layers_fc=2, num_node_fc=20))
            self.fc_list_1 = torch.nn.ModuleList(
                self.get_initial_layers(
                    num_inputs=self.num_inputs + self.num_outputs + 8,
                    num_outputs=1, num_layers_fc=2, num_node_fc=20))
        elif self.extra_description == '_deps_sig_epsp_mc':
            from material_cell import MaterialCell
            self.mc = MaterialCell(
                num_sig=3, x_std=self.x_std,
                x_mean=self.x_mean, y_std=self.y_std, y_mean=self.y_mean,
                q_mean=self.q_mean, q_std=self.q_std)
        else:
            raise ValueError('The extra_description (%s) is not involved!' % self.extra_description)

    def forward(self, eps, h0):
        """

        :param eps:     in shape of (num_samples, num_steps, (00, 01, 11))
        :param h0:    in shape of (num_samples, 1, (00, 01, 11))  actually is sigma_0
        :return:
        """
        h0 = h0.to(self.device)
        deps_path = eps[:, 1:, :] - eps[:, :-1, :]
        # plt.plot(eps[1, :].detach().numpy()); plt.show()
        num_sample, num_steps, num_inputs = eps.shape
        sig = h0[:, 0]
        sig_list = [sig[:, :4]] if 'epspvec' not in self.extra_description else [sig]
        if self.extra_description == '_deps_sig_h0':
            h0_1 = torch.zeros(num_sample, self.hidden_size, device=self.device)
            for step in range(num_steps - 1):
                sig, h0_1 = self.get_sig_h0(deps=deps_path[:, step], h0=sig, h0_1=h0_1)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_extract':
            if self.hidden_size % 3 != 0:
                raise ValueError('Hidden size should be an integer multiple of 3!! But here is %d ' % self.hidden_size)
            else:
                num_to_each_component = self.hidden_size // 3
            self.rotate_index = list(range(2 * num_to_each_component, 3 * num_to_each_component)) + \
                                list(range(num_to_each_component, 2 * num_to_each_component)) + \
                                list(range(0, num_to_each_component))
            h0_1 = torch.zeros(num_sample, self.hidden_size, device=self.device)
            for step in range(num_steps - 1):
                # h0_1 = self.get_h0_extract(deps=deps_path[:, step], h0_1=h0_1)
                # sig_list.append(self.linear(h0_1)*self.y_std+self.y_mean)
                sig, h0_1 = self.get_sig_extract(deps=deps_path[:, step], h0_1=h0_1)
                sig_list.append(sig)

        elif self.extra_description == '_deps_datadriven_sig_epspvec':
            h0_0 = torch.zeros(num_sample, self.hidden_size)
            h0_1 = torch.zeros(num_sample, self.hidden_size)
            h0_2 = torch.zeros(num_sample, self.num_outputs)
            for step in range(num_steps - 1):
                sig, h0_1, h0_1, h0_2 = self.get_sig_datadiven_sig(deps=deps_path[:, step], h0_0=sig, h0_1=h0_1,
                                                                   h0_2=h0_2)
                sig_list.append(sig)

        elif self.extra_description == '_deps_sig':
            for step in range(num_steps - 1):
                sig = self.get_sig(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_simple':
            for step in range(num_steps - 1):
                sig = self.get_sig_simple(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_muLtiGRU':
            for step in range(num_steps - 1):
                sig = self.get_sig_multigru(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_p':
            for step in range(num_steps - 1):
                sig = self.get_sig_p(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_q':
            for step in range(num_steps - 1):
                sig = self.get_sig_q(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_q_h0':
            h0_1 = torch.zeros(num_sample, self.hidden_size - 1 - self.num_outputs)
            for step in range(num_steps - 1):
                sig, h0_1 = self.get_sig_q_h0(deps=deps_path[:, step], h0=sig, h0_1=h0_1)
                sig_list.append(sig)
        elif self.extra_description == '_cal':
            self.kroneker = torch.concat(  # 用于二维计算的 Voigt kronecker operator
                (torch.ones(num_sample, 1, device=self.device), torch.zeros(num_sample, 1, device=self.device),
                 torch.ones(num_sample, 1, device=self.device)), dim=1)
            for step in range(num_steps - 1):
                sig = self.get_sig_cal(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_fc':
            for step in range(num_steps - 1):
                sig = self.get_sig_fc(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_epsp':
            for step in range(num_steps - 1):
                sig = self.get_sig_epsp(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_epsp_split':
            for step in range(num_steps - 1):
                sig = self.get_sig_epsp_split(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_epsp_fc':
            for step in range(num_steps - 1):
                sig = self.get_sig_epsp_fc(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_epsp_split_fc':
            for step in range(num_steps - 1):
                sig = self.get_sig_epsp_split_fc(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_epsp_mc':
            for step in range(num_steps - 1):
                sig = self.get_sig_epsp_mc(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig[:, :4])
        elif self.extra_description == '_deps_sig_epsp_split_classify':
            for step in range(num_steps - 1):
                sig = self.get_sig_epsp_split_classify(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        elif self.extra_description == '_deps_sig_epspvec_split_classify':
            for step in range(num_steps - 1):
                sig = self.get_sig_epspvec_split_classify(deps=deps_path[:, step], h0=sig)
                sig_list.append(sig)
        else:
            raise ValueError('The extra_description (%s) is not involved!' % self.extra_description)
        sig_list = torch.stack(sig_list)  # in shape of (num_steps, num_samples, hidden_size)
        sig_list = torch.transpose(sig_list, 0, 1)  # in shape of (num_samples, num_steps, hidden_size)      q
        # for i in range(0, 10):
        #     plt.plot(sig_list[i, :].detach().numpy())
        #     plt.show()
        return sig_list

    def get_sig_h0(self, deps, h0, h0_1):
        h0_1 = self.grucell_1(deps, h0_1)
        h0 = self.activation(self.fc1(h0))
        h0 = self.grucell(h0_1, h0)  # hidden_size -> hidden_size
        h0 = self.fc2(h0)  # hidden_size -> 3
        return h0, h0_1

    def get_h0_extract(self, deps, h0_1):
        deps = deps / self.x_std[:3]
        h0_2 = self.grucell_1(deps, h0_1)
        h0_3 = self.grucell_2(deps, h0_2)
        h0_4 = self.grucell_3(deps, h0_3)
        # h0_5 = self.grucell_4(deps, h0_4)
        return h0_4

    def get_sig_datadiven_sig(self, deps, h0_0, h0_1, h0_2):
        sig, h0_0, h0_1, h0_2 = self.get_sig_datadiven_sig_inner(deps=deps, h0_0=h0_0, h0_1=h0_1, h0_2=h0_2)
        sig_transformed, h0_0_transformed, h0_1_transformed, h0_2_transformed = \
            self.get_sig_datadiven_sig_inner(deps=deps[:, [2, 1, 0]], h0_0=h0_0, h0_1=h0_1, h0_2=h0_2)
        return sig, h0_0, h0_1, h0_2

    def get_sig_datadiven_sig_inner(self, deps, h0_0, h0_1, h0_2):
        ## TODO complete the data-driven mode. Problem: how to exchange the axis of 00, 11???

        sig = 0
        return sig, h0_0, h0_1, h0_2

    def get_sig_simple(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        sig = (self.get_sig_simple_inner(deps=deps, h0=h0) +
               self.get_sig_simple_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0]])[:, [2, 1, 0]]) * 0.5

        return sig

    def get_sig_simple_inner(self, deps, h0):
        deps = deps / self.x_std[:3]
        h0 = (h0 - self.y_mean[: 3]) / self.y_std[:3]
        h1 = self.grucell(deps, h0)
        sig = h1 * self.y_std[:3] + self.y_mean[:3]
        return sig

    def get_sig(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        # deps_temp, h0_temp = deps[:, [2, 1, 0]], h0[:, [2, 1, 0]]
        # sig_temp = ((self.get_sig_inner(deps=deps_temp, h0=h0_temp) +
        #             self.get_sig_inner(
        #                 deps=deps_temp[:, [2, 1, 0]], h0=h0_temp[:, [2, 1, 0]])[:, [2, 1, 0]]) * 0.5)[:, [2, 1, 0]]

        # sig = self.get_sig_inner(deps=deps, h0=h0)

        sig = (self.get_sig_inner(deps=deps, h0=h0) +
               self.get_sig_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0]])[:, [2, 1, 0]]) * 0.5

        return sig

    def get_sig_inner(self, deps, h0):
        """

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        deps = deps.to(self.device)
        h0 = h0.to(self.device)
        # get the stress angle and the strain angle
        with torch.no_grad():
            # cal the angl
            # e according to stress rotation
            angle_sig = torch.atan_(
                torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
            angle_deps = torch.atan_(
                torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
            sig_principl = self.get_principal(sig=h0, angle=angle_sig)
            deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # deps_temp = deps[0].detach().numpy()
        # deps_temp = np.array([[deps_temp[0], deps_temp[1]], [deps_temp[1], deps_temp[2]]])
        # angle_deps_temp = angle_deps[0, 0].detach().numpy()
        # Q = np.array([[np.cos(angle_deps_temp), np.sin(angle_deps_temp)],
        #               [-np.sin(angle_deps_temp), np.cos(angle_deps_temp)]])
        # deps_origin = Q@deps_temp@Q.T

        # standard normalization for the deps
        deps = deps / self.x_std
        deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        q = (get_q_2d(h0) - self.q_mean) / self.q_std
        sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        # # minmax normalization
        # q = (get_q_2d(h0)-self.q_min)/((self.q_max-self.q_min)*1.2)
        # h0 = (h0-self.y_min)/((self.y_max-self.y_min)*1.2)

        h1 = torch.concat((h0, q, angle_sig, sig_principl), dim=1)  # 3+1+1+2 = 7
        deps = torch.concat((deps, angle_deps, deps_principl), dim=1)  # 3+1+2   = 6
        h1 = self.grucell(deps, h1)
        # sig = self.fc_2(self.activation(self.fc_1(sig)))
        # sig_1 = self.fc_2(self.fc_1(h0))
        # sig_2 = self.fc_3(self.fc_1_2(h0*h0))
        # sig = sig_1+sig_2

        sig = h1[:, :3] * self.y_std + self.y_mean
        # sig = h1[:, :-2]*((self.y_max-self.y_min)*1.2)+self.y_min
        return sig

    def get_sig_p(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        sig = (self.get_sig_p_inner(deps=deps, h0=h0) +
               self.get_sig_p_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0]])[:, [2, 1, 0]]) * 0.5
        return sig

    def get_sig_p_inner(self, deps, h0):
        """

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        deps = deps.to(self.device)
        h0 = h0.to(self.device)
        # before normalization get the stress angle and the strain angle
        with torch.no_grad():
            # calculate the angle according to stress rotation
            angle_sig = torch.atan_(
                torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
            angle_deps = torch.atan_(
                torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
            sig_principl = self.get_principal(sig=h0, angle=angle_sig)
            deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # standard normalization for the deps
        deps = deps / self.x_std
        deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        p = ((h0[:, 0:1] + h0[:, 2:3]) / 2. - (self.y_mean[0] + self.y_mean[2]) / 2.) / (
                    (self.y_std[0] + self.y_std[2]) / 2.)
        q = (get_q_2d(h0) - self.q_mean) / self.q_std
        sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        h1 = torch.concat((h0, p, q, angle_sig, sig_principl), dim=1)  # 3+1+1+1+2 = 8
        deps = torch.concat((deps, angle_deps, deps_principl), dim=1)  # 3+1+2     = 6
        h1 = self.grucell(deps, h1)

        sig = h1[:, :3] * self.y_std + self.y_mean
        return sig

    def get_sig_multigru(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        sig = (self.get_sig_multigru_inner(deps=deps, h0=h0) +
               self.get_sig_multigru_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0]])[:, [2, 1, 0]]) * 0.5
        return sig

    def get_sig_multigru_inner(self, deps, h0):
        """

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        # before normalization get the stress angle and the strain angle
        with torch.no_grad():
            # calculate the angle according to stress rotation
            angle_sig = torch.atan_(
                torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
            angle_deps = torch.atan_(
                torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
            sig_principl = self.get_principal(sig=h0, angle=angle_sig)
            deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # standard normalization for the deps
        deps = deps / self.x_std
        deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        # p = ((h0[:, 0:1] + h0[:, 2:3])/2.-(self.y_mean[0]+self.y_mean[2])/2.)/((self.y_std[0] + self.y_std[2])/2.)
        q = (get_q_2d(h0) - self.q_mean) / self.q_std
        sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        h0_extended = torch.concat((h0, q, angle_sig, sig_principl), dim=1)  # 3+1+1+2 = 7
        deps_extended = torch.concat((deps, angle_deps, deps_principl), dim=1)  # 3+1+2     = 6

        h_0 = self.grucell(deps_extended, h0_extended)
        sig_normed = self.forward_fc(h_0)

        sig = sig_normed * self.y_std + self.y_mean
        return sig

    def get_sig_cal(self, deps, h0):
        """

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """
        K, G, q_yield = self.K_norm * self.K_origin, \
                        self.G_norm * self.G_origin, \
                        self.q_yield_norm * self.q_yield_origin

        num_sample = len(deps)
        sig = h0
        p = (sig[:, 0:1] + sig[:, 2:3]) / 2.
        s = sig - self.kroneker * p
        deps_v = deps[:, 0:1] + deps[:, 2:3]
        de = deps - self.kroneker * deps_v / 2.
        p_trial = p + K * deps_v
        s_trial = s + 2. * G * de
        q_trial = get_q_2d(s_trial)
        sig = p_trial * self.kroneker + \
              s_trial * torch.min(q_yield / q_trial,
                                  torch.ones(num_sample, 1, device=self.device))

        return sig

    def get_sig_fc(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """

        sig = (self.get_sig_fc_inner(deps=deps, h0=h0) +
               self.get_sig_fc_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0]])[:, [2, 1, 0]]) * 0.5
        return sig

    def get_sig_fc_inner(self, deps, h0):
        # with torch.no_grad():
        #     # cal the angl
        #     # e according to stress rotation
        #     angle_sig = torch.atan_(
        #         torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
        #     angle_deps = torch.atan_(
        #         torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
        #     sig_principl = self.get_principal(sig=h0, angle=angle_sig)
        #     deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # standard normalization for the deps
        deps = deps / self.x_std
        # deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        q = (get_q_2d(h0[:, :3]) - self.q_mean) / self.q_std
        q_norm = (q - self.q_mean) / self.q_std
        # sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        x = torch.concat((deps, h0, q_norm), dim=1)  # 3+1+2+3+1+1+2= 13

        sig = self.forward_fc(x=x)

        sig = sig * self.y_std + self.y_mean

        return sig

    def get_sig_epsp(self, deps, h0):
        """
                             rotate (xx,xy,yy) -> (yy,xy,xx)

                :param deps:           (num_samples, (00, 01, 11))
                :param h0:   \sigma_0  (num_samples, (00, 01, 11, epsp))
                :return:
                """
        sig = (self.get_sig_epsp_inner(deps=deps, h0=h0) +
               self.get_sig_epsp_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0, 3]])[:, [2, 1, 0, 3]]) * 0.5
        return sig

    def get_sig_epsp_inner(self, deps, h0):
        """

                :param deps:           (num_samples, (00, 01, 11))
                :param h0:   \sigma_0  (num_samples, (00, 01, 11))
                :return:
                """
        # get the stress angle and the strain angle
        with torch.no_grad():
            # cal the angl
            # e according to stress rotation
            angle_sig = torch.atan_(
                torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
            angle_deps = torch.atan_(
                torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
            sig_principl = self.get_principal(sig=h0[:, :3], angle=angle_sig)
            deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # standard normalization for the deps
        deps = deps / self.x_std
        deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        q = (get_q_2d(h0[:, :3]) - self.q_mean) / self.q_std
        sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        h1 = torch.concat((h0, q, angle_sig, sig_principl), dim=1)  # 4+1+1+2 = 8
        deps = torch.concat((deps, angle_deps, deps_principl), dim=1)  # 4+1+2   = 7
        h1 = self.grucell(deps, h1)

        sig = h1[:, :4] * self.y_std + self.y_mean

        return sig

    def get_sig_epsp_fc(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """

        sig = (self.get_sig_epsp_fc_inner(deps=deps, h0=h0) +
               self.get_sig_epsp_fc_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0, 3]])[:, [2, 1, 0, 3]]) * 0.5
        return sig

    def get_sig_epsp_fc_inner(self, deps, h0):
        with torch.no_grad():
            # cal the angl
            # e according to stress rotation
            angle_sig = torch.atan_(
                torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
            angle_deps = torch.atan_(
                torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
            sig_principl = self.get_principal(sig=h0[:, :3], angle=angle_sig)
            deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # standard normalization for the deps
        deps = deps / self.x_std
        deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        q = (get_q_2d(h0[:, :3]) - self.q_mean) / self.q_std
        sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        x = torch.concat((deps, angle_deps, deps_principl, h0, q, angle_sig, sig_principl), dim=1)  # 3+1+2+3+1+1+2= 13

        sig = self.forward_fc(x=x)

        sig = sig * self.y_std + self.y_mean
        return sig

    def get_sig_epsp_split(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """

        sig = (self.get_sig_epsp_split_inner(deps=deps, h0=h0) +
               self.get_sig_epsp_split_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0, 3]])[:, [2, 1, 0, 3]]) * 0.5
        return sig

    def get_sig_epsp_split_inner(self, deps, h0):
        """

                :param deps:           (num_samples, (00, 01, 11))
                :param h0:   \sigma_0  (num_samples, (00, 01, 11))
                :return:
                """
        deps_concat, h0_concat = self.deps_h0_preprocessing(deps=deps, h0=h0)

        # sig_prediction
        h1 = self.grucell(deps_concat, h0_concat)
        # epsp prediction
        epsp_norm_pre = self.grucell_epsp(deps_concat, h1)[:, 3:4]

        # reverse to the original magnitude
        sig = torch.concat((h1[:, :3], epsp_norm_pre), dim=1) * self.y_std + self.y_mean

        return sig

    def get_sig_epsp_split_classify(self, deps, h0):
        """
                     rotate (xx,xy,yy) -> (yy,xy,xx)

        :param deps:           (num_samples, (00, 01, 11))
        :param h0:   \sigma_0  (num_samples, (00, 01, 11))
        :return:
        """

        sig = (self.get_sig_epsp_split_classify_inner(deps=deps, h0=h0) +
               self.get_sig_epsp_split_classify_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0, 3]])[:,
               [2, 1, 0, 3]]) * 0.5
        return sig

    def get_sig_epsp_split_classify_inner(self, deps, h0):
        """

                :param deps:           (num_samples, (00, 01, 11))
                :param h0:   \sigma_0  (num_samples, (00, 01, 11, epsp))
                :return:
                """
        deps_concat, h0_concat = self.deps_h0_preprocessing(deps=deps, h0=h0)

        # sig_prediction
        temp = torch.concat((deps_concat, h0_concat), dim=1)
        h1 = self.forward_fc(temp)
        # h1 = self.grucell(deps_concat, h0_concat)
        # epsp prediction
        sig_temp = h1[:, :3] * self.y_std[:3] + self.y_mean[:3]
        temp = self.forward_fc_1(torch.concat((get_q_2d(sig_temp) / self.q_std, h0_concat[:, 3:4]), dim=1))
        epsp_norm_pre = temp

        # reverse to the original magnitude
        sig = torch.concat((h1[:, :3], epsp_norm_pre), dim=1) * self.y_std + self.y_mean

        return sig

    def get_sig_epspvec_split_classify(self, deps, h0):
        """

        :param deps:   (num_samples, [00, 01, 11])    3
        :param h0:     (numsaples, [sig_00, sig_01, sig_11, epsp, epspvec_00, epspvec_01, epspvec_11])  7
        :return:                      0       1        2      3        4          5           6
        """

        sig = (self.get_sig_epspvec_split_classify_inner(deps=deps, h0=h0) +
               self.get_sig_epspvec_split_classify_inner(
                   deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0, 3, 6, 5, 4]])[:, [2, 1, 0, 3, 6, 5, 4]]) * 0.5
        return sig

    def get_sig_epspvec_split_classify_inner(self, deps, h0):
        deps = deps / self.x_std
        q = get_q_2d(h0[:, :3])
        q_norm = (q - self.q_mean) / self.q_std
        epsp = torch.sqrt(h0[:, 4] ** 2 + 2. * h0[:, 5] ** 2 + h0[:, 6] ** 2).reshape(-1, 1)
        h0_new = torch.concat((h0[:, :3], epsp, h0[:, 4:]), dim=1)
        h0_normed = (h0_new - self.y_mean) / self.y_std

        # sig_prediction
        temp = torch.concat((deps, h0_normed, q_norm), dim=1)  # 3+7+1 = 11 -> 8
        y_norm = self.forward_fc(temp)

        y = y_norm[:, :7] * self.y_std + self.y_mean

        epsp_vec = self.sigmoid(y_norm[:, 7:8]) * (y[:, 4:] - h0[:, 4:]) + h0[:, 4:]

        y_pre = torch.concat(
            (y[:, :3],
             torch.sqrt(epsp_vec[:, 0:1] ** 2 + 2. * epsp_vec[:, 1:2] ** 2 + epsp_vec[:, 2:3] ** 2), epsp_vec), dim=1)

        # y[:, 3] = torch.sqrt(y[:, 4]**2 + 2. * y[:, 5]**2 + y[:, 6]**2)

        # epspvec = self.forward_fc_1(temp) * self.y_std[4:] + self.y_mean[4:]
        # epsp = torch.sqrt(epspvec[:, 0:1]**2 + 2. * epspvec[:, 1:2]**2 + epspvec[:, 2:3]**2)

        # pre_value = torch.concat((sig, epsp, epspvec), dim=1)
        return y_pre

    def get_sig_epsp_split_fc(self, deps, h0):

        sig = (self.get_sig_epsp_split_fc_inner(deps=deps, h0=h0) +
               self.get_sig_epsp_split_fc_inner(deps=deps[:, [2, 1, 0]], h0=h0[:, [2, 1, 0, 3]])[:, [2, 1, 0, 3]]) * 0.5
        return sig

    def get_sig_epsp_split_fc_inner(self, deps, h0):
        # standard normalization
        deps_concat, h0_concat = self.deps_h0_preprocessing(deps=deps, h0=h0)

        x = torch.concat((deps_concat, h0_concat), dim=1)  # 3+1+2+3+1+1+2= 13

        sig = torch.concat((self.forward_fc(x=x), self.forward_fc_1(x=x)), dim=1)

        sig = sig * self.y_std + self.y_mean
        return sig

    def get_sig_epsp_mc(self, deps, h0):
        """

        :param deps:    (num_samples, (00, 10, 11))
        :param h0:      (num_samples, (00, 10, 11, epsp))
        :return:
        """
        # deps = deps/self.x_std
        # h0 = (h0-self.y_mean)/self.y_std
        sig, epsp = self.mc.forward(deps=deps, sig_0=h0[:, :3], epsp_vec=h0[:, 4:])
        # sig_H = torch.concat((sig, H), dim=1)*self.y_std+self.y_mean
        # return sig_H
        return torch.concat((sig, epsp), dim=1)

    def get_principal(self, sig, angle):
        """

        :param sig:     (num_samples, [00 01 11])
        :param angle:   (num_samples, [1])
        :return:
        """
        coss, sinn = torch.cos(angle), torch.sin(angle)
        coss2, sinn2, consin = coss * coss, sinn * sinn, coss * sinn
        sig_1 = sig[:, 0:1] * coss2 + sig[:, 2:3] * sinn2 + sig[:, 1:2] * consin * 2.
        sig_2 = sig[:, 0:1] * sinn2 + sig[:, 2:3] * coss2 - sig[:, 1:2] * consin * 2.
        # sig_12 = (coss2-sinn2)*sig[:, 1:2] + consin*(sig[:, 2:3] - sig[:, 0:1])
        return torch.concat((sig_1, sig_2), dim=1)

    def deps_h0_preprocessing(self, deps, h0):
        """
            deps_concated: (num_samples, (00, 01, 11, q, angle_deps, principal_1, principal_2))   3+1+1+2=7
            h0_concated:   (num_samples, (00, 01, 11, angle_deps, principal_1, principal_2))      3+1+2=6

                :param deps:           (num_samples, (00, 01, 11))
                :param h0:   \sigma_0  (num_samples, (00, 01, 11))
                :return:
                """
        # get the stress angle and the strain angle
        with torch.no_grad():
            # cal the angl
            # e according to stress rotation
            angle_sig = torch.atan_(
                torch.nan_to_num(h0[:, 1:2] * 2. / (h0[:, 0:1] - h0[:, 2:3]), nan=0.)) * 0.5
            angle_deps = torch.atan_(
                torch.nan_to_num(deps[:, 1:2] * 2. / (deps[:, 0:1] - deps[:, 2:3]), nan=0.)) * 0.5
            sig_principl = self.get_principal(sig=h0[:, :3], angle=angle_sig)
            deps_principl = self.get_principal(sig=deps, angle=angle_deps)

        # standard normalization for the deps
        deps = deps / self.x_std
        deps_principl = deps_principl / self.x_std[[0, 2]]

        # standard normalization
        q = (get_q_2d(h0[:, :3]) - self.q_mean) / self.q_std
        sig_principl = (sig_principl - self.q_mean[[0, 0]]) / self.q_std[[0, 0]]
        h0 = (h0 - self.y_mean) / self.y_std

        # concatenate  the hidden and the input
        h0_concat = torch.concat((h0, q, angle_sig, sig_principl), dim=1)  # 4+1+1+2 = 8
        deps_concat = torch.concat((deps, angle_deps, deps_principl), dim=1)  # 4+1+2   = 7
        return deps_concat, h0_concat

    def get_1_0_neg1(self, x, threshold=0.5):
        """

        :param x:  (num_samples, 1)
        :return:
        """
        threshold_use = self.threshold * threshold
        ones = torch.ones_like(x) * self.threshold
        zeros = torch.zeros_like(x)
        judge_neg1 = x <= -threshold_use
        judge_0 = (x > -threshold_use) & (x < threshold_use)
        judge_1 = x >= threshold_use
        x = -ones * judge_neg1 + zeros * judge_0 + ones * judge_1
        return x

    def get_sig_extract(self, deps, h0_1):
        deps_norm = torch.sqrt(deps[:, 0] ** 2 + 2. * deps[:, 1] ** 2 + deps[:, 2] ** 2)

        d_adapt = self.train_x_norm_median / deps_norm
        d_adapt = torch.nan_to_num(d_adapt, nan=1., posinf=1., neginf=1.)
        d_adapt = torch.where(d_adapt < 1.0, 1.0, d_adapt)

        h0_1_no_rotated = self.get_sig_extract_inner(deps=deps, h0_1=h0_1, d_adapt=d_adapt)
        h0_1_rotated = self.get_sig_extract_inner(
            deps=deps[:, [2, 1, 0]],
            h0_1=h0_1[:, self.rotate_index], d_adapt=d_adapt)
        h0_1 = (h0_1_no_rotated + h0_1_rotated[:, self.rotate_index]) * 0.5

        sig = self.linear(h0_1) * self.y_std + self.y_mean
        return sig, h0_1

    def get_sig_extract_inner(self, deps, h0_1, d_adapt):

        """

        :param deps: (num_samples, 3)
        :param h0_1: (num_samples, hidden_size)
        :return:
        """

        # d_adapt = torch.ones(size = [len(deps)], device=torch.device('cuda'))

        d_h0_1 = self.get_h0_extract(deps=torch.einsum('ij, i->ij', deps, d_adapt), h0_1=h0_1) - h0_1
        h0_1 = torch.einsum('ij, i->ij', d_h0_1, 1. / d_adapt) + h0_1
        return h0_1
