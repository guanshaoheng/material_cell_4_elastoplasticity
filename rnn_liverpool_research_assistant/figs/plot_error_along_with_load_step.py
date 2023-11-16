import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from FEMxML.rnn_liverpool_research_assistant.utils_rnn_cons import get_q_2d_arr

from sciptes4figures.utils_plot import (
    plot_training_loss,
    get_color_list,
    configurations,
)
from utilSelf.general import echo
from rnn_liverpool_research_assistant.rnn_model_trainer_j2 import (
    call_and_prediction,
    data_reading,
    restore_model,
)


font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 200
# fix random seeds
axes = {"labelsize": "large"}
font = {"family": "serif", "weight": "normal", "size": 17}
legend = {"fontsize": "large"}
lines = {"linewidth": 3, "markersize": 7}
mpl.rc("font", **font)
mpl.rc("axes", **axes)
mpl.rc("legend", **legend)
mpl.rc("lines", **lines)

colorlist = [
    "#008080",
    "#FF7F50",
    "#4169E1",
    "#DA70D6",
    "#808000",
    "#4B0082",
    "#FF8C00",
    "#FF1493",
    "#4682B4",
    "#DAA520",
    "#9370DB",
    "#2E8B57",
    "#483D8B",
    "#FF6347",
    "#008B8B",
    "#BA55D3",
    "#B8860B",
    "#1E90FF",
    "#3CB371",
]

# 创建颜色循环
colors = plt.cm.viridis(np.linspace(0, 1, 10))

sav_fig_path = "/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/figs"


# ====================================
# j2 physics extensions 模型预测
mode = "j2"
nn_architecture = "sig"  # 'classic' , 'sig'
len_sequence = 20
num_samples = 201
extra_description = "_deps_sig"
lc = 1.0  # 1.0
model_saved_dir = os.path.join(
    "/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant",
    "rnn_j2_numSamples201_NNsig_deps_sig_lc1.0_len20")

# =====================================
# computation configurations
device = torch.device('cpu')


# 读取数据
"""
x: strain in shape of (num_samples, num_loading_step, 3 components) 
y: stress in shape of (num_samples, num_loading_step, 3 components) 
"""
x, y, eps_p_norm, eps_p_vec = data_reading(
    num_samples=999, mode=mode, voigt_flag=True
)  # make sure use the samples not involved in the training

# 训练后的模型
model = restore_model(model_full_path=os.path.join(model_saved_dir, 'rnn.pt'), device=device)
model.x_std = model.x_std.to(device)
model.y_std = model.y_std.to(device)
model.x_mean = model.x_mean.to(device)
model.y_mean = model.y_mean.to(device)


# 预测数据
test_len = 1000
y_pre = model.forward(
    torch.from_numpy(x[:test_len]).float().to(device), 
    h0=torch.from_numpy(y[:test_len, :1, :]).float().to(device)
).cpu().detach().numpy()

q = get_q_2d_arr(y[:test_len])
error = np.average(np.abs(get_q_2d_arr(y_pre) - q), axis=-1)

# 计算均值和标准差
mean_curve = np.mean(error, axis=0)
std_dev_curve = np.std(error, axis=0)

# 绘制均值曲线
plot_start = 25
plt.plot(range(plot_start, len(y[0])), (mean_curve/(np.mean(q, axis=0).squeeze()+1000)*100)[plot_start:], 
         label='Mean', color='blue', linestyle='-', linewidth=2)

# 绘制带状区域表示分布
plt.fill_between(
        range(plot_start, len(y[0])),
        ((mean_curve - std_dev_curve)/(np.mean(q, axis=0).squeeze()+1000)*100)[plot_start:], 
        ((mean_curve + std_dev_curve)/(np.mean(q, axis=0).squeeze()+1000)*100)[plot_start:], 
        alpha=0.2, label='Variance', color='lightgray')

plt.grid(True, linestyle='--', alpha=0.5)

# 添加标题和标签
plt.xlabel("Loading step")
plt.ylabel("Predicted VM error (%)")

plt.xlim(0, 210)

# 添加图例
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(sav_fig_path, "prediction_error_along_with_time.png"), dpi=200)
plt.close()



if __name__ == "__main__":
    """
    这个文件用来展示优化基于J2的弹塑性模型
    在训练后模型的预测误差随着时间步的增多逐渐增大的过程
    """
    print('Finished!')