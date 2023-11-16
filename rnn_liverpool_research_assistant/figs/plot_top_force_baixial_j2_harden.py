import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sciptes4figures.utils_plot import readTopForce_biaxial, configurations, get_color_list

# -------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()


mpl.rcParams['figure.dpi'] = 200
# fix random seeds
axes = {'labelsize': 'large'}
font = {'family': 'serif',
        'weight': 'normal',
        'size': 20}
legend = {'fontsize': 20}
lines = {'linewidth': 3,
         'markersize': 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)


print(os.getcwd())
pathList = [
    # Drucker-prager
    "../../../../simu/explicit/biaxial/biaxial_rough_explicit_misesideal_intorder1_numg484_biaxial_0.05_548_vel0"
    ".10_damp1.0e+06_safe0.5_timestep2.0e-04",
    "../../../../simu/explicit/biaxial/biaxial_rough_explicit_rnn_misesideal_intorder1_numg484_biaxial_0.05_548_vel0"
    ".10_damp1.0e+06_safe0.5_timestep2.0e-04_scalar1",
    '../../../../simu/explicit/biaxial/biaxial_rough_explicit_rnn_misesideal_intorder1_numg484_biaxial_0.05_548_vel0'
    '.10_damp1.0e+06_safe0.5_timestep2.0e-04_scalar50',
    "../../../../simu/explicit/biaxial/biaxial_rough_explicit_rnn_misesideal_intorder1_numg484_biaxial_0.05_548_vel0"
    ".10_damp1.0e+06_safe0.5_timestep2.0e-04_scalar60",
    "../../../../simu/explicit/biaxial/biaxial_rough_explicit_rnn_misesideal_intorder1_numg484_biaxial_0.05_548_vel0"
    ".10_damp1.0e+06_safe0.5_timestep2.0e-04_scalarAdap",
]

plt.style.use('seaborn-paper')

fig = plt.figure(figsize=[10, 10*2/3])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# ax3 = fig.add_subplot(313)
# ax4 = fig.add_subplot(414)
i = 0
label_list = [
    r'$J_2$',
    r'MC $s=1$',
    r'MC $s=50$',
    r'MC $s=60$',
    r"MC adapt."
]

datas, _ = readTopForce_biaxial(path=pathList[0], split_keyword='safe0.4')
n = len(datas[:, 0])
plot_index = np.arange(0, n, n//20) if n > 35 else np.arange(0, n)
ax1.plot(-datas[:, 0][plot_index], -datas[:, 1][plot_index] / 1e3, label=label_list[0], c=color_list[i],
         marker='P', linewidth=3, markersize=12)
ax2.plot(-datas[:, 0][plot_index], datas[:, 3][plot_index], c=color_list[0], marker='P',
         markersize=12, linewidth=3, label=label_list[0])

datas, _ = readTopForce_biaxial(path=pathList[1], split_keyword='safe0.4')
ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label_list[1], c=color_list[1], linewidth=3)
ax2.plot(-datas[:, 0],  datas[:, 3], c=color_list[1], linewidth=3, label=label_list[1])

# datas, _ = readTopForce_biaxial(path=pathList[2], split_keyword='safe0.4')
# ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label_list[2], c=color_list[2], linewidth=5, linestyle='--', zorder=10)
# ax2.plot(-datas[:, 0],  datas[:, 3], c=color_list[2], linewidth=5, linestyle='--', zorder=10, label=label_list[2])

datas, _ = readTopForce_biaxial(path=pathList[3], split_keyword='safe0.4')
ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label_list[3], c=color_list[5], linewidth=5, zorder=10)
ax2.plot(-datas[:, 0],  datas[:, 3], c=color_list[5], linewidth=5, zorder=10, label=label_list[3])

datas, _ = readTopForce_biaxial(path=pathList[4], split_keyword='safe0.4')
ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label_list[4], c=color_list[8], linewidth=3, linestyle='--', zorder=11)
ax2.plot(-datas[:, 0],  datas[:, 3], c=color_list[8], linewidth=3, linestyle='--', zorder=10, label=label_list[4])


for i, aax in enumerate([ax1, ax2]):
    aax.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=20)
    aax.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=20)
    if i != 1:
        aax.xaxis.set_ticklabels([])

legendDic['prop'] = font_3
ax2.legend(loc='upper right', **legendDic)
# ax1.legend()
ax1.set_ylim([-3, 70])
ax2.set_ylim([-0.05, 0.05])
ax1.set_ylabel(r'Top force (kN)', fontdict=font_1)
ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_1)
ax2.set_xlabel(r'Axial strain', fontdict=font_1)
plt.tight_layout()
# plt.subplots_adjust(left=0.14)
plt.savefig('/home/tongming/fem-ml-dem/FEMxML/rnn_liverpool_research_assistant/figs/topforce_j2.png', dpi=200)
plt.show()
