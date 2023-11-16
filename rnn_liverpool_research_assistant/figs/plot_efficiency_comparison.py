import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



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


def get_time(file_path: str) -> float:
    with open(os.path.join(file_path, "biaxial_surf.dat"), 'r') as f:
        lines = f.readlines()
    time_used = float(lines[-1].split(" ")[-1].replace("\n", ""))*60 # in mins
    return time_used


def plot_time_used(time_used_dic: dict) -> None:
    fig = plt.figure()

    width = 1

    index = 1
    for i in time_used_dic:
        plt.bar(index, time_used_dic[i][0], width,  color="#1F77B4")
        plt.bar(index + width,  time_used_dic[i][1], width, color="#FF7F0E")
        index += width*2.5

    plt.legend(["Con.", "MC"], loc="center right")

    plt.ylabel("Consumed time (mins)")
    plt.xticks([1.5, 4, 6.5], ['$J_2$', "Drucker", "$J_2$-harden"])

    plt.tight_layout()
    plt.savefig('./computational efficiency.png', dpi=300)
    plt.show()

    return



def main():
    outer_path = "../../../../simu/explicit/biaxial"
    dir_paths = {
        "j2": [
            "biaxial_rough_explicit_mises_harden_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_timestep2.0e-04",
            "biaxial_rough_explicit_rnn_mises_harden_extract_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_timestep2.0e-04_scalarAdap",
        ],
        "drucker": [
            "biaxial_rough_explicit_drucker_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_timestep2.0e-04",
            "biaxial_rough_explicit_rnn_drucker_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_timestep2.0e-04_scalarAdap",
        ],

        "j2-harden": [
            "biaxial_rough_explicit_misesideal_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_timestep2.0e-04",
            "biaxial_rough_explicit_rnn_misesideal_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_timestep2.0e-04_scalarAdap",
        ]
    }

    time_used = {}
    for i in dir_paths:
        model_time = get_time(os.path.join(outer_path, dir_paths[i][0]))
        nn_time = get_time(os.path.join(outer_path, dir_paths[i][1]))
        time_used[i] = [model_time, nn_time]


    plot_time_used(time_used)





if __name__ == "__main__":
    main()