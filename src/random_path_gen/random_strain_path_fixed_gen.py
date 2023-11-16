import os
import matplotlib.pyplot as plt
import numpy as np
from gaussian_utils import kernel_gaussian
from utilSelf.general import check_mkdir, echo

'''
    we should generate the principal strain and the Load angle and then transform them to strain tensor, 
    instead of directly generating the 3 components of the strain tensor.
'''


def path_generator(x, x_, num_points,
                   relative_length_range, s_range, s_theta_range, initial_theta_range, index=0, ):
    # np.random.seed(index)
    relative_length = np.random.uniform(*relative_length_range)
    s_strain = np.random.uniform(*s_range)
    s_theta = np.random.uniform(*s_theta_range)
    initial_theta = np.random.uniform(*initial_theta_range)

    # strain
    k = kernel_gaussian(x, x, l=relative_length, s=s_strain)
    k_ = kernel_gaussian(x, x_, l=relative_length, s=s_strain)
    k__ = kernel_gaussian(x_, x_, l=relative_length, s=s_strain)
    L = np.linalg.cholesky(k)
    v = np.linalg.solve(L, k_)
    fixed_covariance_kernel = k__ - v.T @ v

    strains_principal = np.random.multivariate_normal(
        mean=np.zeros(shape=len(x_)), cov=fixed_covariance_kernel, size=2)

    # theta
    theta = np.random.multivariate_normal(
        mean=np.ones(len(x_)) * initial_theta, cov=fixed_covariance_kernel * (s_theta / s_strain) ** 2)

    # for i in range(3):
    #     test = fixed_covariance_kernel @ np.random.normal(size=200)
    #     plt.plot(test)
    # plt.show()

    plt.imshow(fixed_covariance_kernel* (s_theta / s_strain) ** 2); plt.tight_layout(); plt.colorbar(); plt.show()



    strain = np.zeros(shape=[num_points, 3])  # in format of [00, 01, 11]

    for i in range(num_points):
        cos_ = np.cos(theta[i])
        cos2 = cos_ ** 2
        sin2 = 1 - cos2
        strain[i, 0] = strains_principal[0, i] + (strains_principal[1, i] - strains_principal[0, i]) * sin2  # 00
        strain[i, 1] = np.cos(theta[i]) * np.sin(theta[i]) * (strains_principal[1, i] - strains_principal[0, i])  # 01
        strain[i, 2] = strains_principal[1, i] + (strains_principal[0, i] - strains_principal[1, i]) * sin2  # 11

    # global counter
    # counter += 1
    return strain


#
# # initialize worker processes
# def init_worker(shared_counter):
#     # declare scope of a new global variable
#     global counter
#     # store argument in the global variable for this process
#     counter = shared_counter


## main
def main(
        num_points=200,
        total_sample_num=999,
        only_plot=True):
    total_sample_num = total_sample_num if not only_plot else 10
    l = 10
    relative_length_range = [0.1 * l, 0.5 * l]
    max_strain_range = [0.1, 0.18]
    max_theta_range = [0, np.pi / 4.]
    initial_theta_range = [-np.pi, np.pi]
    s_range = [max_strain_range[i] / 2 for i in range(2)]
    s_theta_range = [max_theta_range[i] / 2 for i in range(2)]
    # constrain the location and the 1st order derivation
    # x = np.array([-0.01, 0]).reshape([-1, 1])
    # y = np.array([0., 0.]).reshape([-1, 1])
    x = np.array([0]).reshape([-1, 1])
    y = np.array([0.]).reshape([-1, 1])
    x_ = np.linspace(0, l, num_points).reshape(-1, 1)

    paths = []
    size_num = 3
    samples_num = 0
    num_p, num_per = 4, 10
    while samples_num < total_sample_num:
        # with multiprocessing.Pool(num_p, initializer=init_worker, initargs=(0, )) as pool:
        # NOTE: THE RANDOM SEED IS THE SAME IN DIFFERENT PROCESSES!!!
        # with multiprocessing.Pool(num_p) as pool:
        #     paths_temp = pool.map(path_generator, range(samples_num, min(total_sample_num, samples_num + num_p*num_per)))
        # paths.append(strain)
        paths_temp = path_generator(x, x_, num_points,
                                    relative_length_range, s_range, s_theta_range, initial_theta_range)
        paths.append(paths_temp)
        samples_num += 1
        if samples_num % 10 == 0:
            print('Generated samples number %d' % samples_num)
    paths = np.array(paths)

    if only_plot:
        # plot the 1st 3 for checking
        for i in range(10):
            plt.plot(paths[i, :], label=['00', '01', '11'])
            plt.title("sample %d" % i)
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        # save the paths
        save_dir = '../loading_results'
        save_dir_name = os.path.join(save_dir, 'random_strain_path')
        check_mkdir(save_dir_name)
        file_name = os.path.join(save_dir_name, 'paths_%d.npy' % total_sample_num)
        with open(file_name, mode='wb') as f:
            np.save(f, paths)
        echo('The loading path is saved as %s' % file_name)


if __name__ == "__main__":
    main()
    # load the saved file
    # with open(os.path.join('random_strain_path', 'paths.npy'), mode='rb') as f:
    #     paths = np.load(f)
    # print()
