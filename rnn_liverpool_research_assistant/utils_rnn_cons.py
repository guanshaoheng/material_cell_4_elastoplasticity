import torch
import numpy as np


def get_q_2d(sig: torch.Tensor, plain_stress_flag=True):
    """
        Under plane stress assumption:
            q = \sqrt{(\sigma_{00} - \sigma_{11})^2 + 4 \sigma_{01}^2 }


    :param sig:  in shape of (num_steps, num_samples, (00, 01, 11))
    :return:

    s = sig - np.trace(sig)/2.*np.eye(2)
    q = np.sqrt(2*np.sum(s*s))

    """
    if plain_stress_flag == True:
        q = torch.sqrt((sig[..., 0:1] - sig[..., 2:3]) ** 2 + 4 * sig[..., 1:2] ** 2)
    else:
        raise ValueError('Only plain stress is involved currently!')
    return q


def get_p_2d(sig, twoD_flaf=True):
    """

    :param sig:       2D   in shape of (num_steps, num_samples, (00, 01, 11))
                      3D   in shape of (....)
    :param twoD_flaf:
    :return:
    """
    if twoD_flaf:  # 2D
        p = (sig[..., 0:1] + sig[..., 2:3]) / 2.
    else:  # 3D
        raise ValueError('Please set the voigt order for the 3D situation')
    return p


def get_q_2d_arr(sig: np.ndarray, plain_stress_flag=True):
    if plain_stress_flag == True:
        q = np.sqrt((sig[..., 0:1] - sig[..., 2:3]) ** 2 + 4 * sig[..., 1:2] ** 2)
    else:
        raise ValueError('Only plain stress is involved currently!')
    return q


def get_p_2d_arr(sig: np.ndarray, twoD_flaf=True):
    if twoD_flaf:  # 2D
        p = (sig[..., 0:1] + sig[..., 2:3]) / 2.
    else:  # 3D
        raise ValueError('Please set the voigt order for the 3D situation')
    return p