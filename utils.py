import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import csv


def split_data(data, mode=None):

    if mode == 'train':  # 60%
        np.random.seed(20)
        np.random.shuffle(data)
        data_info = data[:int(0.6 * len(data))]

    elif mode == 'test':  # 20% = 60%->80%
        np.random.seed(20)
        np.random.shuffle(data)
        data_info = data[int(0.6 * len(data)):int(0.8 * len(data))]
        # data_info = data[int(0.8 * len(data)):int(0.9 * len(data))]

    else:  # 20% = 80%->100%
        data_info = data[int(0.8 * len(data)):]

    return data_info


def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    signal, label = [data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))]

    seq_len = [s.size(0) for s in signal]

    label = torch.tensor(np.array(label))
    signal = pad_sequence(signal, batch_first=True)
    # label = pad_sequence(label, batch_first=True)

    # idx = torch.randperm(signal.shape[0])
    # signal = signal[idx].view(signal.size())
    # label = label[idx].view(label.size())

    signal = pack_padded_sequence(signal, seq_len, batch_first=True)
    # label = pack_padded_sequence(label, seq_len, batch_first=True)

    return signal, label


def collate_fn_2(data):
    """

    :param data: raw inter-pulse parameter data
    :return: padding with same length
    only for classifying label=0 and 4 data
    """
    signal, label = [data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))]
    label = torch.tensor(np.array(label))
    signal = pad_sequence(signal, batch_first=True)
    signal = signal.permute(0, 2, 1)
    return signal, label


def PostionalEncoding(dim, len, device):
    """
    compute sinusoid encoding.
    """
    # same size with input matrix (for adding with input matrix)
    encoding = torch.zeros(dim, len, device=device)
    encoding.requires_grad = False  # we don't need to compute gradient

    pos = torch.arange(0, dim, device=device)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, len, step=2, device=device).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / dim)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / dim)))

    return encoding


def get_index(lst=None, item=None):
    return [index for (index, value) in enumerate(lst) if value == item]


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def padding(data, target_length):
    assert type(target_length) is int
    assert len(data) <= target_length
    data_ = np.append(data, np.zeros((target_length-len(data), 3)), axis=0)
    # data_ = np.append(data, np.zeros((target_length - len(data), 4)), axis=0)

    return data_


def sort_csv_file(file):
    data = pd.read_csv(file, sep=',', header='infer')
    # sort by first column
    new_data = data.sort_values(by=data.columns[0], ascending=True)
    # new filename
    new_file = file.split('/')[-1].split('.')[0] + str(2) + '.csv'
    new_data.to_csv(new_file, mode='a+', index=False)


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)
