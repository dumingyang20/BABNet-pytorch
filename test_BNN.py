from dataset import Dataset_pulse
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import torch.nn.functional as F
from utils import collate_fn, logmeanexp
from BNN import BNN_Attention, BNN_Attention_encoder, num_class, BNN_mutiscale_Attention
import torch
import metrics
from tqdm import tqdm
import csv

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--bs', default=40, type=int, help='batch_size')
parser.add_argument('--dataset_dir',
                    required=True,
                    type=str, help='your dataset')
parser.add_argument('--model_dir',
                    required=True,
                    type=str, help='your model')
args = parser.parse_args()


def run(bs, dataset_dir, filename, model_dir):
    # prepare data
    print('loading data...')
    signals_test = Dataset_pulse(dataset_dir, filename=filename,
                                 mode='test')  # pulse stream data
    dataloader_test = DataLoader(signals_test, batch_size=bs, shuffle=True, collate_fn=collate_fn)

    # load model
    model_CKPT = torch.load(model_dir)

    model = BNN_Attention(outputs=7, inputs=3, priors=None, activation_type='relu')
    # model = BNN_Attention_encoder(outputs=7, inputs=3, priors=None, activation_type='relu')
    # model = BNN_mutiscale_Attention(outputs=7, inputs=3, priors=None, activation_type='relu')

    model.load_state_dict(model_CKPT['model'])
    model.cuda()
    model.eval()

    criterion = metrics.ELBO(len(dataloader_test))
    criterion.cuda()

    # print test accuracy after each epoch
    test_loss = 0.0
    accs = []
    pbar = tqdm(total=len(dataloader_test))
    num_ens = 1
    beta_type = 'Standard'
    confusion_matrix = torch.zeros(num_class, num_class)
    for idx, data in enumerate(dataloader_test):
        signal_test, labels_test = data
        signal_test, labels_test = signal_test.cuda(), labels_test.cuda()
        outputs = torch.zeros(labels_test.shape[0], model.num_classes, num_ens).cuda()
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = model(signal_test)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(idx - 1, len(dataloader_test), beta_type, None, None)
        test_loss += criterion(log_outputs, labels_test.squeeze().long(), kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels_test))

        # calculate confusion matrix
        labels_np = labels_test.squeeze().cpu().detach().numpy()
        _, predicted = torch.max(log_outputs.data, 1)
        predicted_np = predicted.squeeze().cpu().detach().numpy()
        for t, p in zip(labels_np, predicted_np):
            confusion_matrix[int(t), int(p)] += 1

        pbar.update(1)

    pbar.close()

    test_loss = test_loss / len(dataloader_test)
    test_acc = np.mean(accs)

    print('test accuracy: %.3f, ' % test_acc,
          'test loss: %.3f' % test_loss)

    print(confusion_matrix.diag() / confusion_matrix.sum(1))

    print()

    return test_loss, test_acc


# get dataset
# filenames = os.listdir(args.dataset_dir)
filenames = 'pulse_noisy_miss_0.02_0.0.pkl'

# ratio = [float(x.split('_')[-1][0:3]) for x in filenames]  # for fake
ratio = [float(x.split('_')[-1][0:-4]) for x in filenames]  # for missing
# ratio = [float(x.split('_')[-2]) for x in filenames]  # for noisy

# get model
models_dir = []
for root, dirs, files in os.walk(args.model_dir):
    for filename in files:
        if 'BNN.pth.tar' in filename:
            models_dir.append(root + "/" + filename)

# save results
f = open('acc' + '_' + args.dataset_dir.split('/')[-1] + '.csv', 'w', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow([args.dataset_dir.split('/')[-1]] + ratio + ['average'])  # table head
for model_item in models_dir:
    acc_list = []
    model_idx = model_item.split('/')[-2].split('\\')[-1]
    for item in filenames:
        loss, acc = run(args.bs, args.dataset_dir, item, model_item)
        acc_list.append(acc)

    acc_list.append(sum(acc_list) / len(acc_list))
    csv_writer.writerow([model_idx] + acc_list)

f.close()
