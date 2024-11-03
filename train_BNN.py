from dataset import Dataset_pulse
from torch.utils.data import DataLoader
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from utils import adjust_learning_rate, collate_fn, logmeanexp
from BNN import BNN_Attention_encoder, BNN_mutiscale_Attention, BNN_Attention, num_class
import torch
import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--bs', default=40, type=int, help='batch_size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning_rate')
parser.add_argument('--epoch', default=200, type=int, help='max_epoch')
parser.add_argument('--lr_rate', default=20, type=int, help='lr_update_freq')
parser.add_argument('--dir', required=True, type=str, help='dataset_dir')
args = parser.parse_args()

# prepare data
print('loading data...')
signals_train = Dataset_pulse(args.dir, filename='pulse_noisy_miss_0.02_0.0.pkl', mode='train')  # pulse stream data
# signals_train = Dataset_pulse(args.dir, filename='pulse_noisy_miss_fake_0.02_0.0_0.0.pkl', mode='train')  # multi-modulation
dataloader_train = DataLoader(signals_train, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)

signals_valid = Dataset_pulse(args.dir, filename='pulse_noisy_miss_0.02_0.0.pkl', mode='test')
# signals_valid = Dataset_pulse(args.dir, filename='pulse_noisy_miss_fake_0.02_0.0_0.0.pkl', mode='test')
dataloader_valid = DataLoader(signals_valid, batch_size=args.bs, shuffle=True, collate_fn=collate_fn)

# prepare model
model = BNN_Attention_encoder(outputs=num_class, inputs=3, priors=None, activation_type='relu')
# model = BNN_Attention(outputs=num_class, inputs=3, priors=None, activation_type='relu')
# model = BNN_mutiscale_Attention(outputs=num_class, inputs=3, priors=None, activation_type='relu')
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = metrics.ELBO(len(dataloader_train))
criterion.cuda()

acc_train_list = []
acc_valid_list = []
loss_train_list = []
loss_valid_list = []
best_acc = 0

print('begin training')
for epoch in range(args.epoch + 1):
    training_loss = 0.0
    accs = []
    kl_list = []
    num_ens = 1
    beta_type = 'standard'
    optimizer = adjust_learning_rate(optimizer, epoch, args.lr_rate)
    pbar = tqdm(total=len(dataloader_train))
    for i, (inputs, labels) in enumerate(dataloader_train, 1):

        model.train()
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = torch.zeros(labels.shape[0], model.num_classes, num_ens).cuda()

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = model(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i - 1, len(dataloader_train), beta_type, epoch, args.epoch)
        loss = criterion(log_outputs, labels.squeeze().long(), kl, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()

    pbar.close()

    train_loss = training_loss / len(dataloader_train)
    loss_train_list.append(train_loss)

    train_acc = np.mean(accs)
    acc_train_list.append(train_acc)

    train_kl = np.mean(kl_list)

    # print test accuracy after each epoch
    with torch.no_grad():
        valid_loss = 0.0
        accs = []
        for idx, data in enumerate(dataloader_valid):
            model.eval()
            signal_test, labels_test = data
            signal_test, labels_test = signal_test.cuda(), labels_test.cuda()
            outputs = torch.zeros(labels_test.shape[0], model.num_classes, num_ens).cuda()
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = model(signal_test)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

            log_outputs = logmeanexp(outputs, dim=2)

            beta = metrics.get_beta(idx - 1, len(dataloader_valid), beta_type, epoch, args.epoch)

            valid_loss += criterion(log_outputs, labels_test.squeeze().long(), kl, beta).item()
            accs.append(metrics.acc(log_outputs, labels_test))

        valid_loss = valid_loss / len(dataloader_valid)
        loss_valid_list.append(valid_loss)

        valid_acc = np.mean(accs)
        acc_valid_list.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            # save model
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, 'BNN.pth.tar')

    print(
        'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation '
        'Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

# plot
plt.figure(1)
plt.plot(acc_train_list, label='train')
plt.plot(acc_valid_list, label='valid')
plt.legend()
plt.ylim((0, 1))
plt.xlabel('epoch')
plt.ylabel('accuracy')
# plt.show()
plt.savefig('acc_average.png')

plt.figure(2)
plt.plot(loss_train_list, label='train')
plt.plot(loss_valid_list, label='valid')
plt.legend()
# plt.ylim((0, 1))
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()
plt.savefig('loss_average.png')

