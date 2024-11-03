from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from utils import calculate_kl as KL_DIV
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


num_class = 7  # for single pulse
# num_class = 6  # for multi-modulation


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)

        out = (x - mean) / (std + self.eps)

        out = self.gamma * out + self.beta

        return out


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)


class BBBConv1d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, priors=None):

        super(BBBConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        else:
            pretrained_CKPT = torch.load(priors)
            mu_mean = torch.mean(pretrained_CKPT['model']['encoder.0.weight'])
            mu_var = torch.var(pretrained_CKPT['model']['encoder.0.weight'])
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (mu_mean, 0.1),
                'posterior_rho_initial': (-3, mu_var),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_channels, in_channels, self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv1d(input, weight, bias, self.stride, self.padding)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        else:
            pretrained_CKPT = torch.load(priors)
            mu_mean = torch.mean(pretrained_CKPT['model']['fc.weight'])
            mu_var = torch.var(pretrained_CKPT['model']['fc.weight'])
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (mu_mean, 0.1),
                'posterior_rho_initial': (-3, mu_var),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


class BBBTransConv1d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, priors=None):

        super(BBBTransConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        else:
            pretrained_CKPT = torch.load(priors)
            mu_mean = torch.mean(pretrained_CKPT['model']['encoder.0.weight'])
            mu_var = torch.var(pretrained_CKPT['model']['encoder.0.weight'])
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (mu_mean, 0.1),
                'posterior_rho_initial': (-3, mu_var),
            }

        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((in_channels, out_channels, self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((in_channels, out_channels, self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv_transpose1d(input, weight, bias, self.stride, self.padding)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


class BNN_Attention(nn.Module):
    """The architecture of LeNet with Bayesian Layers"""

    def __init__(self, outputs, inputs, priors, activation_type='relu'):
        super(BNN_Attention, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.encoder = nn.Sequential(
            BBBConv1d(in_channels=inputs, out_channels=32, kernel_size=3, stride=1, padding=0,
                      bias=True, priors=self.priors),  # 16,5
            # BBBConv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0,
            #           bias=True, priors=self.priors),
            # BBBConv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0,
            #           bias=True, priors=self.priors),
            # nn.BatchNorm1d(16)
        )
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=1),
        #     nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
        #     # nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=1),
        #     nn.BatchNorm1d(16)
        # )
        self.fc = BBBLinear(32, outputs, bias=True, priors=self.priors)

        # one trick of parameter initialization
        self.w_omega = Parameter(torch.randn(32, 32) / 50)
        self.u_omega = Parameter(torch.randn(32, 1) / 50)
        # self.w_omega = Parameter(torch.Tensor(32, 32))
        # self.u_omega = Parameter(torch.Tensor(32, 1))

    def forward(self, x):
        x2, _ = pad_packed_sequence(x, batch_first=True)
        x2 = x2.permute(0, 2, 1)

        # encoder-decoder
        x3 = self.encoder(x2)
        # x3 = self.decoder(x3)

        x3 = x3.permute(0, 2, 1)

        # self-attention
        u = torch.tanh(torch.matmul(x3, self.w_omega))  # size: [batch_size, seq_len, num_hiddens]
        att = torch.matmul(u, self.u_omega)  # size: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x3 * att_score
        feat = torch.sum(scored_x, dim=1)  # size: [batch_size, num_hiddens]

        # output
        out = F.softmax(self.fc(feat), dim=1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl

    # def count_your_model(model, x, y):


class BNN_mutiscale_Attention(nn.Module):
    """The architecture of LeNet with Bayesian Layers"""

    def __init__(self, outputs, inputs, priors, activation_type='relu'):
        super(BNN_mutiscale_Attention, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.bnn1 = BBBConv1d(in_channels=inputs, out_channels=16, kernel_size=3, stride=1, padding=0,
                              bias=True, priors=self.priors)
        self.bnn2 = BBBConv1d(in_channels=inputs, out_channels=16, kernel_size=3, stride=1, padding=0,
                              bias=True, priors=self.priors)

        self.fc = BBBLinear(32, outputs, bias=True, priors=self.priors)

        # one trick of parameter initialization
        self.w_omega = Parameter(torch.randn(32, 32) / 50)
        self.u_omega = Parameter(torch.randn(32, 1) / 50)
        # self.w_omega = Parameter(torch.Tensor(32, 32))
        # self.u_omega = Parameter(torch.Tensor(32, 1))

    def forward(self, x):
        # padding
        x2, _ = pad_packed_sequence(x, batch_first=True)
        x2 = x2.permute(0, 2, 1)

        # multi-scale BNN
        x3 = self.bnn1(x2)
        x4 = self.bnn2(x2)

        x5 = torch.cat((x3, x4), dim=1)

        x5 = x5.permute(0, 2, 1)

        # self-attention
        u = torch.tanh(torch.matmul(x5, self.w_omega))  # size: [batch_size, seq_len, num_hiddens]
        att = torch.matmul(u, self.u_omega)  # size: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x5 * att_score
        feat = torch.sum(scored_x, dim=1)  # size: [batch_size, num_hiddens]

        # output
        out = F.softmax(self.fc(feat), dim=1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl


class BNN_Attention_2(nn.Module):
    def __init__(self, outputs, inputs, priors, activation_type='relu'):
        super(BNN_Attention_2, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.encoder = nn.Sequential(
            BBBConv1d(in_channels=inputs, out_channels=32, kernel_size=3, stride=1, padding=0,
                      bias=True, priors=self.priors),  # 16,5
            # BBBConv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0,
            #           bias=True, priors=self.priors),
            # BBBConv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0,
            #           bias=True, priors=self.priors),
            # nn.BatchNorm1d(32)
        )

        self.fc = BBBLinear(32, outputs, bias=True, priors=self.priors)
        self.layer_norm = LayerNorm(d_model=32, eps=10e-12)

        # one trick of parameter initialization
        self.w_omega = Parameter(torch.randn(32, 32) / 50)
        self.u_omega = Parameter(torch.randn(32, 1) / 50)
        # self.w_omega = Parameter(torch.Tensor(32, 32))
        # self.u_omega = Parameter(torch.Tensor(32, 1))

    def forward(self, x):
        x2, _ = pad_packed_sequence(x, batch_first=True)
        x2 = x2.permute(0, 2, 1)

        # encoder-decoder
        x3 = self.encoder(x2)
        x3 = x3.permute(0, 2, 1)

        x3 = self.layer_norm(x3)
        # self-attention
        u = torch.tanh(torch.matmul(x3, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = x3 * att_score
        feat = torch.sum(scored_x, dim=1)

        # output
        out = F.softmax(self.fc(feat), dim=1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl


class BNN_Attention_encoder(nn.Module):
    """The architecture of LeNet with Bayesian Layers"""

    def __init__(self, outputs, inputs, priors, activation_type='relu'):
        super(BNN_Attention_encoder, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.encoder = nn.Sequential(
            BBBConv1d(in_channels=inputs, out_channels=32, kernel_size=3, stride=1, padding=0,
                      bias=True, priors=self.priors),  # 16,5
            # BBBConv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0,
            #           bias=True, priors=self.priors),
        )
        self.decoder = nn.Sequential(
            # BBBTransConv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=0,
            #                bias=True, priors=self.priors),
            BBBTransConv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0,
                           bias=True, priors=self.priors),
        )
        self.fc = BBBLinear(32, outputs, bias=True, priors=self.priors)
        # self.layer_norm = LayerNorm(d_model=32, eps=10e-12)

        # one trick of parameter initialization
        self.w_omega = Parameter(torch.randn(32, 32) / 50)
        self.u_omega = Parameter(torch.randn(32, 1) / 50)
        # self.w_omega = Parameter(torch.Tensor(32, 32))
        # self.u_omega = Parameter(torch.Tensor(32, 1))

    def forward(self, x):
        x2, _ = pad_packed_sequence(x, batch_first=True)
        x2 = x2.permute(0, 2, 1)

        # encoder-decoder
        x3 = self.encoder(x2)
        x3 = self.decoder(x3)

        x3 = x3.permute(0, 2, 1)

        # self-attention
        u = torch.tanh(torch.matmul(x3, self.w_omega))  # size: [batch_size, seq_len, num_hiddens]
        att = torch.matmul(u, self.u_omega)  # size: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x3 * att_score
        feat = torch.sum(scored_x, dim=1)  # size: [batch_size, num_hiddens]

        # output
        out = F.softmax(self.fc(feat), dim=1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl


class BNN_Attention_3(nn.Module):
    """The architecture of LeNet with Bayesian Layers"""

    def __init__(self, outputs, inputs, priors, activation_type='relu'):
        super(BNN_Attention_3, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.encoder = nn.Sequential(
            BBBConv1d(in_channels=inputs, out_channels=32, kernel_size=3, stride=1, padding=0,
                      bias=True, priors=self.priors),  # 16,5
        )
        self.decoder = nn.Sequential(
            BBBTransConv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
        )
        self.fc = BBBLinear(32, outputs, bias=True, priors=self.priors)
        self.layer_norm = LayerNorm(d_model=32, eps=10e-12)

        # one trick of parameter initialization
        self.w_omega = Parameter(torch.randn(32, 32) / 50)
        self.u_omega = Parameter(torch.randn(32, 1) / 50)
        # self.w_omega = Parameter(torch.Tensor(32, 32))
        # self.u_omega = Parameter(torch.Tensor(32, 1))

    def forward(self, x):
        x2, _ = pad_packed_sequence(x, batch_first=True)
        x2 = x2.permute(0, 2, 1)

        # encoder-decoder
        x3 = self.encoder(x2)
        x3 = self.decoder(x3)

        x3 = x3.permute(0, 2, 1)
        x3 = self.layer_norm(x3)

        # self-attention
        u = torch.tanh(torch.matmul(x3, self.w_omega))  # size: [batch_size, seq_len, num_hiddens]
        att = torch.matmul(u, self.u_omega)  # size: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x3 * att_score
        feat = torch.sum(scored_x, dim=1)  # size: [batch_size, num_hiddens]

        # output
        out = F.softmax(self.fc(feat), dim=1)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return out, kl

