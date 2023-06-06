import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DyNCA(torch.nn.Module):
    SEED_MODES = ['random', 'center_on', 'zeros']
    """
    Parameters
    ----------
    c_in: int, required
        Number of channels in the input
        Note that each channel will be processed 
        using 3, or 4 (if laplacian=True) convolution filters
    c_out: int, required
        Number of channels in the output
        Note that the NCA will be performed using c_in channels
        and the output of the NCA will be expanded to c_out 
        channels using a learnable 1x1 convolution layer 
    fc_dim: int, default=94
        Number of channels in the intermediate fully connected layer
    random_seed: int, default=None
    seed_mode: {'zeros', 'center_on', 'random'}, default='constant'
        Type of the seed used to initialize the cellular automata
    device: pytorch device
        Device used for performing the computation.
    """

    def __init__(self, c_in, c_out, fc_dim=96,
                 padding_mode='replicate',
                 seed_mode='zeros', pos_emb='CPE',
                 perception_scales=[0],
                 device=torch.device("cuda:0")):

        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.perception_scales = perception_scales
        self.fc_dim = fc_dim
        self.padding_mode = padding_mode
        assert seed_mode in DyNCA.SEED_MODES
        self.seed_mode = seed_mode
        self.random_seed = 42
        self.pos_emb = pos_emb
        self.device = device
        self.expand = 4

        self.c_cond = 0
        if self.pos_emb == 'CPE':
            self.pos_emb_2d = CPE2D()
            self.c_cond += 2
        else:
            self.pos_emb_2d = None

        self.w1 = torch.nn.Conv2d(self.c_in * self.expand + self.c_cond, self.fc_dim, 1, device=self.device)
        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)

        self.w2 = torch.nn.Conv2d(self.fc_dim, self.c_in, 1, bias=True, device=self.device)
        torch.nn.init.xavier_normal_(self.w2.weight, gain=0.1)
        torch.nn.init.zeros_(self.w2.bias)

        self.sobel_filter_x = torch.FloatTensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).to(
            self.device)
        self.sobel_filter_y = self.sobel_filter_x.T

        self.identity_filter = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).to(self.device)
        self.laplacian_filter = torch.FloatTensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]).to(
            self.device)

    def perceive_torch(self, x, scale=0):
        assert scale in [0, 1, 2, 3, 4, 5]
        if scale != 0:
            _, _, h, w = x.shape
            h_new = int(h // (2 ** scale))
            w_new = int(w // (2 ** scale))
            x = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)

        def _perceive_with_torch(z, weight):
            conv_weights = weight.reshape(1, 1, 3, 3).repeat(self.c_in, 1, 1, 1)
            z = F.pad(z, [1, 1, 1, 1], self.padding_mode)
            return F.conv2d(z, conv_weights, groups=self.c_in)

        y1 = _perceive_with_torch(x, self.sobel_filter_x)
        y2 = _perceive_with_torch(x, self.sobel_filter_y)
        y3 = _perceive_with_torch(x, self.laplacian_filter)

        tensor_list = [x]
        tensor_list += [y1, y2, y3]

        y = torch.cat(tensor_list, dim=1)

        if scale != 0:
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)

        return y

    def perceive_multiscale(self, x, pos_emb_mat=None):
        perceptions = []
        y = 0
        for scale in self.perception_scales:
            z = self.perceive_torch(x, scale=scale)
            perceptions.append(z)

        y = sum(perceptions)
        y = y / len(self.perception_scales)

        if pos_emb_mat is not None:
            y = torch.cat([y, pos_emb_mat], dim=1)

        return y

    def forward(self, x, update_rate=0.5, return_perception=False):
        if self.pos_emb_2d:
            y_percept = self.perceive_multiscale(x, pos_emb_mat=self.pos_emb_2d(x))
        else:
            y_percept = self.perceive_multiscale(x)
        y = self.w2(F.relu(self.w1(y_percept)))
        b, c, h, w = y.shape

        update_mask = (torch.rand(b, 1, h, w, device=self.device) + update_rate).floor()

        x = x + y * update_mask

        if return_perception:
            return x, self.to_rgb(x), y_percept
        else:
            return x, self.to_rgb(x)

    def to_rgb(self, x):
        return x[:, :self.c_out, ...] * 2.0

    def seed(self, n, size=128):
        if isinstance(size, int):
            size_x, size_y = size, size
        else:
            size_x, size_y = size

        if self.seed_mode == 'zeros':
            sd = torch.zeros(n, self.c_in, size_y, size_x).to(self.device)
            return sd
        elif self.seed_mode == 'center_on':
            sd = torch.zeros(n, self.c_in, size_y, size_x).to(self.device)
            sd[:, :, size_y // 2, size_x // 2] = 1.0
            return sd
        elif self.seed_mode == 'random':
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            sd = (torch.rand(1, self.c_in, size_y, size_x) - 0.5)
        else:
            sd = None

        sd = torch.cat([sd.clone() for _ in range(n)]).to(self.device)

        return sd

    def forward_nsteps(self, input_state, step_n, update_rate=0.5, return_middle_feature=False):
        nca_state = input_state
        middle_feature_list = []
        for _ in range(step_n):
            nca_state, nca_feature = self(nca_state, update_rate=update_rate)
            if return_middle_feature:
                middle_feature_list.append(nca_feature)
        if return_middle_feature:
            return nca_state, nca_feature, middle_feature_list
        return nca_state, nca_feature


class CPE2D(nn.Module):
    """
    Cartesian Positional Encoding 2D
    """

    def __init__(self):
        super(CPE2D, self).__init__()
        self.cached_penc = None
        self.last_tensor_shape = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, ch, x, y)
        :return: Positional Encoding Matrix of size (batch_size, 2, x, y)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.last_tensor_shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, orig_ch, h, w = tensor.shape
        xs = torch.arange(h, device=tensor.device) / h
        ys = torch.arange(w, device=tensor.device) / w
        xs = 2.0 * (xs - 0.5 + 0.5 / h)
        ys = 2.0 * (ys - 0.5 + 0.5 / w)
        xs = xs[None, :, None]
        ys = ys[None, None, :]
        emb = torch.zeros((2, h, w), device=tensor.device).type(tensor.type())
        emb[:1] = xs
        emb[1: 2] = ys

        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.last_tensor_shape = tensor.shape

        return self.cached_penc
