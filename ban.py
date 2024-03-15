import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import math
import numpy as np



class BANLayer1(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(v_dim, hid_dim)
        self.w_k = nn.Linear(q_dim, hid_dim)
        self.w_v1 = nn.Linear(v_dim, hid_dim)
        self.w_v2 = nn.Linear(q_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

        self.bn = nn.BatchNorm1d(hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

        PE =True
        self.PosEncV = PositionalEncoding(v_dim, 0.2, PE)
        self.PosEncQ = PositionalEncoding(q_dim, 0.2, PE)

    def attention_pooling(self, q, k, att_map):
        fusion_logits = torch.einsum('bhvk,bhvq,bhqk->bhk', (q, att_map, k))
        return fusion_logits

    def forward(self, query, key, per=0.2, att=None, softmax=False):
        bsz = query.shape[0]
        if att != None:
            len_pro = int(per * att.shape[-1])
            # att = att.sum(dim=1)
            att, _ = att.max(dim=1)
            att = torch.argsort(att, dim=1, descending=True)[:, :len_pro]

            for idx in range(bsz):
                if idx == 0:
                    vv = query[idx, att[idx, :], :].unsqueeze(0)
                else:
                    vv = torch.cat([vv, query[idx, att[idx, :], :].unsqueeze(0)], dim=0)
            query = vv

        # query = key [batch_size, seq_len, hid_dim]
        Q = self.do(self.w_q(self.PosEncV(query)))
        K = self.do(self.w_k(self.PosEncV(key)))
        V1 = self.do(self.w_v1(query))
        V2 = self.do(self.w_v2(key))

        # Q, K = [batch_size, seq_len, hid_dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V1 = V1.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V2 = V2.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K = [batch_size, n_heads, seq_len_K, hid_dim // n_heads]
        # Q = [batch_size, n_heads, seq_len_q, hid_dim // n_heads]
        # attention = [batch_size, n_heads, seq_len_q, seq_len_K]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if softmax:
            attention = F.softmax(attention, dim=-1)


        # x = [batch_size, n_heads, hid_dim // n_heads]
        x = self.attention_pooling(V1, V2, attention)
        # x = [batch_size, hid_dim]
        x = x.view(bsz, self.hid_dim)
        x = self.bn(x)

        attention = attention.sum(dim=1) / attention.shape[1]

        return x, att, attention

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, hid_dim, dropout, PE, max_len=2000):
        """
        three parameters:
        hid_dim：sequence_dim
        dropout: dropout rate
        max_len：the max len of the sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hid_dim)  # [max_len,hid_dim]
        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0., hid_dim, 2) *
                             -(math.log(10000.0) / hid_dim))  # [1,hid_dim/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1,max_len,hid_dim]
        self.register_buffer('pe', pe)  # regist buffer, not update parameters
        self.PE=PE

    def forward(self, x):  # x = [1,wordnum,hid_dim]
        # x+position   x.size(1) is the sequence_len
        if self.PE:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
            x = self.dropout(x)
        return x

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        self.vv_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.vq_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

        PE = True
        self.PosEncV = PositionalEncoding(v_dim, 0.2, PE)
        self.PosEncQ = PositionalEncoding(q_dim, 0.2, PE)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, per=0.2, att=None, softmax=False):
        if att != None:
            protein_len = att.shape[-1]
            batch_size = att.shape[0]
            for idx in range(0, batch_size):
                setidx = set({})
                atti = torch.argsort(att[idx, :, :].reshape(-1), dim=0, descending=True)
                atti = np.mod(atti.cpu().numpy(), protein_len)
                for idxx in atti:
                    setidx = setidx | {idxx}
                    if len(setidx) >= int(per * att.shape[-1]):
                        break
                atti = torch.tensor(np.array(list(setidx)))
                if idx == 0:
                    atts = atti.unsqueeze(0)
                else:
                    atts = torch.cat((atts, atti.unsqueeze(0)), 0)
                aats, _ = torch.sort(atts, dim=1, descending=False)

            vv = v[:, :int(per*att.shape[-1]), :]
            for idx in range(vv.shape[0]):
                vv[idx,:] = v[idx,atts[idx,:],:]
            v=vv
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(self.PosEncV(v))
            q_ = self.q_net(self.PosEncQ(q))
            vv_ = self.vv_net(v)
            vq_ = self.vq_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            vv_ = self.vv_net(v)
            vq_ = self.vq_net(q)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(vv_, vq_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(vv_, vq_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)

        return logits, att_maps
