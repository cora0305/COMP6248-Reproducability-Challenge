import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import random
import torch.nn.functional as F
import math

def Embedding(input_dim, model_dim, pad_idx=None, positional_embedding=False):
    """
    Upon choice, creates a positional or normal embedding layer
    """
    m = nn.Embedding(input_dim, model_dim, padding_idx=pad_idx)
    nn.init.normal_(m.weight, mean=0, std=model_dim ** -0.5)
    if pad_idx is not None:
        nn.init.constant_(m.weight[pad_idx], 0)
    if positional_embedding:
        weights = m.weight
        position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / model_dim) for j in range(model_dim)]
        for pos in range(input_dim)
    ])
        weights.requires_grad = False
        weights[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        weights[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        weights.detach_()
        m.weight = weights
    return m


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b,l, h, w = matrices.size()

    indices = torch.triu_indices(l, w, offset=0 if mask_diagonal else 1)
    matrices[:,indices[0],:, indices[1]] = maskval

def get_masks(slen, lengths, causal, config):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    # lenghts: (N)
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)
    mask = mask.to(config.device)
    attn_mask = attn_mask.to(config.device)
    return mask, attn_mask

class Multi_Head_Attention(nn.Module):
    """
    Creates the Multi Head Attention block in transformers
    """
    def __init__(self, config):
        super(Multi_Head_Attention, self).__init__()
        self.model_dim = config.model_dim
        self.head_dim = (self.model_dim // config.num_head)
        self.num_head = config.num_head
        assert self.head_dim*self.num_head == self.model_dim, 'Improper number of heads'

        self.to_query = nn.Linear(self.model_dim, self.head_dim*self.num_head)
        self.to_key = nn.Linear(self.model_dim, self.head_dim*self.num_head)
        self.to_value = nn.Linear(self.model_dim, self.head_dim*self.num_head)

        self.to_out = nn.Linear(self.head_dim*self.num_head, self.model_dim)

    def forward(self, Q, K, V, mask):
        """
        Run the forward path
        """
        # Q: (N, qlen, dm) K: (N, klen, dm) V:(N, vlen, dm)
        N, qlen, dm = Q.size()
        _, klen, _ = K.size()
        _, vlen, _ = V.size()
        mask_reshape = (N,qlen, 1, klen) if mask.dim() == 3 else (N, 1, 1, klen)
        assert dm == self.model_dim, 'Improper size model dimmention'
        # apply linear projection
        q = self.to_query(Q).view(N, qlen, self.num_head, -1) # q: (N, qlen, h, dh)
        k = self.to_key(K).view(N, klen, self.num_head, -1) # k: (N, klen, h, dh)
        v = self.to_value(V).view(N, vlen, self.num_head, -1) # v: (N, vlen, h, dh)
        dot = torch.einsum("bqhd,bkhd->bqhk", [q,k]).contiguous()/math.sqrt(self.head_dim) # dot: (N, qlen, h, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(dot)  # (N, qlen, h, klen)
        dot.masked_fill_(mask, -float('inf'))
        weights = F.softmax(dot.float(), dim = -1).type_as(dot) # weights: (N, qlen, h, klen)
        out = torch.einsum("bqhk,bkhd->bqhd", [weights, v]).contiguous() # out: (N, qlen, h, dh)
        out = out.view(N, qlen, -1) # out: (N, qlen, h*dh)
        out = self.to_out(out) # out: (N, qlen, dm)
        return out

class Feed_Forward(nn.Module):
    """
    the feed forward neural network block in the transformer model
    """
    def __init__(self, config):
        super(Feed_Forward, self).__init__()
        self.input_dim = config.model_dim
        self.hidden_dim = config.model_dim*config.forward_expansion
        self.out_dim = config.model_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
            )
    def forward(self, x):
        # x: (N, slen, dm)
        return self.fc(x)

class EncoderBlock(nn.Module):
    """
    The Encoder block of transformer
    """
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.model_dim = config.model_dim
        self.num_head = config.num_head
        self.forward_expansion = config.forward_expansion
        # Blocks
        self.attention = Multi_Head_Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.model_dim, eps=1e-12)
        self.fc = Feed_Forward(config)
        self.layer_norm2 = nn.LayerNorm(self.model_dim, eps=1e-12)
    def forward(self, x, atten_mask, mask):
        # x: (N, slen, dm) slen : source len
        N, slen, dm = x.size()
        assert mask.size() == (N, slen)
        assert atten_mask.size() == (N, slen)
        # attention block
        out = self.attention(x,x,x,atten_mask) + x
        out = self.layer_norm1(out)
        # Feed Forward
        out = self.fc(out) + out
        out = self.layer_norm2(out)
        out *= mask.unsqueeze(-1).to(out.dtype) #out: (N, slen, dm)
        return out


class DecoderBlock(nn.Module):
    """
    The Decoder block of transformer
    """
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.model_dim = config.model_dim
        self.num_head = config.num_head
        self.forward_expansion = config.forward_expansion
        # Blocks
        self.attention = Multi_Head_Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.model_dim, eps=1e-12)
        self.encoder_attention = Multi_Head_Attention(config)
        self.layer_norm15 = nn.LayerNorm(self.model_dim, eps=1e-12)
        self.fc = Feed_Forward(config)
        self.layer_norm2 = nn.LayerNorm(self.model_dim, eps=1e-12)
    def forward(self, x, encoded, atten_mask, mask, enc_mask):
        # x: (N, tlen, dm) tlen: target len
        N, tlen, dm = x.size()
        _, slen, _ = encoded.size()
        assert mask.size() == (N, tlen)
        assert atten_mask.size() == (N, tlen, tlen)
        assert (encoded.size(0) == N) == (encoded.size(2) == dm)
        assert enc_mask.size() == (N, slen)
        # attention block
        out = self.attention(x,x,x,atten_mask) + x
        out = self.layer_norm1(out)
        # Encoder attention
        out = self.encoder_attention(Q = out, V = encoded, K = encoded, mask = enc_mask) + out
        out = self.layer_norm15(out)
        # Feed Forward
        out = self.fc(out) + out
        out = self.layer_norm2(out)
        out *= mask.unsqueeze(-1).to(out.dtype) #out: (N, tlen, dm)
        return out

class Transformers(nn.Module):
    """
    Creates the transformer model besed on the parameters int the config instance
    """
    def __init__(self, config):
        super(Transformers, self).__init__()
        self.config = config
        self.n_words = config.n_words
        self.model_dim = config.model_dim
        self.num_head = config.num_head
        self.num_enc_layer = config.num_enc_layer
        self.num_dec_layer = config.num_dec_layer
        self.max_position = config.max_position
        self.pad_index = config.pad_index
        self.eos_index = config.eos_index
        # Encoder Embeddings
        self.enc_embedding = Embedding(input_dim = self.n_words, model_dim = self.model_dim, pad_idx=self.pad_index, positional_embedding=False)
        self.positional_embedding = Embedding(input_dim = self.max_position, model_dim = self.model_dim, positional_embedding=True)
        self.enc_emb_layer_norm = nn.LayerNorm(self.model_dim, eps=1e-12)
        # Decoder Embeddings
        self.dec_embedding = Embedding(input_dim = self.n_words, model_dim = self.model_dim, pad_idx=self.pad_index, positional_embedding=False)
        self.dec_emb_layer_norm = nn.LayerNorm(self.model_dim, eps=1e-12)
        # Encoder layers
        self.enc_layers = nn.ModuleList()
        for i in range(self.num_enc_layer):
            self.enc_layers.append(EncoderBlock(config))
        # Decoder layers
        self.dec_layers = nn.ModuleList()
        for i in range(self.num_dec_layer):
            self.dec_layers.append(DecoderBlock(config))
        # Output layer
        self.to_out = nn.Linear(self.model_dim, self.n_words)
        if config.share_inout_emb:
            self.to_out.weight = self.dec_embedding.weight

    def forward(self, mode, **kwargs):
        if mode == 'encode' :
            return self.Encode(**kwargs)
        elif mode == 'decode':
            return self.Decode(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)

        else:
            raise Exception(f"Invalid mode: {mode}")

    def Encode(self, x, len_x):
        """
        Encodes the input sequence using the encoder block
        """
        # x: (slen, N) lenghts: (N)
        slen, N = x.size()
        assert len_x.size(0) == N
        assert len_x.max().item() <= slen
        x = x.transpose(0, 1).contiguous()
        # Generate Masks
        e_mask , e_atten_mask = get_masks(slen, len_x, causal = False, config = self.config)
        # Positions
        positions = x.new(slen).long()
        positions = torch.arange(slen, out=positions).unsqueeze(0)
        # Embeddings
        tensor = self.enc_embedding(x)
        tensor = tensor + self.positional_embedding(positions).expand_as(tensor)
        tensor = self.enc_emb_layer_norm(tensor)
        tensor *= e_mask.unsqueeze(-1).to(tensor.dtype)
        # Encoder
        for i in range(self.num_enc_layer):
            tensor = self.enc_layers[i](x = tensor, atten_mask = e_atten_mask, mask = e_mask)
        # tensor: (N, slen, dm)
        tensor = tensor.transpose(0, 1).contiguous()
        # tensor: (slen, N, dm)
        return tensor

    def Decode(self, y, len_y, encoded, len_enc):
        """
        Decodes the input sequence using the output of the encoder block
        """
        # y: (tlen, N) len_y: (N)
        tlen, N = y.size()
        _, slen, _ = encoded.size()
        assert len_y.size(0) == N
        assert len_y.max().item() <= tlen
        assert encoded.size(0) == N
        assert len_enc.max().item() <= slen
        y = y.transpose(0, 1).contiguous()
        # generate masks
        e_mask , _ = get_masks(slen, len_enc, causal = False, config = self.config)
        d_mask , d_atten_mask = get_masks(tlen, len_y, causal = True, config = self.config)
        # Positions
        positions = y.new(tlen).long()
        positions = torch.arange(tlen, out=positions).unsqueeze(0)
        # Embeddings
        tensor = self.dec_embedding(y)
        tensor = tensor + self.positional_embedding(positions).expand_as(tensor)
        tensor = self.dec_emb_layer_norm(tensor)
        tensor *= d_mask.unsqueeze(-1).to(tensor.dtype)
        # Decoder
        for i in range(self.num_dec_layer):
            tensor = self.dec_layers[i](x = tensor, encoded = encoded, atten_mask = d_atten_mask, mask = d_mask, enc_mask = e_mask)
        # tensor: (N, tlen, dm)
        tensor = tensor.transpose(0, 1).contiguous()
        # tensor: (tlen, N, dm)
        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.model_dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.to_out(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores, y, reduction='mean')
        return scores, loss
