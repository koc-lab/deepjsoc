"""
This code and some of the related codes uses original DeepSC implementation from:
    https://github.com/13274086/DeepSC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from vector_quantize_pytorch import LFQ
from training_utils_estimator import ins_del_channel
from insertion_deletion import insert_regular_markers
from numba import njit
import numpy as np
import sys
from models_estimator import BI_Estimator

@njit
def channel_func(x,channel_input,pi,ps,pd,Nc,marker_seq,safety_bits):
    trainX = np.zeros((x.shape[0]*x.shape[1],safety_bits,safety_bits),dtype=np.float32)
    for i in range(x.shape[0]*x.shape[1]):
      c,mask = insert_regular_markers(channel_input[i,:], Nc, marker_seq)
      y, _ = ins_del_channel(c, pi,ps,pd,safety_bits)
      for j in range(safety_bits):
          trainX[i,j, 0:j] = -2*y[0,0:j] + 1;
          #trainX[i,j,0:j] = -y[0,0:j]
    return trainX
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
  
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9  
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, src_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        return x
        
class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output

class VQ_Layer(nn.Module):
    def __init__(self,embedding_dim,channel_dim,marker_enc_size,safety_bits,estimator_file):
        super(VQ_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        self.channel_dim = channel_dim
        self.ps = sys.float_info.min
        self.marker_enc_size =marker_enc_size
        self.marker_seq = np.array([0,1]).reshape(1,-1)
        self.Nc = 9
        _,self.mask = insert_regular_markers(np.zeros((self.channel_dim,)),self.Nc,self.marker_seq)
        self.safety_bits = safety_bits
        self.Nr = self.marker_seq.shape[-1]
        self.model = BI_Estimator(input_size=self.safety_bits,actual_size = self.marker_enc_size, d_rnn=128, d_mlp=[128, 32], num_bi_layers=3)
        self.model.load_state_dict(torch.load(estimator_file))
        for param in self.model.parameters():
            param.requires_grad =True
        self.vq = LFQ(
                codebook_size = 2**self.embedding_dim,      # codebook size, must be a power of 2
                dim = self.embedding_dim,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
                entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
                diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
            ).cuda()
    def forward(self,x,pd,ps):
        x_flatten = x.reshape(x.shape[0]*x.shape[1],self.channel_dim)
        x_flatten2 = x_flatten.reshape(x_flatten.shape[0],self.channel_dim//self.embedding_dim,self.embedding_dim)
        quantized_1,_,commit_loss =  self.vq(x_flatten2, inv_temperature=100.)
        quantized_1 = x_flatten2 + (quantized_1-x_flatten2).detach()
        channel_input = ((quantized_1.reshape(x.shape[0]*x.shape[1],x.shape[2])) + 1 ) /2
        marker_in = channel_input.detach().cpu().numpy().astype('float64')

        trainX = channel_func(x.detach().cpu().numpy(),marker_in,pd,ps,
        pd,self.Nc,self.marker_seq,self.safety_bits)      
        
        logits = self.model(torch.from_numpy(trainX).cuda())[:,:self.marker_enc_size]
        channel_out=logits.reshape(x.shape[0],x.shape[1],self.marker_enc_size)
        channel_out = channel_out.cuda()
        filtered_numbers = np.where(self.mask[0] == 0)[0].tolist()
        indices = torch.tensor(filtered_numbers).reshape(1,1,self.channel_dim)*torch.ones(x.shape[0],x.shape[1],self.channel_dim).to(torch.int64) 
        
        out_1 = channel_input.reshape(x.shape)
        
        outt = torch.gather(channel_out, 2, indices.cuda())
        outt_2 = outt + (((outt>0.5).long()*2-1)-outt).detach()
        return out_1 + (outt_2-out_1).detach(),commit_loss.cuda(),outt_2

class DeepJSOC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len, 
                 trg_max_len, d_model, num_heads, dff,vq_dim,channel_in_len,marker_enc_size,safety_len,estimator_file, dropout = 0.1):
        super(DeepJSOC, self).__init__()
        
        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len, 
                               d_model, num_heads, dff, dropout)
        self.vqlayer = VQ_Layer(vq_dim,channel_in_len,marker_enc_size,safety_len,estimator_file)
          
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), 
                                             #nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, channel_in_len))


        self.channel_decoder = ChannelDecoder(channel_in_len, d_model, 512)
        
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        
        self.dense = nn.Linear(d_model, trg_vocab_size)



    
        
        
        
        
        

    

    
    
    
    
    


    


