import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
# input_window = 100 # number of input steps
# output_window = 1 # number of prediction steps, in this model its fixed to one
# batch_size = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=10, max_len=5000):
        super(PositionalEncoding, self).__init__()
        if (d_model % 2) == 0:
            pe = torch.zeros(max_len, d_model) # 5000, 10
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 5000,1 
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [5]
            # position * div_term : 5000,5
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)  # torch.Size([max_len, 1, d_model]) # torch.Size([5000, 1, 2])
        else:
            pe = torch.zeros(max_len, d_model) # 5000, 5
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 5000,1 
            sin_div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [5]
            cos_div_term = torch.exp(torch.arange(0, d_model-1, 2).float() * (-math.log(10000.0) / d_model)) # [5]
            # position * div_term : 5000,3
            pe[:, 0::2] = torch.sin(position * sin_div_term)
            pe[:, 1::2] = torch.cos(position * cos_div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
        
        ''' 
        RuntimeError: The expanded size of the tensor (2) 
        must match the existing size (3) at non-singleton dimension 1.  
        Target sizes: [5000, 2].  Tensor sizes: [5000, 3]
        '''
        # torch.Size([max_len, 1, d_model]) # torch.Size([5000, 1, 2])
        # pe.requires_grad = False
        self.register_buffer('pe', pe) ## 매개변수로 간주하지 않기 위한 것

    def forward(self, x):
        # pe :torch.Size([5000, 1, 2])
        # x : torch.Size([10, 64, 2])
        # self.pe[:x.size(0), :] : torch.Size([10, 1, 2])

        return x + self.pe[:x.size(0), :]

class Transformer(nn.Module):
    def __init__(self,m1_input_feature_size,m2_input_feature_size,m3_input_feature_size,
                Transformer_feature_size, nhead,args, 
                num_layers=1, dropout=0.1, x_frames = 10):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.src1_mask = None
        self.src2_mask = None
        self.src3_mask = None
        self.args =args
        ## ----- Modality1 Transformer ------ ##
        self.m1_pos_encoder = PositionalEncoding(m1_input_feature_size,5000)
        self.m1_encoder_layer = EncoderLayer(m1_input_feature_size, nhead, dropout)
        self.m1_decoder = nn.Linear(x_frames,1)

        ## ------ Modality2 Transformer ------ ##
        self.m2_pos_encoder = PositionalEncoding(m2_input_feature_size,5000)
        self.m2_encoder_layer = EncoderLayer(m2_input_feature_size, nhead, dropout)
        self.m2_decoder = nn.Linear(x_frames,1)

        ## ------ Modality3 Transformer ------ ##
        self.m3_pos_encoder = PositionalEncoding(m3_input_feature_size,5000)
        self.m3_encoder_layer = EncoderLayer(m3_input_feature_size, nhead, dropout)
        self.m3_decoder = nn.Linear(x_frames,1)

        self.init_weights()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sigmoid = nn.Sigmoid()
        
        self.Multimodal_linear = nn.Linear(m1_input_feature_size+m2_input_feature_size+m3_input_feature_size,1)


    def init_weights(self):
        initrange = 0.1

        self.m1_decoder.bias.data.zero_()
        self.m1_decoder.weight.data.uniform_(-initrange, initrange)
        self.m2_decoder.bias.data.zero_()
        self.m2_decoder.weight.data.uniform_(-initrange, initrange)
        self.m3_decoder.bias.data.zero_()
        self.m3_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src1,src2,src3):
        
        #print(src1.size()) # [64, 10, 17]
        #print(src2.size()) # [64, 10, 17]
        #print(src3.size()) # [64, 10, 6]

                
        if self.src1_mask is None or self.src1_mask.size(0) != len(src1):
            device = src1.device
            mask = self._generate_square_subsequent_mask(len(src1)).to(device)
            self.src1_mask = mask
        if self.src2_mask is None or self.src2_mask.size(0) != len(src2):
            device = src2.device
            mask = self._generate_square_subsequent_mask(len(src2)).to(device)
            self.src2_mask = mask
        if self.src2_mask is None or self.src2_mask.size(0) != len(src3):
            device = src3.device
            mask = self._generate_square_subsequent_mask(len(src3)).to(device)
            self.src3_mask = mask    
        
        src1 = self.m1_pos_encoder(src1) # [64, 10, 17]
        src2 = self.m2_pos_encoder(src2) # [64, 10, 17]
        src3 = self.m3_pos_encoder(src3) # [64, 10, 6]
            
        encoder_output1, attn_prob1 = self.m1_encoder_layer(src1) #, self.src_mask) [64, 10,64]
        encoder_output2, attn_prob2 = self.m2_encoder_layer(src2) # [64, 10, 64]
        encoder_output3, attn_prob3 = self.m3_encoder_layer(src3) # [64, 10, 64]
  
        transformer_output1 = self.m1_decoder(encoder_output1.transpose(1,2)).squeeze()  # 64, 50
        transformer_output2 = self.m2_decoder(encoder_output2.transpose(1,2)).squeeze()
        transformer_output3 = self.m3_decoder(encoder_output3.transpose(1,2)).squeeze()


        concatted_output = torch.cat((transformer_output1,transformer_output2,transformer_output3),dim = 1)
        output = self.Multimodal_linear(concatted_output).squeeze()
        
        output = self.sigmoid(output)

        return output, attn_prob1, attn_prob2 ,attn_prob3

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self,input_feature_size,nhead,dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(input_feature_size,nhead,input_feature_size)
        self.layer_norm1 = nn.LayerNorm(input_feature_size, eps=1e-12)
        self.pos_ffn = PoswiseFeedForwardNet(input_feature_size,dropout)
        self.layer_norm2 = nn.LayerNorm(input_feature_size, eps=1e-12)
    
    def forward(self, inputs):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)

        # [32, 64, 10]
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob






""" multi head attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidn, n_head, d_head):
        super().__init__()
        self.d_hidn = d_hidn
        self.n_head = n_head
        self.d_head = d_head

        self.W_Q = nn.Linear(d_hidn, n_head * d_head)
        self.W_K = nn.Linear(d_hidn, n_head * d_head)
        self.W_V = nn.Linear(d_hidn, n_head * d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(d_head)
        self.linear = nn.Linear(n_head * d_head, d_hidn)
    
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)

        # print(Q.size()) #torch.Size[32, 10, 40]) torch.Size([32, 6, 10, 40])


        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2).transpose(-1, -2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2).transpose(-1, -2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2).transpose(-1, -2)
        
        #print(q_s.size()) # ([32, 6, 40, 10]
        #print(k_s.size()) #([32, 6, 40, 10]
        #print(v_s.size()) # ([32, 6, 40, 10]
        # (bs, n_head, n_q_seq, n_k_seq)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s)
        
        # print(attn_prob.size()) # [32, 6, 40, 40]
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 3).contiguous().view(batch_size, -1, self.n_head * self.d_head)  #([32, 10, 40, 6]
        # (bs, n_head, n_q_seq, e_embd)
        #print(context.size()) # [32, 10, 240])
        output = self.linear(context)
        # print(output.size()) # [32, 10, 40]
        # print(output.size()) [32, 10, 40])
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob



""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.scale = 1 / (d_head ** 0.5)
    
    def forward(self, Q, K, V): #  ([32, 6, 40, 10]
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale) #([32, 6, 40, 40]
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V) #([32, 6, 40, 10]
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob



""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,input_feature_size,dropout):
        super().__init__()

        self.linear1 = nn.Linear(input_feature_size,input_feature_size*4)
        self.linear2 = nn.Linear(input_feature_size*4,input_feature_size)
        self.active = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):

        #print(inputs.size()) #[32, 10, 40]
        # (bs, d_ff, n_seq)
        
        output = self.linear1(inputs)
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.linear2(output)
        output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        # print(output.size()) # [32, 10, 40]

        return output


 