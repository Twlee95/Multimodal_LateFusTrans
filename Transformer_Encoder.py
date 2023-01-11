import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

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
        self.m1_input_linear = nn.Linear(m1_input_feature_size,Transformer_feature_size)
        self.m1_pos_encoder = PositionalEncoding(Transformer_feature_size,5000)
        self.m1_encoder_layer = nn.TransformerEncoderLayer(d_model=Transformer_feature_size, nhead=nhead, dropout=dropout)
        self.m1_transformer_encoder = nn.TransformerEncoder(self.m1_encoder_layer, num_layers=num_layers)
        self.m1_decoder = nn.Linear(x_frames,1)

        ## ------ Modality2 Transformer ------ ##
        self.m2_input_linear = nn.Linear(m2_input_feature_size,Transformer_feature_size)
        self.m2_pos_encoder = PositionalEncoding(Transformer_feature_size,5000)
        self.m2_encoder_layer = nn.TransformerEncoderLayer(d_model=Transformer_feature_size, nhead=nhead, dropout=dropout)
        self.m2_transformer_encoder = nn.TransformerEncoder(self.m2_encoder_layer, num_layers=num_layers)
        self.m2_decoder = nn.Linear(x_frames,1)

        ## ------ Modality3 Transformer ------ ##
        self.m3_input_linear = nn.Linear(m3_input_feature_size,Transformer_feature_size)
        self.m3_pos_encoder = PositionalEncoding(Transformer_feature_size,5000)
        self.m3_encoder_layer = nn.TransformerEncoderLayer(d_model=Transformer_feature_size, nhead=nhead, dropout=dropout)
        self.m3_transformer_encoder = nn.TransformerEncoder(self.m3_encoder_layer, num_layers=num_layers)
        self.m3_decoder = nn.Linear(x_frames,1)

        self.init_weights()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sigmoid = nn.Sigmoid()
        
        self.Multimodal_linear = nn.Linear(Transformer_feature_size*3,1)


    def init_weights(self):
        initrange = 0.1

        self.m1_input_linear.bias.data.zero_()
        self.m1_input_linear.weight.data.uniform_(-initrange, initrange)
        self.m2_input_linear.bias.data.zero_()
        self.m2_input_linear.weight.data.uniform_(-initrange, initrange)
        self.m3_input_linear.bias.data.zero_()
        self.m3_input_linear.weight.data.uniform_(-initrange, initrange)

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

        src1 = self.m1_input_linear(src1)
        src2 = self.m2_input_linear(src2)
        src3 = self.m3_input_linear(src3)
                

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
            
        encoder_output1 = self.m1_transformer_encoder(src1) #, self.src_mask) [64, 10,64]
        encoder_output2 = self.m2_transformer_encoder(src2) # [64, 10, 64]
        encoder_output3 = self.m3_transformer_encoder(src3) # [64, 10, 64]
  
        
        transformer_output1 = self.m1_decoder(encoder_output1.transpose(1,2)).squeeze()  # 64, 50
        transformer_output2 = self.m2_decoder(encoder_output2.transpose(1,2)).squeeze()
        transformer_output3 = self.m3_decoder(encoder_output3.transpose(1,2)).squeeze()
          

        concatted_output = torch.cat((transformer_output1,transformer_output2,transformer_output3),dim = 1)
        output = self.Multimodal_linear(concatted_output).squeeze()
        
        output = self.sigmoid(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask