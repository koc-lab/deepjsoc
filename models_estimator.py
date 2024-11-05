import torch
import torch.nn as nn

class BI_Estimator(nn.Module):
    def __init__(self, input_size,actual_size, d_rnn=512, d_mlp=[128, 32], num_bi_layers=3, rnn_type='gru'):

        super(BI_Estimator, self).__init__()

        # Parameters init
        self.actual_size = actual_size
        self.num_bi_layers = num_bi_layers
        self.d_rnn = d_rnn
        self.rnn_type = rnn_type
        self.d_mlp = d_mlp

        # BI-RNN layers
        if self.rnn_type == 'lstm':
            self.bir_layers = nn.ModuleList([nn.LSTM(input_size if i == 0 else self.d_rnn * 2, self.d_rnn, bidirectional=True, batch_first=True)
                                             for i in range(self.num_bi_layers)])
        elif self.rnn_type == 'gru':
            self.bir_layers = nn.ModuleList([nn.GRU(input_size if i == 0 else self.d_rnn * 2, self.d_rnn, bidirectional=True, batch_first=True)
                                             for i in range(self.num_bi_layers)])
        else:
            raise ValueError('RNN type must be either gru or lstm!')

        # Layer normalization layers
        self.nor_layers = nn.ModuleList([nn.LayerNorm(self.d_rnn * 2,eps=1e-3) for _ in range(self.num_bi_layers)])

        # MLP layers
        mlp_layers = []
        input_size = self.d_rnn * 2  # Bidirectional output size
        for size in self.d_mlp:
            mlp_layers.append(nn.Linear(input_size, size))
            mlp_layers.append(nn.ReLU())
            input_size = size
        self.mlp_layers = nn.Sequential(*mlp_layers)

        # Output layer
        self.output_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward Pass of the specified model!
        """
        
        for bir_layer, nor_layer in zip(self.bir_layers, self.nor_layers):
            #print(x.shape)
            
            x, _ = bir_layer(x)
            #print(x.shape)
            x = nor_layer(x)
        #print(x.shape)
        x = self.mlp_layers(x)
        #print(x.shape)
      
        x = self.output_layer(x)
        #print(x.shape)
        x = self.sigmoid(x)
        
        return x[:,0:self.actual_size,:]
