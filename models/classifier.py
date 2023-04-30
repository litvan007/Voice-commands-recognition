import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=True, rnn_type='lstm'):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout

        self.dense = nn.Sequential(
            nn.Linear(Classifier_params['in_channels_1'], Classifier_params['out_channels_1']),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(Classifier_params['dropout']),
            nn.Linear(Classifier_params['out_channels_1'], n_class)
        )

    def forward(self, input):
        return self.dense(input)
