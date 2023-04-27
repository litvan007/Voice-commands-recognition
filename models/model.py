import torch
from torch import nn
import yaml

from models.encoder import Encoder
from models.attention import Attention
from models.cnn import ResidualCNN


class Speech_recognition_model(nn.Module):
    """ Speech Recognition Model Inspired by DeepSpeech 2 and LAS """

    # def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, bidirectional=True):
    def __init__(self, CNN_params, ResCNN_params, RNN_params, Attention_params, Classifier_params, Others_params):
        super(Speech_recognition_model, self).__init__()
        n_feats = Others_params['n_feats']
        n_cnn_layers = Others_params['n_cnn_layers']
        n_fully_connected_out = Others_params['n_fully_connected_out']
        n_class = Classifier_params['n_class']

        # self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(**CNN_params) # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(**ResCNN_params) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, n_fully_connected_out)
        self.encoder = Encoder(**RNN_params)

        self.attention = Attention(**Attention_params)
        
        self.classifier = nn.Sequential(
            nn.Linear(Classifier_params['in_channels_1'], Classifier_params['out_channels_1']),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(Classifier_params['dropout']),
            nn.Linear(Classifier_params['out_channels_1'], n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x.transpose(2, 3))

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)

        x, hidden = self.encoder(x)
        att_input = hidden[-1]
        att_output = self.attention(att_input.transpose(0, 1))
        
        output = self.classifier(att_output)
        return output

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @staticmethod
    def serialize(model, optimizer, scheduler, epoch, tr_loss=None, cv_loss=None):
        package = {
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package
    