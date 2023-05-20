import torch
from torch import nn
import yaml

from models.encoder import Encoder
from models.attention import Attention
from models.cnn import ResidualCNN


class Speech_recognition_model(nn.Module):
    """ Speech Recognition Model Inspired by DeepSpeech 2 and LAS """

    # def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, bidirectional=True):
    def __init__(self, CNN_params, ResCNN_params, Fully_connected_params, RNN_params, Attention_params, Classifier_params):
        super(Speech_recognition_model, self).__init__()

        # self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(**CNN_params) # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = self.__rescnn_layers_create(**ResCNN_params)
        self.fully_connected = nn.Linear(**Fully_connected_params)
        self.encoder = Encoder(**RNN_params)
        self.attention = Attention(**Attention_params)
        self.classifier = self.__classifier_create(**Classifier_params)

    def __classifier_create(self, in_features, out_features, dropout, n_class):
        return nn.Sequential(
            nn.Linear(in_features, out_features),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, n_class)
        )
    
    def __rescnn_layers_create(self, in_channels, out_channels, kernel_size, stride, dropout, n_feats, padding, n_cnn_layers):
        return nn.Sequential(*[
            ResidualCNN(in_channels, out_channels, kernel_size, stride, dropout, n_feats, padding) 
            for _ in range(n_cnn_layers)
        ])

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

    @staticmethod
    def serialize(model, optimizer, scheduler, epoch, tr_losses, cv_losses, tr_accuracy, cv_accuracy):
        package = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'tr_loss': tr_losses,
            'cv_loss': cv_losses,
            'tr_accuracy': tr_accuracy,
            'tr_accuracy': cv_accuracy,
            'epoch': epoch
        }
        return package
    