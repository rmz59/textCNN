import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field


class TextCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, output_channel_num, kernel_widths, dropout_rate=0.5, from_pretrained=None):
        super(TextCNN, self).__init__()
        self.embed_num, self.embed_dim, self.class_num = embed_num, embed_dim, class_num
        self.output_channel_num = output_channel_num
        self.kernel_widths = kernel_widths 
        self.dropout_rate = dropout_rate

        if from_pretrained is not None:
            self.embedding = nn.Embedding(embed_num, embed_dim).from_pretrained(from_pretrained)
        else:
            self.embedding = nn.Embedding(embed_num, embed_dim)

        self.conv_layers = nn.ModuleList([nn.Conv1d(self.embed_dim, 
                                                    self.output_channel_num, 
                                                    kernel_size=kernel_width) 
                                         for kernel_width in self.kernel_widths])
        self.drop_out = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(sum(self.output_channel_num for k in self.kernel_widths), class_num)

    def forward(self, input_sentences):
        """ 
        Args:
            input_sentences: torch.tensor -> shape: (batch_size, max_sent_length)
        """
        x = self.embedding(input_sentences).transpose(-1, -2) # X shape: (batch_size, embed_size, max_sent_length)
        x = torch.cat([self.conv_pool(conv_layer, x) for conv_layer in self.conv_layers], dim=-1) # (batch_size, output_channels_sum)
        x = self.fc(self.drop_out(x))
        return x
    
    @staticmethod
    def conv_pool(conv_layer, x):
        """
        Args:
            conv_layer: nn.Conv1D
            x: input tensor (batch_size, embed_size, max_length)
        Example:
        >>> batch_size, input_channel, output_channel, kernel_size = 10, 5,3,2
        >>> input_sents = torch.rand((batch_size, input_channel, 100))
        >>> conv_layer = nn.Conv1d(input_channel, output_channel, kernel_size)
        >>> TextCNN.conv_pool(conv_layer, input_sents).shape
        torch.Size([10, 3])
        """
        x = F.relu(conv_layer(x))  # (batch_size, output_channel, max_length-kernel_size+1)
        x = F.max_pool1d(x, x.shape[-1]).squeeze(-1) # (batch_size, output_channel)
        return x 
    

    @classmethod
    def load(cls, model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = cls(args['embed_num'], args['embed_dim'], args['class_num'], args['output_channel_num'], args['kernel_widths'])
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        logging.info(f'save model parameters to [{path}]')

        params = {
            'args': dict(embed_num=self.embed_num, 
                         embed_dim=self.embed_dim, 
                         class_num = self.class_num,
                         output_channel_num = self.output_channel_num,
                         kernel_widths = self.kernel_widths,
                         dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
