import torch
import torch.nn as nn
import math
import os

from utils.positional_encoding import PositionalEncoding1d, PositionalEncoding2d

class Im2LaTeXModel(nn.Module):

    def __init__(self, encoder_out_size, embedding_dim, vocab_size, num_layers, num_heads, feedforward_dim, dropout, max_len, device, vocab_dict = {}):
        super(Im2LaTeXModel, self).__init__()

        self.encoder_out_size = encoder_out_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.max_len = max_len

        self.vocab_dict = vocab_dict
        self.invaling_set = set([self.vocab_dict["{"], self.vocab_dict["}"], \
                                self.vocab_dict["_"], self.vocab_dict["\!"], \
                                self.vocab_dict["~"], self.vocab_dict["\cdot"], \
                                self.vocab_dict["."]])




        # IMPORTANT: The encoder_out_size must be the same as the embedding_dim

        self.device = device
        self.max_len = max_len

        self.Encoder = Encoder(encoder_out_size)
        self.Decoder = Decoder(embedding_dim, vocab_size, num_layers, num_heads, feedforward_dim, dropout, max_len)

    def forward(self, image, formula):

        encoder_out = self.Encoder(image)
        # shape of encoder_out: (batch_size, height * width, encoder_out_size)

        output = self.Decoder(encoder_out, formula)
        # shape of output: (batch_size, formula_length, vocab_size)
        
        return output

    def predict(self, image):
            
        encoder_out = self.Encoder(image)
        # shape of encoder_out: (batch_size, height * width, encoder_out_size)
    
        outputs = torch.zeros((encoder_out.shape[0], self.max_len), dtype=torch.long, device=self.device)
        # shape of outputs: (batch_size, max_len)
        outputs[:, 0] = 1  # <SOS> token

        n = 15
        for i in range(1, self.max_len):
            # if the last n tokens are in the invalid set, fill the rest with { token
            if i > n:
                if all([token in self.invaling_set for token in outputs[:, i-n:i]]):
                    outputs[:, i:] = self.vocab_dict["{"]
                    break
            logits = self.Decoder(encoder_out, outputs[:, :i])
            
            # shape of logits: (i, batch_size, vocab_size)
            preds = torch.argmax(logits, dim=-1)
            # shape of preds: (i, batch_size)
            outputs[:, i] = preds[-1]

            # if all sentences are finished, break
            count = 0
            for j in range(encoder_out.shape[0]):
                if 2 in preds[:, j]:
                    count += 1
            if count == encoder_out.shape[0]:
                break
        
        # set all the tokens after <EOS> to 0
        for output in outputs:
            for i, token in enumerate(output):
                if token == 2:
                    output[i+1: ] = 0
                    break

        return outputs

class Encoder(nn.Module):

    def __init__(self, encoder_out_size):

        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1), 0),
            nn.Conv2d(512, encoder_out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(encoder_out_size),
            nn.ReLU()
        )

        self.positional_encoding = PositionalEncoding2d(encoder_out_size)

    def forward(self, image):

        # shape of image: (batch_size, channels, height, width)
        cnn_out = self.cnn(image)
        # shape of cnn_out: (batch_size, encoder_out_size, height, width)
        positional_encoding = self.positional_encoding(cnn_out)
        # shape of positional_encoding: (batch_size, encoder_out_size, height, width)

        # flatten the pixels
        positional_encoding = positional_encoding.permute(0, 2, 3, 1)
        # shape of positional_encoding: (batch_size, height, width, encoder_out_size)
        positional_encoding = positional_encoding.reshape(positional_encoding.shape[0], -1, positional_encoding.shape[-1])
        # shape of positional_encoding: (batch_size, height * width, encoder_out_size)
        return positional_encoding

class Decoder(nn.Module):
    
    def __init__(self, embedding_dim, vocab_size, num_layers, num_heads, feedforward_dim, dropout, max_len):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding1d(embedding_dim)

        self.attention_mask = nn.Transformer().generate_square_subsequent_mask(max_len)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, num_heads, feedforward_dim, dropout),
            num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, encoder_out, formula):

        # shape of encoder_out: (batch_size, height * width, encoder_out_size)
        # shape of formula: (batch_size, formula_len)

        embedding = self.embedding(formula)
        embedding *= math.sqrt(embedding.shape[-1])
        # shape of embedding: (batch_size, formula_len, embedding_dim)

        positional_encoding = self.positional_encoding(embedding)
        # shape of positional_encoding: (batch_size, formula_len, embedding_dim)
        positional_encoding = positional_encoding.permute(1, 0, 2)
        # shape of positional_encoding: (formula_len, batch_size, embedding_dim)

        attention_mask = self.attention_mask[:positional_encoding.shape[0], :positional_encoding.shape[0]].type_as(encoder_out)
        # shape of attention_mask: (formula_len, formula_len)

        encoder_out = encoder_out.permute(1, 0, 2)
        # shape of encoder_out: (height * width, batch_size, encoder_out_size)
        
        output = self.decoder(positional_encoding, encoder_out, tgt_mask=attention_mask)
        # shape of output: (formula_len, batch_size, embedding_dim)

        output = self.fc(output)
        # shape of output: (formula_len, batch_size, vocab_size)

        return output

# save the model
def save_model(model, vocab_dict, file_name, path):

    model = {

        'encoder_out_size': model.encoder_out_size,
        'embedding_dim': model.embedding_dim,
        'vocab_size': model.vocab_size,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'feedforward_dim': model.feedforward_dim,
        'dropout': model.dropout,
        'max_len': model.max_len,

        'model': model.state_dict(),
        'vocab_dict': vocab_dict
    }
    torch.save(model, os.path.join(path, file_name))
    print('Model saved to {}'.format(os.path.join(path, file_name)))

# it would return the model and the vocab dictionary
def load_model(location, device):

    saved_model = torch.load(location)
    model = Im2LaTeXModel(saved_model['encoder_out_size'], 
                            saved_model['embedding_dim'], 
                            saved_model['vocab_size'], 
                            saved_model['num_layers'], 
                            saved_model['num_heads'], 
                            saved_model['feedforward_dim'], 
                            saved_model['dropout'], 
                            saved_model['max_len'],
                            device).to(device)

    print('Model loaded from {}'.format(location))
    
    print('Hyperparameters:')
    print('encoder_out_size: {}'.format(saved_model['encoder_out_size']))
    print('embedding_dim: {}'.format(saved_model['embedding_dim']))
    print('vocab_size: {}'.format(saved_model['vocab_size']))
    print('num_layers: {}'.format(saved_model['num_layers']))
    print('num_heads: {}'.format(saved_model['num_heads']))
    print('feedforward_dim: {}'.format(saved_model['feedforward_dim']))
    print('dropout: {}'.format(saved_model['dropout']))
    print('max_len: {}'.format(saved_model['max_len']))
    
    model.load_state_dict(saved_model['model'])
    model.eval()

    return model, saved_model['vocab_dict']
    