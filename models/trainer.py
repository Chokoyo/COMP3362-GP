import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Trainer():

    def __init__(self, model, device):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=2, verbose=False)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) # vocab_dict['<PAD>'] = 0

        # performance record
        self.train_losses = []
        self.test_losses = []
        self.train_perplexities = []
        self.test_perplexities = []

    # function for train one epoch
    def train(self, train_dataloader, epoch_idx):
        self.model.train()

        total_loss = 0
        
        for batch_idx, (image, formula) in enumerate(train_dataloader):
            
            image = image.to(self.device)
            formula = formula.to(self.device)

            self.optimizer.zero_grad()
            
            output = self.model(image, formula[:, :-1])
            # shape of output: (formula_length, batch_size, vocab_size)

            loss = self.criterion(output.permute(1, 2, 0), formula[:, 1:])

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_idx, batch_idx * len(image), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

        avg_loss = total_loss / len(train_dataloader)

        print('Train Epoch: {} Average loss: {:.6f} Perplexity: {:.6f}'.format(
            epoch_idx, avg_loss, np.exp(avg_loss)))
        
        self.lr_scheduler.step(avg_loss)

        self.train_losses.append(avg_loss)
        self.train_perplexities.append(np.exp(avg_loss))

        return avg_loss, np.exp(avg_loss)

    # test the model on the given test dataset
    def test(self, test_dataloader):
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (image, formula) in enumerate(test_dataloader):
                
                image = image.to(self.device)
                formula = formula.to(self.device)

                output = self.model(image, formula[:, :-1])
                # shape of output: (formula_length, batch_size, vocab_size)

                loss = self.criterion(output.permute(1, 2, 0), formula[:, 1:])

                test_loss += loss.item()

        avg_loss = test_loss / len(test_dataloader)

        print('Test Average loss: {:.6f} Perplexity: {:.6f}'.format(
            avg_loss, np.exp(avg_loss)))

        self.test_losses.append(avg_loss)
        self.test_perplexities.append(np.exp(avg_loss))

        return avg_loss, np.exp(avg_loss)

    def plot_loss(self):
        plt.plot(self.train_losses, label='train')
        plt.plot(self.test_losses, label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_perplexity(self):
        plt.plot(self.train_perplexities, label='train')
        plt.plot(self.test_perplexities, label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.show()
    
    def save_checkpoint(self, checkpoint_dir, epoch_idx, loss):
        checkpoint = {
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(epoch_idx)))
        print('Checkpoint saved at {}'.format(os.path.join(checkpoint_dir, 'checkpoint_{}.pth'.format(epoch_idx))))
    
    def load_checkpoint(self, checkpoint_location):
        checkpoint = torch.load(checkpoint_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Checkpoint loaded from {} at epoch {} with loss {}'.format(checkpoint_location, epoch, loss))
        return epoch, loss
        