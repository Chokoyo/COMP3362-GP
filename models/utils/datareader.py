from torchvision import transforms
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
import random
from torch.utils.data import Sampler
import numpy as np

from . import tokenize

class DataReader():

    def __init__(self, data_root_dir, image_dir, train_ann_name, dev_ann_name, test_ann_name):

        self.data_root_dir = data_root_dir
        self.image_dir = data_root_dir + image_dir
        self.train_ann_path = data_root_dir + train_ann_name
        self.dev_ann_name = data_root_dir + dev_ann_name
        self.test_ann_name = data_root_dir + test_ann_name

    def read_data(self, train_data_size: int, dev_data_size: int, test_data_size: int):

        train_ann_df = pd.read_csv(self.train_ann_path)
        dev_ann_df = pd.read_csv(self.dev_ann_name)
        test_ann_df = pd.read_csv(self.test_ann_name)

        # remove NaN from the data frame
        train_ann_df = train_ann_df.dropna()
        dev_ann_df = dev_ann_df.dropna()
        test_ann_df = test_ann_df.dropna()

        train_df = train_ann_df.head(train_data_size)
        dev_df = dev_ann_df.head(dev_data_size)
        test_df = test_ann_df.head(test_data_size)

        train_image_tensors = [self.read_image(image_name) for image_name in train_df['image'].to_numpy()]
        dev_image_tensors = [self.read_image(image_name) for image_name in dev_df['image'].to_numpy()]
        test_image_tensors = [self.read_image(image_name) for image_name in test_df['image'].to_numpy()]

        train_formulas = train_df['formula']
        dev_formulas = dev_df['formula']
        test_formulas = test_df['formula']

        return (train_image_tensors, train_formulas), (dev_image_tensors, dev_formulas), (test_image_tensors, test_formulas)

    def read_image(self, image_name) -> torch.Tensor:

        transform = transforms.ToTensor()
        image = Image.open(self.image_dir + image_name).convert('L')
        image_tensor = transform(image)
        return image_tensor

    def build_vocab(self, formulas: pd.Series, num_of_prune = 5) -> dict:

        token_dict_processor = tokenize.TokenDict()
        for formula in formulas.to_numpy():
            token_dict_processor.account(formula)
        token_dict_processor.prune(min_count = num_of_prune)

        self.vocab_dict = token_dict_processor.dict
        
        return token_dict_processor.dict
    
    def tokenize_formulas(self, formulas: pd.Series, token_dict: dict) -> list:

        tokenizer = tokenize.Tokenizer(token_dict)
        formulas_indexed = tokenizer.tokenize_idx(formulas.to_numpy())

        return list(formulas_indexed)

    def build_dataset(self, image_tensors: list, formulas_indexed: list) -> Dataset:

        dataset = Im2LatexDataset(image_tensors, formulas_indexed)
        
        return dataset

    # This function is used to pad the formulas and images to the same shape
    def collate_fn(self, batch):

        images, formulas = zip(*batch)

        batch_size = len(images)
        max_H = max([image.shape[1] for image in images])
        max_W = max([image.shape[2] for image in images])
        max_length = max([len(formula) for formula in formulas])

        padded_images = torch.zeros(batch_size, 1, max_H, max_W)
        batched_indices = torch.zeros((batch_size, max_length + 2), dtype=torch.long)

        for i in range(batch_size):

            H, W = images[i].shape[1], images[i].shape[2]
            y, x = random.randint(0, max_H - H), random.randint(0, max_W - W)
            padded_images[i, :, y:y+H, x:x+W] = images[i]

            batched_indices[i, 0] = self.vocab_dict['<SOS>']
            batched_indices[i, 1: len(formulas[i]) + 1] = torch.tensor(formulas[i])
            batched_indices[i, len(formulas[i]) + 1] = self.vocab_dict['<EOS>']

        return padded_images, batched_indices

class Im2LatexDataset(Dataset):
    
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

# ref: https://gist.github.com/TrentBrick/bac21af244e7c772dc8651ab9c58328c
class BinningSampler(Sampler):
    def __init__(self, data_source, bucket_boundary, batch_size):
        idx_seq_len = [(idx, len(formula)) for idx, (image, formula) in enumerate(data_source)]
        self.data_source = data_source
        self.idx_seq_len = idx_seq_len
        self.bucket_boundary = bucket_boundary
        self.batch_size = batch_size
        self.num_batch = 0
    
    def __iter__(self):
        buckets = dict()
        for idx, seq_len in self.idx_seq_len:
            for boundary in self.bucket_boundary:
                if seq_len <= boundary:
                    if boundary not in buckets:
                        buckets[boundary] = []
                    buckets[boundary].append(idx)
                    break

        for boundary in buckets:
            buckets[boundary] = np.array(buckets[boundary])

        batches = []
        for boundary in self.bucket_boundary:
            np.random.shuffle(buckets[boundary])
            batches += (np.array_split(buckets[boundary], int(len(buckets[boundary]) / self.batch_size) + 1))
        np.random.shuffle(batches)

        self.num_batch = len(batches)

        for batch in batches:
            yield batch.tolist()

    def __len__(self):
        return self.num_batch