from typing import List
import pandas as pd
import numpy as np 
import torch

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils import load_dataset

class MIDIDataset(Dataset):
    def __init__(self, dataset_path: str, max_len: str = 256, num_classes = 512):
        self.max_len = max_len
        self.num_classes = num_classes
        self.org_dataset = load_dataset(dataset_path)
        self.split_dataset = self._split_dataset(self.org_dataset)
        
    def __len__(self):
        return len(self.split_dataset)
    
    def _split_dataset(self, dataset:pd.DataFrame) -> List:
        """Splitting the dataset based on sequence length.

        Args:
            dataset (pd.DataFrame): Dataset to split

        Returns:
            List: splited dataset
        """
        new_dataset = []
        for data in dataset['data']:
            data_len = len(data)
            for i in range(0, data_len, self.max_len):
                end_idx = i + self.max_len
                if i + self.max_len >= data_len:
                    end_idx = data_len
                new_data = data[i:end_idx] 
                if len(new_data) < self.max_len:
                    new_data = new_data + [0] * (self.max_len - len(new_data))
                new_dataset.append(new_data)
       
        return new_dataset
    
    def __getitem__(self, idx):
        return F.one_hot(torch.tensor(self.split_dataset[idx], dtype=torch.long), num_classes=self.num_classes).float()
    
    
def load_dataloader(conf: DictConfig, dataset_path: str, num_classes: str) -> DataLoader:
    dataset = MIDIDataset(dataset_path=dataset_path,
                          max_len=conf.max_len, 
                          num_classes=2**num_classes)
    return DataLoader(dataset, 
                      batch_size=conf.batch_size,
                      num_workers = conf.num_workers,
                      pin_memory = True,
                      shuffle = conf.shuffle
                      )