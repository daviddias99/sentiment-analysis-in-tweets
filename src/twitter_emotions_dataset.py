import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import read_dataset
from preprocess import preprocess


class TwitterEmotionsDataset(Dataset):
    def __init__(self, path, tweets, labels=None):
        self.tweets = tweets
        self.labels = labels if labels is not None else torch.zeros((len(tweets), 11))
        self._test = labels is None
        self._size = len(self.tweets)
        self._path = path

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return {
            'tweet': self.tweets[index],
            'labels': self.labels[index] if not self.is_test() else None
        }

    def is_test(self):
        return self._test

    @classmethod
    def from_raw_csv(cls, path):
        raw_tweets, labels = read_dataset(path)

        if labels[0, 0] == 'NONE':
            labels = None

        tokenized_tweets = preprocess(path, raw_tweets)

        return cls(path, tokenized_tweets, labels)

    def save(self, path):
        """ Add labels to test dataset and save it as csv file
        """
        df = pd.read_csv(self._path, sep='\t')
        df.iloc[:, 2:] = self.labels.numpy().astype(int)
        df.to_csv(path, sep='\t', index=False)

