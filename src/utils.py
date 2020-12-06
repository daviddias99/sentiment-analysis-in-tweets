import pandas as pd
import torch
from torch.utils.data import DataLoader

def read_dataset(path):
    df = pd.read_csv(path, sep='\t')
    tweets = df['Tweet'].to_numpy()
    labels = df.iloc[:, 2:].to_numpy()
    return tweets, labels

def read_categories(path):
    df = pd.read_csv(path, sep='\t')
    return df.columns.values[2:]

def pad_sentences(sents, pad_token):
    sents_padded = []

    max_len = max([len(sent) for sent in sents])
    for sent in sents:
        sent_len = len(sent)
        sents_padded.append(sent + [pad_token] * (max_len - sent_len))

    return sents_padded


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_fn
    )

def collate_fn(batch):
    tweets = [instance['tweet'] for instance in batch]
    if batch[0]['labels'] is None:
        labels = None
    else:
        labels = torch.Tensor([instance['labels'] for instance in batch])

    return {
        'tweet': tweets,
        'labels': labels
    }

def checkpoint(path, model, optimizer, train_state):

    torch.save({
        'train_state': train_state,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)

    return path

def load_checkpoint(path, device, model, optimizer=None):
    cp = torch.load(path, map_location=device)

    model.load_state_dict(cp['model'])
    model.eval() # just to be safe

    if optimizer is not None:
        optimizer.load_state_dict(cp['optimizer'])

    return cp['train_state']
