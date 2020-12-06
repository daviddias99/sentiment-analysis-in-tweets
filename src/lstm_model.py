import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import List

from vocab import Vocabulary

class LstmWithClassifier(nn.Module):
    def __init__(self,
                 vocab,
                 embedding_weights,
                 hidden_size=None,
                 bilstm=True):
        super(LstmWithClassifier, self).__init__()
        self.vocab = vocab
        self.emb_size = embedding_weights.shape[1]
        self.hidden_size = self.emb_size if hidden_size is None else hidden_size
        self.bidirectional = bilstm

        self.embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_weights),
            freeze=True,
            padding_idx=vocab['<pad>']
        )

        self.encoder = nn.LSTM(
            input_size=self.emb_size,
            hidden_size=self.hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.classifier = nn.Linear(
            in_features=self.hidden_size * (2 if self.bidirectional else 1),
            out_features=11
        )


    def forward(self, tweets: List[List[str]]):
        batch_size = len(tweets)
        tweet_lengths = [len(tweet) for tweet in tweets]
        tweet_ids = self.vocab.to_input_tensor(tweets, next(self.parameters()).device)

        # shape: (batch size, max seq length, embedding size)
        embedded_tweets = self.embed(tweet_ids)

        packed_seqs = pack_padded_sequence(embedded_tweets, tweet_lengths, batch_first=True, enforce_sorted=False)

        # last hidden/cell shape: (num directions, batch size, hidden size)
        output_packed, (last_hidden, last_cell) = self.encoder(packed_seqs)

        # shape: (batch size, max seq length, num directions x hidden size)
        output, _ = pad_packed_sequence(output_packed, batch_first=True)

        # shape: (batch size, num directions x hidden size)
        rep = torch.cat((last_hidden[0], last_hidden[1]), -1) if self.bidirectional else last_hidden[0]

        return self.classifier(rep)
