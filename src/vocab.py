import argparse
import json
import pandas as pd
import torch
from collections import Counter
from itertools import chain

from utils import read_dataset, pad_sentences
from preprocess import preprocess


class Vocabulary(object):
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<oov>'] = 1
        self.oov_id = self.word2id['<oov>']
        self.id2word = { v: k for k, v in self.word2id.items() }

    def __getitem__(self, word):
        return self.word2id.get(word, self.oov_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size={}]'.format(len(self))

    def id2word(self, id):
        return self.id2word[id]

    def add(self, word):
        if word not in self:
            id = self.word2id[word] = len(self)
            self.id2word[id] = word
            return id
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]


    def to_input_tensor(self, sents, device):
        word_ids = self.words2indices(sents)
        word_ids_padded = pad_sentences(word_ids, self['<pad>'])

        return torch.tensor(word_ids_padded, dtype=torch.long, device=device)


    @staticmethod
    def build(corpus, freq_cutoff=1):
        vocab = Vocabulary()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]

        print('Number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))

        for word in valid_words:
            vocab.add(word)

        return vocab

    @staticmethod
    def load(file_path):
        content = json.load(open(file_path, 'r'))

        return Vocabulary(content)


    def save(self, file_path):
        json.dump(self.word2id, open(file_path, 'w'), indent=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', help='Path to file containing dataset for which to generate Vocabulary',
                        dest='dataset_path', required=True, type=str)
    parser.add_argument('--save-path', help='Where to save generated Vocabulary',
                        dest='save_path', default='vocabulary.json', type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    tweets, labels = read_dataset(args.dataset_path)
    tokenized_tweets = preprocess(args.dataset_path, tweets)
    print(tokenized_tweets)
    vocab = Vocabulary.build(tokenized_tweets)
    print('Generated vocabulary with %d tokens' % (len(vocab)))

    vocab.save(args.save_path)
    print('Vocabulary saved to %s' % args.save_path)

if __name__ == '__main__':
    main()

