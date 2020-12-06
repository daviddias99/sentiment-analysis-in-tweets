import argparse
import numpy as np
import torch
from tqdm import tqdm

from vocab import Vocabulary


def load_embeddings_from_path(path):
    word2id = {}
    embeddings = []

    print('Loading embeddings from %s ...' % path)

    with open(path, "r",encoding="utf8") as f:
        first_line = f.readline()
        second_line = f.readline()
        f.seek(0)

        if len(first_line.split(" ")) != len(second_line.split(" ")):
            next(f)

        for idx, line in enumerate(tqdm(f)):
            line = line.split(" ")
            word2id[line[0]] = idx
            embedding = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding)

    print('Loaded %d embeddings of size %d' % (len(word2id), len(embeddings[0])))

    return word2id, np.stack(embeddings)

def gen_embeddings_matrix(vocab, embeddings, embs_word2id):
    print('\nGenerating embeddings matrix ...')

    embedding_size = embeddings.shape[1]

    tokens_without_embeddings = 0
    tokens_with_embeddings = 0

    embeddings_matrix = np.zeros((len(vocab), embedding_size))

    for word, id in tqdm(vocab.word2id.items()):
        if word in embs_word2id:
            embeddings_matrix[id, :] = embeddings[embs_word2id[word]]
            tokens_with_embeddings += 1
        else:
            print('Vocabulary token without embedding: %s' % word)
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            embeddings_matrix[id, :] = embedding_i
            tokens_without_embeddings += 1

    print('%d vocab tokens without embedding' % tokens_without_embeddings)
    print('%d vocab tokens with embedding' % tokens_with_embeddings)

    print('Embeddings matrix size (%d, %d)' % (embeddings_matrix.shape[0], embeddings_matrix.shape[1]))
    return embeddings_matrix

def save_embeddings(path, embeddings_matrix):
    np.save(path, embeddings_matrix)

def load_embeddings(path):
    return np.load(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings-path', help='Path to file containing pretrained embeddings',
                        dest='embeddings_path', required=True, type=str)
    parser.add_argument('--save-path', help='Where to save generated embeddings matrix',
                        dest='save_path', default='embeddings.npy', type=str)
    parser.add_argument('--vocab-path', help='Path to vocabulary JSON file',
                    dest='vocab_path', default='vocabulary.json', type=str)
    return parser.parse_args()

def main():
    args = parse_args()

    print('Loading vocabulary from %s ...' % args.vocab_path)
    vocab = Vocabulary.load(args.vocab_path)
    print(vocab)

    embs_word2id, embeddings = load_embeddings_from_path(args.embeddings_path)

    matrix = gen_embeddings_matrix(vocab, embeddings, embs_word2id)

    save_embeddings(args.save_path, matrix)
    print('Saved embeddings to %s' % args.save_path)

    n_matrix = load_embeddings(args.save_path)
    print(n_matrix.shape)

if __name__ == "__main__":
    main()

