import torch
import torch.optim as optim
import torch.nn as nn
from argparse import Namespace
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import time
import numpy as np

from vocab import Vocabulary
from embeddings import load_embeddings
from twitter_emotions_dataset import TwitterEmotionsDataset
from lstm_model import LstmWithClassifier
from utils import checkpoint, get_dataloader, read_categories
from train_classic import evaluate, print_stats


args = Namespace(
    disable_cuda=False,
    # Data and path information
    pretrained_embeddings_matrix_path='embeddings.npy',
    vocab_path='vocabulary.json',
    train_dataset_path='data/2018-E-c-En-train.txt',
    validation_dataset_path='data/2018-E-c-En-dev.txt',
    test_dataset_path='data/2018-E-c-En-test.txt',
    checkpoint_path='cp.pth',
    # Model hyperparameters
    lstm_hidden_size=256,
    lstm_bidirectional=True,
    # Train parameters
    batch_size=128,
    early_stopping_criteria=3,
    learning_rate=0.001,
    num_epochs=20
)


def make_train_state(args):
    return {
        'epoch_idx': 0,
        'train_loss': [],
        'train_metrics': [],
        'val_loss': [],
        'val_metrics': [],
        'best_loss': float('inf'),
        'best_epoch': 0,
        'early_stop_tracker': 0,
        'stats': {},
        'start_time': 0,
        'best_epoch_time': 0
    }


def main():
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_state = make_train_state(args)

    train_dataset = TwitterEmotionsDataset.from_raw_csv(args.train_dataset_path)
    validation_dataset = TwitterEmotionsDataset.from_raw_csv(args.validation_dataset_path)

    embedding_weights = load_embeddings('embeddings.npy')
    vocab = Vocabulary.load('vocabulary.json')

    classifier = LstmWithClassifier(vocab, embedding_weights, bilstm=args.lstm_bidirectional, hidden_size=args.lstm_hidden_size)
    classifier = classifier.to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    early_stop = False

    train_state['start_time'] = time.time()


    for epoch_idx in range(args.num_epochs):
        if early_stop:
            break

        train_state['epoch_idx'] = epoch_idx

        ###
        # Train
        ###
        running_loss = 0.0
        running_jaccard_score = 0.0
        classifier.train()

        train_dataloader = get_dataloader(train_dataset, args.batch_size)

        print("\n --- Epoch {} ---\n".format(epoch_idx))

        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, desc="Train")):
            optimizer.zero_grad()

            y = batch_data['labels'].to(device)
            y_hat = classifier(batch_data['tweet'])

            loss = criterion(y_hat, y)
            loss.backward()

            running_loss += loss.item()
            optimizer.step()

            prediction = (torch.sigmoid(y_hat).detach() >= 0.5).float()
            running_jaccard_score += jaccard_score(y.cpu(), prediction.cpu(), average='samples') * len(batch_data['tweet'])


        epoch_loss = running_loss / len(train_dataloader)
        epoch_jaccard_score = running_jaccard_score / len(train_dataset)

        train_state['train_loss'].append(epoch_loss)
        print("Loss: {}".format(train_state['train_loss'][epoch_idx]))
        train_state['train_metrics'].append(epoch_jaccard_score)
        print("Jaccard score: {}".format(train_state['train_metrics'][epoch_idx]))
        print("")

        ###
        # Validation
        ###
        validation_dataloader = get_dataloader(validation_dataset, args.batch_size, shuffle=False)

        running_loss = 0.0
        running_jaccard_score = 0.0
        classifier.eval()
        y_total = np.zeros((len(validation_dataset), 11))

        for batch_idx, batch_data in enumerate(tqdm(validation_dataloader, desc="Validation")):
            y = batch_data['labels'].to(device)
            y_hat = classifier(batch_data['tweet'])

            loss = criterion(y_hat, y)
            running_loss += loss.item()

            prediction = (torch.sigmoid(y_hat).detach() >= 0.5).float()
            running_jaccard_score += jaccard_score(y.cpu(), prediction.cpu(), average='samples') * len(batch_data['tweet'])

            base_idx = batch_idx * args.batch_size
            y_total[base_idx:base_idx + len(batch_data['tweet']), :] = prediction.cpu().numpy()


        epoch_loss = running_loss / len(validation_dataloader)
        epoch_jaccard_score = running_jaccard_score / len(validation_dataset)

        train_state['val_loss'].append(epoch_loss)
        print("Loss: {}".format(train_state['val_loss'][epoch_idx]))
        train_state['val_metrics'].append(epoch_jaccard_score)
        print("Jaccard score: {}".format(train_state['val_metrics'][epoch_idx]))

        if epoch_loss < train_state['best_loss']:
            train_state['best_loss'] = epoch_loss
            train_state['best_epoch'] = epoch_idx
            train_state['early_stop_tracker'] = 0
            train_state['stats'] = evaluate(validation_dataset.labels, y_total)
            train_state['best_epoch_time'] = time.time()
            cp_path = checkpoint(args.checkpoint_path, classifier, optimizer, train_state)
            print("Saved checkpoint to {}".format(cp_path))
        else:
            train_state['early_stop_tracker'] += 1
            print("Early stop counter: {}/{}".format(train_state['early_stop_tracker'], args.early_stopping_criteria))
            if train_state['early_stop_tracker'] == args.early_stopping_criteria:
                early_stop = True

    f = open("stats.txt", "w")
    print_stats(read_categories(args.train_dataset_path),'DL', train_state['stats'], train_state['best_epoch_time'] - train_state['start_time'], f)


if __name__ == "__main__":
    main()