import torch
from argparse import Namespace
from utils import load_checkpoint, get_dataloader
from twitter_emotions_dataset import TwitterEmotionsDataset
from lstm_model import LstmWithClassifier
from tqdm import tqdm
from vocab import Vocabulary
from embeddings import load_embeddings

args = Namespace(
    disable_cuda=False,
    # Data and path information
    pretrained_embeddings_matrix_path='embeddings.npy',
    vocab_path='vocabulary.json',
    checkpoint_path='cp.pth',
    test_dataset_path='data/2018-E-c-En-test.txt',
    predictions_save_path='E-C_en_pred.txt',
    # Model hyperparameters
    lstm_hidden_size=256,
    lstm_bidirectional=True,
    # Train parameters
    batch_size=128
)

def main():
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    embedding_weights = load_embeddings('embeddings.npy')
    vocab = Vocabulary.load('vocabulary.json')

    classifier = LstmWithClassifier(vocab, embedding_weights, bilstm=args.lstm_bidirectional, hidden_size=args.lstm_hidden_size)

    test_dataset = TwitterEmotionsDataset.from_raw_csv(args.test_dataset_path)

    train_info = load_checkpoint(args.checkpoint_path, device, classifier)
    print(train_info)

    ###
    # Test
    ###
    test_dataloader = get_dataloader(test_dataset, args.batch_size, shuffle=False)

    classifier.eval()
    for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Predicting on test set")):
        y_hat = classifier(batch_data['tweet'])
        prediction = (torch.sigmoid(y_hat).detach() >= 0.5).int()

        first_idx = batch_idx * args.batch_size
        last_idx = first_idx + len(batch_data['tweet'])
        test_dataset.labels[first_idx:last_idx] = prediction

    test_dataset.save(args.predictions_save_path)


if __name__ == "__main__":
    main()