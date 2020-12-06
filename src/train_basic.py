import numpy as np
import sys
import time
from utils import read_dataset, read_categories
from argparse import Namespace
from basic_model import Model
from sklearn.metrics import f1_score, jaccard_score, multilabel_confusion_matrix,precision_recall_fscore_support

args = Namespace(
    train_dataset_path='data/2018-E-c-En-train.txt',
    validation_dataset_path='data/2018-E-c-En-dev.txt',
    test_dataset_path='data/2018-E-c-En-test.txt',
)

def evaluate(y, y_hat):
    return {
        "jaccard": jaccard_score(y, y_hat, average='samples'),
        "f1_macro": f1_score(y, y_hat, average='macro'),
        "f1_micro": f1_score(y, y_hat, average='micro'),
        "conf_matrix": np.round((multilabel_confusion_matrix(y,y_hat)/len(y_hat)),2),
        "measures": precision_recall_fscore_support(y,y_hat)
    }

def print_stats(categories,name, stats, eltime, output_file=sys.stdout):
    print("\n" + name, file=output_file)
    print("Elapsed Time: ",round(eltime,3),file=output_file)
    print("Jaccard: {}".format(round(stats['jaccard'],3)),file=output_file)
    print("F1 (macro): {}".format(round(stats['f1_macro'],3)),file=output_file)
    print("F1 (micro): {}".format(round(stats['f1_micro'],3)),file=output_file)
    print("Confusion matrix:", file=output_file)

    for x,y in zip(categories, stats['conf_matrix']):
        print(x,file=output_file)
        print(y.flatten(),file=output_file)
        
    print("Measures:",file=output_file)
    names = ['precision','recall','F-measure', 'support']
    for x,y in zip(stats['measures'],names):
        print(y,file=output_file)
        print(np.round(x,2), file=output_file)
    print("--------------------------------",file=output_file)


def main():

    X_train, y_train = read_dataset(args.train_dataset_path)
    X_val, y_val = read_dataset(args.validation_dataset_path)
    categories = read_categories(args.train_dataset_path)

    print(categories)
    print(X_val.shape)
    print(y_val.shape)

    X_train2 = Model.preprocess("Preprocess X_train", X_train, 'stem')
    X_val2 = Model.preprocess("Preprocess X_val", X_val, 'stem')

    ####################################################

    print("> Executing Logistic Regression")
    model = Model("lr")
    start = time.time()
    pred_val = model.fit_multilabel(X_train2, X_val2, y_train, categories)
    end = time.time()
    res = evaluate(y_val, pred_val)
    etime = end - start

    ###################################################

    print("> Executing Support Vector Classification")
    model = Model("svc")
    start = time.time()
    pred_val = model.fit_multilabel(X_train2, X_val2, y_train, categories)
    end = time.time()
    res2 = evaluate(y_val, pred_val)
    etime2 = end - start

    ###################################################

    print("> Executing Naive Bayes")
    model = Model("naive")
    start = time.time()
    pred_val = model.fit_multilabel(X_train2, X_val2, y_train, categories)
    end = time.time()
    res3 = evaluate(y_val, pred_val)
    etime3 = end - start

    ###################################################

    print("> Executing Perceptron")
    model = Model("perc")
    start = time.time()
    pred_val = model.fit_multilabel( X_train2, X_val2, y_train, categories)
    end = time.time()
    res4 = evaluate(y_val, pred_val)
    etime4 = end - start

    ###################################################

    print("> Executing Decision Tree Classifier")
    model = Model("dtc")
    start = time.time()
    pred_val = model.fit_multilabel(X_train2, X_val2, y_train, categories)
    end = time.time()
    res5 = evaluate(y_val, pred_val)
    etime5 = end - start
    ###################################################

    print("> Executing Random Forest Classifier")
    model = Model("rfc")
    start = time.time()
    pred_val = model.fit_multilabel(X_train2, X_val2, y_train, categories)
    end = time.time()
    res6 = evaluate(y_val, pred_val)
    etime6 = end - start
    ###################################################

    f = open("demofile.txt", "w")

    print_stats(categories,'Logistic Regression',res, etime,f)
    print_stats(categories,'Support Vector Classification',res2,etime2,f)
    print_stats(categories,'Naive Bayes',res3,etime3,f)
    print_stats(categories,'Perceptron',res4,etime4,f)
    print_stats(categories,'Decision Tree',res5,etime5,f)
    print_stats(categories,'Random Forest',res6,etime6,f)


if __name__ == "__main__":
    main()