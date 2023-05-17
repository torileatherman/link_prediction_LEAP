import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it trains on the cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="LEAP.")
    
    parser.add_argument('--transductive', type = bool, default= False)
    parser.add_argument('--device', type = str, default= 'cuda:0')
    parser.add_argument('--root', type = str, default= './data/pakdd2023/')
    parser.add_argument('--name', type = str, default= 'crocodile')
    parser.add_argument('--epochs', type = int, default = 200)


    return parser.parse_args()