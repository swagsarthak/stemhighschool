import argparse
import sys

# Add the source directory to the Python path
sys.path.append('src')

from src.train import train
from src.evaluate import run_evaluation

def main():
    """
    Main function to run the training and evaluation of the ScienceQA model.
    """
    parser = argparse.ArgumentParser(description="Train or evaluate the ScienceQA model.")
    
    # Create a subparser to handle 'train' and 'evaluate' commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Training command
    parser_train = subparsers.add_parser('train', help='Train a new model')
    
    # Evaluation command
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate the best model on the test set')

    args = parser.parse_args()

    if args.command == 'train':
        train()
    elif args.command == 'evaluate':
        run_evaluation()

if __name__ == '__main__':
    main()