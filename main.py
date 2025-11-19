import argparse

from train import train
from evaluate import run_evaluation
from inference import answer_question

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

    # Inference command
    parser_pred = subparsers.add_parser('predict', help='Run inference on a custom question')
    parser_pred.add_argument('--question', required=True, help='Question to answer')
    parser_pred.add_argument(
        '--choices',
        nargs='+',
        required=True,
        help='List of answer options (wrap each option in quotes)',
    )
    parser_pred.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='Optional path to a related image for the question',
    )

    args = parser.parse_args()

    if args.command == 'train':
        train()
    elif args.command == 'evaluate':
        run_evaluation()
    elif args.command == 'predict':
        prediction, probabilities = answer_question(
            question=args.question,
            choices=args.choices,
            image_path=args.image_path,
        )
        print(f"Predicted answer: {prediction}")
        print("Probabilities (aligned to provided choices):")
        for choice, prob in zip(args.choices, probabilities):
            print(f"- {choice}: {prob:.3f}")

if __name__ == '__main__':
    main()
