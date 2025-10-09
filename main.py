import argparse
from helpers import validate
from helpers import generate_f1_report

def main():
    parser = argparse.ArgumentParser(description='Clarity Project Controller')
    parser.add_argument('--validate', action='store_true', help='Run validation on model predictions')
    parser.add_argument('--evaluate', action='store_true', help="Generate F1 score reports")

    args = parser.parse_args()

    if args.validate:
        print('Running Validation')
        validate.main()
    elif args.evaluate:
        print('Generating F1 Score Reports')
        generate_f1_report.main()

if __name__ == "__main__":
    main()