import argparse

from model.model import ModelArgs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-s', '--save-model', action='store_true')
    parser.add_argument('-o', '--output-name')
    parser.add_argument('-f', '--from-file')
    parser.add_argument('-d', '--data')
    parser.add_argument('--train-data')
    parser.add_argument('--test-data')
    parser.add_argument('--train-data-label')
    parser.add_argument('--test-data-label')
    parser.add_argument('--train-data-count', type=int)
    parser.add_argument('--test-data-count', type=int)
    parser.add_argument('--text-column', default='text')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('-g', '--greedy', action='store_true')
    return parser.parse_args()

def get_model_args():
    args = parse_args()
    return {
        'model': args.model,
        'save_in_file': args.save_model,
        'output_name': args.output_name,
        'pretrained_path': args.from_file,
        'data_file': args.data,
        'train_data_file': args.train_data,
        'test_data_file': args.test_data,
        'train_data_label': args.train_data_label,
        'test_data_label': args.test_data_label,
        'train_data_count': args.train_data_count,
        'test_data_count': args.test_data_count,
        'text_column_name': args.text_column,
        'label_column_name': args.label_column,
        'greedy': args.greedy,
        }