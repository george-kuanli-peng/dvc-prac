import argparse
from typing import Text

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def data_split(config_path: Text) -> None:
    with open(config_path, 'rt') as fin:
        config = yaml.safe_load(fin)
        
    print('✨ Load features')
    dataset = pd.read_csv(config['data']['features_path'])

    train_dataset, test_dataset = train_test_split(dataset, test_size=config['data']['test_size'], random_state=config['base']['random_state'])
    
    # Save train and test sets
    train_dataset.to_csv(config['data']['trainset_path'])
    test_dataset.to_csv(config['data']['testset_path'])
    
    print('✨ Data split DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    data_split(config_path=args.config)
