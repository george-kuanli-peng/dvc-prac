import argparse
from typing import Text

import pandas as pd
import yaml


def featurize(config_path: Text) -> None:
    with open(config_path, 'rt') as fin:
        config = yaml.safe_load(fin)
        
    print('✨ Load raw data')
    dataset = pd.read_csv(config['data']['dataset_csv'])
    
    print('✨ Extract features')
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']

    dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    #     'sepal_length_in_square', 'sepal_width_in_square', 'petal_length_in_square', 'petal_width_in_square',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]
    
    print('✨ Save features')
    dataset.to_csv(config['data']['features_path'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    featurize(config_path=args.config)
