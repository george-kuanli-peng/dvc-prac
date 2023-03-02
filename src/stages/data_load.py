import argparse
from typing import Text

import pandas as pd
import yaml
from sklearn.datasets import load_iris


def data_load(config_path: Text) -> None:
    with open(config_path, 'rt') as fin:
        config = yaml.safe_load(fin)

    data = load_iris(as_frame=True)
    dataset = data.frame
    
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    dataset.to_csv(config['data']['dataset_csv'], index=False)
    
    print('✨ Data load DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    data_load(config_path=args.config)
