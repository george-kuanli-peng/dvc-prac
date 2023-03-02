import argparse
from typing import Text

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression


def train(config_path: Text) -> None:
    with open(config_path, 'rt') as fin:
        config = yaml.safe_load(fin)
        
    print('✨ Load training data')
    train_dataset = pd.read_csv(config['data']['trainset_path'])
    
    # Get X and Y
    y_train = train_dataset.loc[:, 'target'].values.astype('int32')
    X_train = train_dataset.drop('target', axis=1).values.astype('float32')
    
    # Create an instance of Logistic Regression Classifier CV and fit the data
    logreg = LogisticRegression(**config['train']['clf_params'], random_state=config['base']['random_state'])
    logreg.fit(X_train, y_train)
    
    print('✨ Save trained model')
    joblib.dump(logreg, config['train']['model_path'])
    
    print('✨ Train DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    train(config_path=args.config)
