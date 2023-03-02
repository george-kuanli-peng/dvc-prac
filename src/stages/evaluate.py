import argparse
import json
from typing import Text

import joblib
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score

from src.report.visualization import plot_confusion_matrix


def evaluate(config_path: Text) -> None:
    with open(config_path, 'rt') as fin:
        config = yaml.safe_load(fin)
        
    print('✨ Load test data')
    test_dataset = pd.read_csv(config['data']['testset_path'])
    
    # Get X and Y
    y_test = test_dataset.loc[:, 'target'].values.astype('int32')
    X_test = test_dataset.drop('target', axis=1).values.astype('float32')
    
    print('✨ Load trained model')
    logreg = joblib.load(config['train']['model_path'])
    
    print('✨ Start evaluation')
    prediction = logreg.predict(X_test)
    cm = confusion_matrix(prediction, y_test)
    f1 = f1_score(y_true = y_test, y_pred = prediction, average='macro')
    
    # Save metrics
    print('✨ Save metrics')
    metrics = {
        'f1': f1
    }

    with open(config['reports']['metrics_file'], 'w') as mf:
        json.dump(
            obj=metrics,
            fp=mf,
            indent=4
        )
        
    # Save confusion matrix image
    print('✨ Save confusion matrix image')
    data = load_iris(as_frame=True)
    cm_plot = plot_confusion_matrix(cm, data.target_names, normalize=False)
    cm_plot.savefig(config['reports']['confusion_matrix_image'])
    
    print('✨ Evaluation DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    evaluate(config_path=args.config)
