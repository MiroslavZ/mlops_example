from argparse import ArgumentParser, Namespace
from json import dump
from pickle import load

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для получения метрик модели на валиационном наборе данных",)
    parser.add_argument("-v", "--val-data-dir", type=str, help="Директория, где лежат наборы данных для валидации модели", required=True, default='data/stage5')
    parser.add_argument("-m", "--model-path", type=str, help="Путь до файла обученной модели", required=True, default='models/best_model')
    parser.add_argument("-s", "--score-path", type=str, help="Путь для сохранения метрик модели на валидационном наборе данных", required=True, default='evaluate/score.json')
    return parser.parse_args()


args = _arg_parse()
VAL_DATA_DIR = args.val_data_dir
MODEL_PATH = args.model_path
SCORE_PATH = args.score_path


def get_metrics(model, X_test, y_test):
  y_prediction_test = model.predict(X_test)  
  return {'MSE': mean_squared_error(y_test, y_prediction_test), 'R2': r2_score(y_test, y_prediction_test)}

X_val = np.loadtxt(f'{VAL_DATA_DIR}/X_val.txt')
y_val = np.loadtxt(f'{VAL_DATA_DIR}/y_val.txt')

with open(MODEL_PATH, 'rb') as fd:
    regressor = load(fd)

metrics = get_metrics(regressor, X_val, y_val)
with open(SCORE_PATH, 'w') as fd:
    dump(metrics, fd)
