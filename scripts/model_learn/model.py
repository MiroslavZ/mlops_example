from argparse import ArgumentParser, Namespace
from pickle import dump

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для обучения модели",)
    parser.add_argument("-s", "--source-dir", type=str, help="Директория, где лежат файлы для обучения модели", required=True, default='data/stage5')
    parser.add_argument( "-m", "--model-path", type=str, help="Путь для сохранения обученной модели", required=True, default='models/best_model')
    return parser.parse_args()


args = _arg_parse()
SOURCE_DIR = args.source_dir
MODEL_PATH = args.model_path


X_train = np.loadtxt(f'{SOURCE_DIR}/X_train.txt')
y_train = np.loadtxt(f'{SOURCE_DIR}/y_train.txt')


best_model = HistGradientBoostingRegressor(l2_regularization=0.2, learning_rate=0.04, max_iter=1100, max_leaf_nodes=60, random_state=42)
best_model.fit(X_train, y_train)

with open(MODEL_PATH, "wb") as fd:
    dump(best_model, fd)