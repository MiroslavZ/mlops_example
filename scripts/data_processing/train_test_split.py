from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для выделения обучающей и тестовой выборки",) # по идее не тестовой, а валидационной:)
    parser.add_argument( "-s", "--source-dir", type=str, help="Директория, где лежат файлы для обработки", required=True, default='data/stage4')
    parser.add_argument( "-t", "--target-var-dir", type=str, help="Директория, где лежит файл со значениями целевой переменной", required=True, default='data/raw')
    parser.add_argument( "-d", "--target-dir", type=str, help="Директория для сохранения обработанных файлов", required=True, default='data/stage5')
    parser.add_argument( "-ts", "--test-size", type=float, help="Доля данных для тестовой выборки", required=True, default=0.2)
    return parser.parse_args()


args = _arg_parse()
SOURCE_DIR = args.source_dir
TARGET_VARIABLE_DIR = args.target_var_dir
TARGET_DIR = args.target_dir
TEST_SIZE = args.test_size


RANDOM_STATE = 42

X_train = np.loadtxt(f'{SOURCE_DIR}/train.txt')
target = pd.read_csv(f'{TARGET_VARIABLE_DIR}/Train_Target_reg.csv', index_col = 'ID')


X_train, X_val, y_train, y_val = train_test_split(X_train, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# зто не датафреймы
np.savetxt(f'{TARGET_DIR}/X_train.txt', X_train)
np.savetxt(f'{TARGET_DIR}/X_val.txt', X_val)
np.savetxt(f'{TARGET_DIR}/y_train.txt', y_train)
np.savetxt(f'{TARGET_DIR}/y_val.txt', y_val)