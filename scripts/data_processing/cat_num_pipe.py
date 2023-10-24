from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для обработки категориальных и числовых данных через Pipeline",)
    parser.add_argument("-s", "--source-dir", type=str, help="Директория, где лежат файлы для обработки", required=True, default='data/stage3')
    parser.add_argument("-d", "--target-dir", type=str, help="Директория для сохранения обработанных файлов", required=True, default='data/stage4')
    return parser.parse_args()


args = _arg_parse()
SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir
# SOURCE_DIR = 'data/baselines'
# TARGET_DIR = 'data/baselines'

# python3 scripts/data_processing/cat_num_pipe.py -s data/baselines -d data/baselines

def get_cat_num_cols(df):
  cat_columns = []
  num_columns = []

  for column_name in df.columns:
    if (df[column_name].dtypes == object):
      cat_columns +=[column_name]
    else:
      num_columns +=[column_name]
  return cat_columns, num_columns


train = pd.read_csv(f'{SOURCE_DIR}/train.csv')
test = pd.read_csv(f'{SOURCE_DIR}/test.csv')


cat_cols, num_cols = get_cat_num_cols(train)

numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent', )),
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

preprocessors = ColumnTransformer(transformers=[
    ('num', numerical_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

preprocessors.fit(train)

X_train = preprocessors.transform(train)
X_test = preprocessors.transform(test)

# зто не датафреймы
np.savetxt(f'{TARGET_DIR}/train.txt', X_train)
np.savetxt(f'{TARGET_DIR}/test.txt', X_test)
