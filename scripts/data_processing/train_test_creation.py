from argparse import ArgumentParser, Namespace

import pandas as pd


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для формирования обучающего набора данных",)
    parser.add_argument("-r", "--raw-data-dir", type=str, help="Директория, где лежат необработанные файлы train и test", required=True, default='data/raw')
    parser.add_argument("-m", "--merged-data-dir", type=str, help="Директория, где лежит объединенный набор данных", required=True, default='data/stage1')
    parser.add_argument("-d", "--target-dir", type=str, help="Директория для сохранения обработанных файлов", required=True, default='data/stage2')
    return parser.parse_args()


args = _arg_parse()
RAW_TRAIN_TEST_DIR = args.raw_data_dir
MERGED_DATA_DIR = args.merged_data_dir
TARGET_DIR = args.target_dir


train_who = pd.read_csv(f'{RAW_TRAIN_TEST_DIR}/Train_Who.csv')
test_who = pd.read_csv(f'{RAW_TRAIN_TEST_DIR}/Test_Who.csv')
merged = pd.read_csv(f'{MERGED_DATA_DIR}/merged.csv')


train = pd.merge(train_who, merged, on=['id_student', 'code_module', 'code_presentation'], how='left')

train.index = train.ID
train.drop(columns = ['ID'], inplace = True)


test = pd.merge(test_who, merged, on=['id_student', 'code_module', 'code_presentation'], how='left')

test.index = test.ID
test.drop(columns = ['ID'], inplace = True)


train.drop(columns = ['id_student'], inplace = True)
test.drop(columns = ['id_student'], inplace = True)

# тестовый набор здесь - набор, для которого делались предсказания на kaggle
train.to_csv(f'{TARGET_DIR}/train.csv')
test.to_csv(f'{TARGET_DIR}/test.csv')
