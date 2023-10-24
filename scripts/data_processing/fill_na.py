from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для заполнения пропусков в данных",)
    parser.add_argument("-s", "--source-dir", type=str, help="Директория, где лежат файлы для обработки", required=True, default='data/stage2')
    parser.add_argument("-d", "--target-dir", type=str, help="Директория для сохранения обработанных файлов", required=True, default='data/stage3')
    return parser.parse_args()


args = _arg_parse()
SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir
# SOURCE_DIR = 'data/baselines'
# TARGET_DIR = 'data/baselines'

# python3 scripts/data_processing/fill_na.py -s data/baselines -d data/baselines

train = pd.read_csv(f'{SOURCE_DIR}/train.csv')
test = pd.read_csv(f'{SOURCE_DIR}/test.csv')


# заменим все нулевые значения соответствующими наиболее часто встречающимися значениями imd_bands
regions_list = list(train[train['imd_band'].isnull()]['region'].unique())

for i in regions_list:
    train['imd_band'] = np.where(((train['imd_band'].isnull()) & (train['region'] == i) ),train[train['region'] == i].imd_band.mode(),train['imd_band'])

regions_list = list(test[test['imd_band'].isnull()]['region'].unique())

for i in regions_list:
    test['imd_band'] = np.where(((test['imd_band'].isnull()) & (test['region'] == i) ),test[test['region'] == i].imd_band.mode(),test['imd_band'])


train['total_click'] = train['total_click'].replace(np.nan).fillna(0)
test['total_click'] = test['total_click'].replace(np.nan).fillna(0)

train['date_registration'].fillna(train['date_registration'].median(), inplace=True)
test['date_registration'].fillna(test['date_registration'].median(), inplace=True)

train['total_n_days'].fillna(train['total_n_days'].median(), inplace=True)
test['total_n_days'].fillna(test['total_n_days'].median(), inplace=True)

train['late_submission_rate'] = train['late_submission_rate'].replace(np.nan).fillna(1.0)
test['late_submission_rate'] = test['late_submission_rate'].replace(np.nan).fillna(1.0)

train['total_weighted_score'].fillna(train['total_weighted_score'].median(), inplace=True)
test['total_weighted_score'].fillna(test['total_weighted_score'].median(), inplace=True)

train['days_of_out'].fillna(train['days_of_out'].median(), inplace=True)
test['days_of_out'].fillna(test['days_of_out'].median(), inplace=True)

train['late_submission'] = train['days_of_out'] <= 0
test['late_submission'] = test['days_of_out'] <= 0

# объеденим уровень образования

train['highest_education'] = np.where( (train['highest_education'] == 'No Formal quals'),
                                           'Lower Than A Level',
                                           train['highest_education']
                                    )

train['highest_education'] = np.where( (train['highest_education'] == 'Post Graduate Qualification'),
                                           'HE Qualification',
                                           train['highest_education']
                                    )


test['highest_education'] = np.where( (test['highest_education'] == 'No Formal quals'),
                                           'Lower Than A Level',
                                           test['highest_education']
                                    )

test['highest_education'] = np.where( (test['highest_education'] == 'Post Graduate Qualification'),
                                           'HE Qualification',
                                           test['highest_education']
                                    )


# объеденим 55+ и 35-55 в 35+
train['age_band'] = np.where( (train['age_band'] == '55<='),
                                           '35+',
                                           train['age_band']
                                    )

train['age_band'] = np.where( (train['age_band'] == '35-55'),
                                           '35+',
                                           train['age_band']
                                    )

test['age_band'] = np.where( (test['age_band'] == '55<='),
                                           '35+',
                                           test['age_band']
                                    )

test['age_band'] = np.where( (test['age_band'] == '35-55'),
                                           '35+',
                                           test['age_band']
                                    )


train.to_csv(f'{TARGET_DIR}/train.csv')
test.to_csv(f'{TARGET_DIR}/test.csv')
