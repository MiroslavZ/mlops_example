from argparse import ArgumentParser, Namespace

import pandas as pd


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для предобработки и объединения наборов данных",)
    parser.add_argument("-s", "--source-dir", type=str, help="Директория, где лежат файлы для обработки", required=True, default='data/raw')
    parser.add_argument("-d", "--target-dir", type=str, help="Директория для сохранения обработанных файлов", required=True, default='data/stage1')
    return parser.parse_args()


args = _arg_parse()
SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir
# SOURCE_DIR = 'data/raw'
# TARGET_DIR = 'data/baselines'

# python3 scripts/data_processing/preprocessing.py -s data/raw -d data/baselines

# проще передать путь к директории, 
# чем передавать отдельно путь для каждого файла
assessments = pd.read_csv(f'{SOURCE_DIR}/assessments.csv')
courses = pd.read_csv(f'{SOURCE_DIR}/courses.csv')
results = pd.read_csv(f'{SOURCE_DIR}/studentAssessment.csv')
info = pd.read_csv(f'{SOURCE_DIR}/studentInfo.csv')
registration = pd.read_csv(f'{SOURCE_DIR}/studentRegistration.csv')
vle = pd.read_csv(f'{SOURCE_DIR}/studentVle.csv')
materials = pd.read_csv(f'{SOURCE_DIR}/vle.csv')


# конвертим id в object
assessments['id_assessment'] = assessments['id_assessment'].astype(object)

# правим веса оценок в датасете
assessments.loc[(assessments.code_module=='GGG') & (assessments.assessment_type=='TMA'),'weight'] = (100/3)
assessments.loc[(assessments.code_module=='GGG') & (assessments.assessment_type=='CMA'),'weight'] = (0)
assessments.loc[(assessments.code_module=='CCC') & (assessments.assessment_type=='Exam'),'weight'] = (100/2)


# конвертим id в object
results['id_assessment'] = results['id_assessment'].astype(object)
results['id_student'] = results['id_student'].astype(object)
registration['id_student'] = registration['id_student'].astype(object)
materials['id_site'] = materials['id_site'].astype(object)
vle['id_student'] = vle['id_student'].astype(object)
vle['id_site'] = vle['id_site'].astype(object)
info['id_student'] = info['id_student'].astype(object)



# regisrtation + courses
registration_courses = pd.merge(registration, courses, on=['code_module', 'code_presentation'], how='inner')


# regisrtation_courses + info
registration_courses_info = pd.merge(registration_courses, info, on=['code_module', 'code_presentation', 'id_student'], how='inner')


# vle + materials
materials_vle = pd.merge(vle, materials, on=['code_module', 'code_presentation', 'id_site'], how='inner')
materials_vle.drop(columns=['week_from', 'week_to', 'date'], inplace=True)


# общее количество дней взаимодействия студента
keys_act = ['code_module', 'code_presentation', 'id_student']
n_tot_days = vle.groupby(keys_act)['date'].nunique().to_frame().reset_index().rename(columns={'date': 'total_n_days'})


# получим общее количество кликов на учащегося за презентацию модуля.
total_click_per_student = materials_vle\
.groupby(['code_module', 'code_presentation', 'id_student'])\
.agg(total_click = ("sum_click",sum))\
.reset_index()


vle_grouped = vle.groupby(['code_module', 'code_presentation', 'id_student'])['date'].max().reset_index()
vle_courses = pd.merge(courses, vle_grouped, on=['code_module', 'code_presentation'], how='inner')
vle_courses['days_of_out'] = vle_courses['module_presentation_length'] - vle_courses['date']
vle_courses['late_submission_vle'] = vle_courses['days_of_out'] <= 0
vle_courses.drop(columns=['module_presentation_length', 'date'], inplace=True)


# assessments + results
assessment_results = pd.merge(assessments, results, on=['id_assessment'], how='inner')
assessment_results = assessment_results[['id_student', 'code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'date', 'date_submitted', 'weight', 'is_banked']]


# Рассчитаем разницу между датами сдачи
assessment_results['submission_days'] = assessment_results['date_submitted'] - assessment_results['date']
# Создадим столбец, указывающий, была ли подача запоздалой или нет
assessment_results['late_submission'] = assessment_results['submission_days'] > 0
# Рассчитаем столбец с коэфициентом поздних сдач
assessment_results['weighted_score'] = assessment_results['weight']
assessment_results.loc[assessment_results['late_submission'], 'weighted_score'] = 0
# Создадим общий вес всех сдач по студентам
total_w_score = assessment_results.groupby(['id_student', 'code_module', 'code_presentation']).agg({
    'weighted_score': 'sum',
    'late_submission': 'mean'
}).reset_index()

total_w_score = total_w_score.rename(columns={'weighted_score': 'total_weighted_score', 'late_submission': 'late_submission_rate'})


# соединяем все полученные таблицы
merged = pd.merge(registration_courses_info, total_click_per_student, on=['id_student', 'code_module', 'code_presentation'], how='left')
merged = pd.merge(merged, n_tot_days, on=['id_student', 'code_module', 'code_presentation'], how='left')
merged = pd.merge(merged, total_w_score, on=['id_student', 'code_module', 'code_presentation'], how='left')
merged = pd.merge(merged, vle_courses, on=['id_student', 'code_module', 'code_presentation'], how='left')

merged.to_csv(f'{TARGET_DIR}/merged.csv')
