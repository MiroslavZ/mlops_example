from argparse import ArgumentParser, Namespace
from subprocess import PIPE, run

# датасет взят из соревнования https://www.kaggle.com/competitions/urfuaieng2022reg2
# для загрузки архива необходимо поместить свой токен kaggle.json в /home/username/.kaggle


def run_command(command: str) -> None:
    result = run(
        command,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        universal_newlines=True,
        check=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)


def get_data(competition: str, filename: str, target_dir: str):
    run_command(f'kaggle competitions download --force -p {target_dir} -c {competition}')
    print(f'Файл {filename} загружен')

    run_command(f'unzip -o {target_dir}/{filename} -d {target_dir}')
    print(f'Файл {filename} распакован')

    run_command(f'rm {target_dir}/{filename}')
    print(f'Файл {filename} удален')


def _arg_parse() -> Namespace:
    parser = ArgumentParser(description="Скрипт для получения датасета с kaggle",)
    parser.add_argument("-c", "--competition", type=str, help="Название соревнования", required=True, default='urfuaieng2022reg2')
    parser.add_argument("-f", "--filename", type=str, help="Имя файла для сохранения", required=True, default='urfuaieng2022reg2.zip')
    parser.add_argument("-d", "--target-dir", type=str, help="Директория для сохранения файла", required=True, default='data/raw')
    return parser.parse_args()


if __name__ == '__main__':
    args = _arg_parse()
    competition = args.competition
    filename = args.filename
    target_dir = args.target_dir
    get_data(competition, filename, target_dir)
    