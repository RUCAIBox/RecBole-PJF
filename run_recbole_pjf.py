# @Time   : 2022/3/2
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn


import argparse

from recbole_pjf.quick_start import run_recbole_pjf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='PJFNN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_pjf(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
