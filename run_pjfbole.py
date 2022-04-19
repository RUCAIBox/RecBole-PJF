# @Time   : 2022/3/2
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn


import argparse

from pjfbole.quick_start import run_pjfbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='IPJF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='zhilian', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_pjfbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
