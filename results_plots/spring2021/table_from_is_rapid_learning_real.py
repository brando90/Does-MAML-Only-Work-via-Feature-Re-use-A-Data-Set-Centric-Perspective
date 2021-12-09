#%%
from pathlib import Path
from pprint import pprint

import argparse

import torch
import pandas as pd
from pandas import DataFrame
from uutils import to_latex_is_rapid_learning_real
from uutils.torch import floatify_results, get_mean_std_pairs

import matplotlib.pyplot as plt

from collections import OrderedDict

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

CXA = ['cca', 'cka']
L2 = ['nes', 'cosine']
OUTPUT = ['nes_output', 'query_loss']

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_plot', action='store_true', help='stores plot if true')
args = parser.parse_args()

def get_results(stats_and_sims):
    stats = stats_and_sims['stats']
    stats = floatify_results(stats)
    return stats

def get_latex_table(sig, stats):
    # cca = stats['cca']
    # data = [
    #     ('cca', *get_mean_std_pairs(cca), *get_mean_std_pairs(cca['rep']), *get_mean_std_pairs(cca['all']), '-')
    # ]
    data = []
    for metric_name, metrics in stats.items():
        print(f'{metric_name=}')
        if metric_name in OUTPUT:
            row_v = ['-']*6
            pair = get_mean_std_pairs({'avg': metrics['avg'], 'std': metrics['std']})
            row_v = row_v + pair
        else:
            layers = get_mean_std_pairs(metrics)
            rep_layer = get_mean_std_pairs(metrics['rep'])
            all_layers = get_mean_std_pairs(metrics['all'])
            row_v = (*layers, *rep_layer, *all_layers, '-')
        data.append(row_v)

    columns = ['L1', 'L2', 'L3', 'L4', 'L1-3 (rep)', 'L1-4 (all)', 'Ouput']
    print(f'{columns=}')

    # print(f'{data=}')
    for d in data:
        print(d)

    rows = stats.keys()
    print(f'{rows=}')

    print()
    print(f'{sig=}')
    df = DataFrame(data=data, columns=columns, index=rows)
    print(df)

    print()
    # latex = df.to_latex(column_format='|')
    # latex = df.to_latex()
    latex = to_latex_is_rapid_learning_real(df)

    sig_table_rows = "\\hline \n" \
                     "{} & \\multicolumn {7} { | c | }{ $\\sigma ^ {(1)}$ = {" + sig +"} } \\\\ \n"
    # print(sig_table_rows)
    latex_str = ''
    for i, line in enumerate(latex.splitlines()):
        if i == 1:
            latex_str = latex_str + sig_table_rows
        latex_str = latex_str + line + "\n"

    print(latex_str)

def plot_sig_vs_sim(sigs, metric_name):
    sigmas = []
    mus = []
    stds = []
    for sig, stats in sigs.items():
        sigmas.append(sig)
        if metric_name == 'nes_output':
            mu, std = stats[metric_name]['avg'], stats[metric_name]['std']
        elif metric_name == 'cca_rep':
            metric, metric_sort = metric_name.split('_')
            mu, std = stats[metric][metric_sort]['avg'], stats[metric][metric_sort]['std']
        elif metric_name == 'cka_rep':
            metric, metric_sort = metric_name.split('_')
            mu, std = stats[metric][metric_sort]['avg'], stats[metric][metric_sort]['std']
        else:
            raise ValueError(f'Unknown {metric_name=}')
        mus.append(mu)
        stds.append(std)

    fig, axs = plt.subplots(1, 1, sharex=True, tight_layout=True)

    # plot 0
    metric_name = metric_name.replace('_', ' ')
    axs.errorbar(sigmas, mus, yerr=std, marker='o', label=metric_name)
    # axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
    axs.legend()
    axs.set_title('Rapid learning vs Std of meta-learning data set')
    axs.set_xlabel('Std of task')
    axs.set_ylabel(f'Represenation Similarity ({metric_name})')
    # axs[0].set_ylim([0, 1])
    plt.tight_layout()

    if args.save_plot:
        fname = f'experiment_result_{metric_name}'
        # fname = f'experiment_result'
        root = Path('~/Desktop/').expanduser()
        plt.savefig(root / f'{fname}.png')
        plt.savefig(root / f'{fname}.svg')
        plt.savefig(root / f'{fname}.pdf')

    plt.show()

def main():
    sigs = OrderedDict()

    sig = '1e-16'
    stats_and_sims = torch.load('/Users/brando/data/logs/logs_Feb25_14-57-40_jobid_485176.iam-pbs_1e-16/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '1e-8'
    stats_and_sims = torch.load('/Users/brando/data/logs/logs_Feb25_11-22-32_jobid_1e-08/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '1e-4'
    stats_and_sims = torch.load('/Users/brando/data/logs/logs_Feb25_11-06-05_jobid_1e-4/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '1e-2'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb16_11-40-13_jobid_482800.iam-pbs/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '1e-1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb25_11-52-32_jobid_485174.iam-pbs_0.1/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '1e1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb25_12-18-24_jobid_485173.iam-pbs_1.0/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '2e1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb25_12-20-03_jobid_485173.iam-pbs_2.0/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '4e1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb16_11-41-28_jobid_482801.iam-pbs/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '8e1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb26_10-32-10_jobid_485432.iam-pbs_8.0/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '16e1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb26_10-35-41_jobid_485432.iam-pbs_16.0/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    sig = '32e1'
    stats_and_sims = torch.load(
        '/Users/brando/data/logs/logs_Feb26_10-38-04_jobid_485432.iam-pbs_32.0/stats_and_all_sims.pt')
    print(sig)
    sigs[sig] = get_results(stats_and_sims)

    # load data
    # stats = stats_and_sims['stats']
    # stats = floatify_results(stats)

    # do data analysis (to table and latex)
    # get_latex_table(sig, sigs[sig])

    # do data analysis (to sig vs sim metric plot)
    plot_sig_vs_sim(sigs, metric_name='nes_output')
    plot_sig_vs_sim(sigs, metric_name='cca_rep')
    plot_sig_vs_sim(sigs, metric_name='cka_rep')

if __name__ == '__main__':
    main()