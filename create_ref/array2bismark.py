import glob
import os
import subprocess
import pandas as pd
from tqdm import tqdm


'''
/mnt/media32TB/home/yyasumizu/bioinformatics/MEnet/notebooks/210422_array2bismark.ipynb
convert methyaltion arry file (series_matrix.txt downloaded in GEO) to bismark.
n_read : virtual coverage
'''

n_read = 10

list_manifest = ['../data/EPIC.hg38.manifest.tsv.gz',
                '../data/HM450.hg38.manifest.tsv.gz',
                '../data/HM27.hg38.manifest.tsv.gz']

l_m = []
for f_m in list_manifest:
    l_m.append(pd.read_csv(f_m, sep='\t')[['CpG_chrm', 'CpG_beg', 'CpG_end', 'probeID']].dropna())
    
df_manifest = pd.concat(l_m)

df_manifest['CpG_pos'] = df_manifest[['CpG_beg', 'CpG_end']].mean(axis=1).astype(int)

list_series_matrix = glob.glob('../data/**/*series_matrix.txt*', recursive=True)

for f_ser in tqdm(list_series_matrix):
    dir_ser = os.path.dirname(f_ser)

    df_array = pd.read_csv(f_ser, comment="!", sep='\t', index_col = 0)

    cols = df_array.columns
    df_array['probeID'] = df_array.index
    df_array = pd.merge(df_array, df_manifest, on='probeID', how='left')

    for c in cols:
        if not os.path.exists('{d}/{c}.bismark.cov.gz'.format(d=dir_ser, c=c)):
            df_q = df_array[['CpG_chrm', 'CpG_pos', c]].copy()
            df_q = df_q.dropna()
            df_q['CpG_pos'] = df_q['CpG_pos'].astype(int)
            df_q.columns = ['chr', 'pos', 'rate']
            df_q['met'] = n_read * df_q['rate']
            df_q['met'] = df_q['met'].round().astype(int)
            df_q['unmet'] = n_read * (1 - df_q['rate'])
            df_q['unmet'] = df_q['unmet'].round().astype(int)
            df_q['rate'] *= 100
            df_q = df_q[['chr', 'pos', 'pos', 'rate', 'met', 'unmet']]
            df_q.to_csv('{d}/{c}.bismark.cov.gz'.format(d=dir_ser, c=c), sep='\t', header=None,
                        index=None)