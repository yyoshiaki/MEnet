# base file : 201010_createref.ipynb

import os
import sys
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
import statsmodels.stats.multitest as multi

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

version = "210411"
f_ref = './data/210411_ref.csv'
f_ref_ordered = './data/210411_ref_ordered.csv'
f_cat = './data/210411_categories.csv'
list_n_region = [500]
th_diff = 0.5
range_cov = (0.025, 0.975) # select bins in which read coverage in 95%

xbp = sys.argv[1]
num_processes = int(sys.argv[2])

# restrict autosomes.
list_chr = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
       'chr16', 'chr17', 'chr18', 'chr19', 'chr2', 'chr20', 'chr21',
       'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9']


# def read_ref(f_ref):
#     df_ref = pd.read_csv(f_ref)
#     df_ref = df_ref.loc[df_ref['Tissue'].dropna().index]
#     dict_sort_Tissue = {x:i for i,x in enumerate(df_ref['Tissue'].unique())}
#     dict_sort_MinorGroup = {x:i for i,x in enumerate(df_ref['MinorGroup'].unique())}
#     df_ref['order_Tissue'] = df_ref['Tissue'].map(dict_sort_Tissue)
#     df_ref['order_MinorGroup'] = df_ref['MinorGroup'].map(dict_sort_MinorGroup)
#     df_ref['order'] = df_ref['order_Tissue'] * 1000 + df_ref['order_MinorGroup']
#     df_ref = df_ref.sort_values(by='order')
#     return df_ref


# df_ref = read_ref(f_ref)

def read_ref():
    return pd.read_csv(f_ref_ordered)

def create_orderd_ref():
    df_ref = pd.read_csv(f_ref)
    df_ref = df_ref.loc[df_ref['MinorGroup'].dropna().index]
    df_cat = pd.read_csv(f_cat)
    df_cat['order'] = df_cat.index
    df_cat.index = df_cat.MinorGroup
    df_ref['order'] = [df_cat.loc[x, 'order'] if x in df_cat.index else np.nan for x in df_ref.MinorGroup]
    df_ref = df_ref.sort_values(by='order')
    # df_ref = df_ref.drop('order', axis=1)
    df_ref.to_csv(f_ref_ordered, index=None)


create_orderd_ref()
df_ref = read_ref()

n_nonexit = 0
for p,r in tqdm(df_ref.iterrows(), total=df_ref.shape[0]):
    f = './data/' + r['Directory'] + '/' + r['FileID'] + '.{}bpwin.bed'.format(xbp)
    if not os.path.exists(f):
        print(f)
        n_nonexit += 1
    
if n_nonexit > 0:
    raise SystemExit('Unable to open {} files.'.format(n_nonexit))
    

i = 0
for p,r in tqdm(df_ref.iterrows(), total=df_ref.shape[0]):
    f = './data/' + r['Directory'] + '/' + r['FileID'] + '.{}bpwin.bed'.format(xbp)
    if not os.path.exists(f):
        print(r)

    s = r['FileID']
    d = pd.read_csv(f, sep='\t', header=None)
    d.columns = ['chr', 'start', 'end', 'rate_{}'.format(s), 'meth_{}'.format(s), 'unmeth_{}'.format(s)]

    if i == 0:
        df = d.copy()
        i = 1
    else:
        df = pd.merge(df, d, how='outer', on=['chr', 'start', 'end'])
df = df.fillna(0)

# read coverage filter

df['coverage'] = df[['meth_' + x for x in df_ref['FileID']] + 
        ['unmeth_' + x for x in df_ref['FileID']]].sum(axis=1)

l_cov = np.percentile(df['coverage'], range_cov[0]*100)
u_cov = np.percentile(df['coverage'], range_cov[1]*100)

df = df[(df['coverage'] >= l_cov) & (df['coverage'] <= u_cov)]


for f in df_ref['FileID']:
    df['rate_{}'.format(f)] = df['meth_{}'.format(f)] / (df['meth_{}'.format(f)] + df['unmeth_{}'.format(f)])
    #     df['rate_{}'.format(f)] = (df['meth_{}'.format(f)] + 1) / (df['meth_{}'.format(f)] + df['unmeth_{}'.format(f)] + 2)
    
for t in tqdm(df_ref['Tissue'].unique()):
    l_a = list(df_ref.loc[df_ref['Tissue'] == t, 'FileID'])
    l_b = list(df_ref.loc[df_ref['Tissue'] != t, 'FileID'])

    df['meth_{}'.format(t)] = df[['meth_{}'.format(x) for x in l_a]].sum(axis=1).astype(int)
    df['unmeth_{}'.format(t)] = df[['unmeth_{}'.format(x) for x in l_a]].sum(axis=1).astype(int)

    df['meth_non{}'.format(t)] = df[['meth_{}'.format(x) for x in l_b]].sum(axis=1).astype(int)
    df['unmeth_non{}'.format(t)] = df[['unmeth_{}'.format(x) for x in l_b]].sum(axis=1).astype(int)
    
    chunk_size = int(df.shape[0]/num_processes)
    chunks = np.array_split(df, num_processes)
    
    def chi2(d):
        l_d = []
        l_p = []
        for p,r in d.iterrows():
            try:
                g, p, dof, expctd = stats.chi2_contingency([[r['meth_{}'.format(t)], r['unmeth_{}'.format(t)]], 
                                                    [r['meth_non{}'.format(t)], r['unmeth_non{}'.format(t)]]])
        #         d = ((r['meth_{}'.format(t)] + 1) / (r['meth_{}'.format(t)] + r['unmeth_{}'.format(t)]) + 2) - \
        #         ((r['meth_non{}'.format(t)] + 1) / (r['meth_non{}'.format(t)] + r['unmeth_non{}'.format(t)]) + 2)
                diff = (r['meth_{}'.format(t)] / (r['meth_{}'.format(t)] + r['unmeth_{}'.format(t)])) - \
                (r['meth_non{}'.format(t)] / (r['meth_non{}'.format(t)] + r['unmeth_non{}'.format(t)]))
            except:
                p = 1
                diff = 0
            l_d.append(diff)
            l_p.append(p)
        return [l_d,l_p]
    
    
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        result = pool.map(chi2, chunks)
        pool.close()
        pool.terminate()
    else:
        result = [chi2(df)]
    
    list_d = []
    list_p = []
    for r in result:
        list_d += r[0]
        list_p += r[1]
    df['diff_{}'.format(t)] = list_d
    df['p_{}'.format(t)] = list_p
    df['fdr_{}'.format(t)] = multi.fdrcorrection(list_p)[1]
    
df_ref = df_ref.loc[df_ref['MinorGroup'].dropna().index]
for t in tqdm(df_ref['MinorGroup'].unique()):
    l_a = list(df_ref.loc[df_ref['MinorGroup'] == t, 'FileID'])
    l_b = list(df_ref.loc[df_ref['MinorGroup'] != t, 'FileID'])

    df['meth_{}'.format(t)] = df[['meth_{}'.format(x) for x in l_a]].sum(axis=1).astype(int)
    df['unmeth_{}'.format(t)] = df[['unmeth_{}'.format(x) for x in l_a]].sum(axis=1).astype(int)

    df['meth_non{}'.format(t)] = df[['meth_{}'.format(x) for x in l_b]].sum(axis=1).astype(int)
    df['unmeth_non{}'.format(t)] = df[['unmeth_{}'.format(x) for x in l_b]].sum(axis=1).astype(int)
    
    chunk_size = int(df.shape[0]/num_processes)
    chunks = np.array_split(df, num_processes)
    
    
    def chi2(d):
        l_d = []
        l_p = []
        for p,r in d.iterrows():
            try:
                g, p, dof, expctd = stats.chi2_contingency([[r['meth_{}'.format(t)], r['unmeth_{}'.format(t)]], 
                                                    [r['meth_non{}'.format(t)], r['unmeth_non{}'.format(t)]]])
        #         d = ((r['meth_{}'.format(t)] + 1) / (r['meth_{}'.format(t)] + r['unmeth_{}'.format(t)]) + 2) - \
        #         ((r['meth_non{}'.format(t)] + 1) / (r['meth_non{}'.format(t)] + r['unmeth_non{}'.format(t)]) + 2)
                diff = (r['meth_{}'.format(t)] / (r['meth_{}'.format(t)] + r['unmeth_{}'.format(t)])) - \
                (r['meth_non{}'.format(t)] / (r['meth_non{}'.format(t)] + r['unmeth_non{}'.format(t)]))
            except:
                p = 1
                diff = 0
            l_d.append(diff)
            l_p.append(p)
        return [l_d,l_p]
    
    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
        result = pool.map(chi2, chunks)
        pool.close()
        pool.terminate()
    else:
        result = [chi2(df)]
       
    list_d = []
    list_p = []
    for r in result:
        list_d += r[0]
        list_p += r[1]
    df['diff_{}'.format(t)] = list_d
    df['p_{}'.format(t)] = list_p
    df['fdr_{}'.format(t)] = multi.fdrcorrection(list_p)[1]
    
df.to_csv('./data/integrated/{v}_integrated_{x}bp.csv.gz'.format(v=version,x=xbp))

os.makedirs("./img/{}".format(version), exist_ok=True)
os.makedirs("./results/{}".format(version), exist_ok=True)

f = './data/integrated/{v}_integrated_{x}bp.csv.gz'.format(v=version, x=str(xbp))
print(f)
df = pd.read_csv(f, index_col=0)
df = df[df.chr.isin(list_chr)]
    
for n_region in list_n_region:
    # df_ref = read_ref(f_ref)
    df_ref = read_ref()
##### TISSUE
    list_idex = []
    txt = ""
    txt += "# Hyper methylated regions \n"
    for i,t in enumerate(df_ref['Tissue'].unique()):
        _l = []
        d_p = df[df['diff_{}'.format(t)] > th_diff]
        if d_p[d_p['fdr_{}'.format(t)] == 0].shape[0] >= n_region:
            _l += list(d_p[d_p['fdr_{}'.format(t)] == 0].sort_values(by='diff_{}'.format(t), ascending=False
                                                                      )[:n_region].index)
        else:
            _l += list(d_p[d_p['fdr_{}'.format(t)] == 0].index)
            _n_rest = n_region - d_p[d_p['fdr_{}'.format(t)] == 0].shape[0]
            _l += list(d_p[d_p['fdr_{}'.format(t)] > 0].sort_values(by='fdr_{}'.format(t))[:_n_rest].index)

        if i == 0:
            list_idx = _l
        else:
            list_idx += _l
#             print(t, len(_l))
        txt += "{t} : {l}\n".format(t=t, l=len(_l))
    txt += "# Hypo methylated regions \n"
    for i,t in enumerate(df_ref['Tissue'].unique()):
        _l = []
        d_n = df[df['diff_{}'.format(t)] < -th_diff]
        if d_n[d_n['fdr_{}'.format(t)] == 0].shape[0] >= n_region:
            _l += list(d_n[d_n['fdr_{}'.format(t)] == 0].sort_values(by='diff_{}'.format(t), ascending=True
                                                                      )[:n_region].index)
        else:
            _l += list(d_n[d_n['fdr_{}'.format(t)] == 0].index)
            _n_rest = n_region - d_n[d_n['fdr_{}'.format(t)] == 0].shape[0]
            _l += list(d_n[d_n['fdr_{}'.format(t)] > 0].sort_values(by='fdr_{}'.format(t))[:_n_rest].index)
        list_idx += _l
#             print(t, len(_l))
        txt += "{t} : {l}\n".format(t=t, l=len(_l))
        # list_idx = list(set(list_idx))
        
    with open('./results/{v}/ref_Tissue_{x}bp_{n}regions_{d}diff.txt'.format(
        v=version, x=xbp, n=n_region, d=th_diff), 'w') as f:
        f.write(txt)

    df_tis = df.loc[list_idx, ['rate_{}'.format(x) for x in df_ref.sort_values(by='order')['FileID']] + ['chr', 'start', 'end']]
    df_tis.index = df_tis['chr'] + ":" + df_tis['start'].astype(str) + "-" + df_tis['end'].astype(str)
    df_tis = df_tis.drop(['chr', 'start', 'end'], axis=1)
    df_tis.columns = df_ref.sort_values(by='order')['Tissue']

    plt.figure(figsize=(20,10))
    sns.heatmap(df_tis, cmap='cividis_r', vmin=0, vmax=1, yticklabels=False)
    plt.savefig('./img/{v}/heatmap_Tissue_{x}bp_{n}regions_{d}diff_all.pdf'.format(
            v=version, x=xbp, n=n_region, d=th_diff), bbox_inches='tight')
        
    df_tis_group = df_tis.T
    df_tis_group['group'] = df_tis_group.index
    df_tis_group = df_tis_group.groupby(by='group').mean().T
    df_tis_group = df_tis_group[df_ref['Tissue'].unique()]

    plt.figure(figsize=(8,5))
    sns.heatmap(df_tis_group, cmap='cividis_r', vmin=0, vmax=1, yticklabels=False)
    plt.savefig('./img/{v}/heatmap_Tissue_{x}bp_{n}regions_{d}diff_group.pdf'.format(
       v=version, x=xbp, n=n_region, d=th_diff), bbox_inches='tight')
        
    df_tis_group.to_csv('./results/{v}/ref_Tissue_{x}bp_{n}regions_{d}diff.csv'.format(
            v=version, x=xbp, n=n_region, d=th_diff))
        
##### MINOR GROUP
    df_ref = df_ref.loc[df_ref['MinorGroup'].dropna().index]
    list_idex = []
    txt = ""
    txt += "# Hyper methylated regions \n"
    for i,t in enumerate(df_ref['MinorGroup'].unique()):
        _l = []
        d_p = df[df['diff_{}'.format(t)] > th_diff]
        if d_p[d_p['fdr_{}'.format(t)] == 0].shape[0] >= n_region:
            _l += list(d_p[d_p['fdr_{}'.format(t)] == 0].sort_values(by='diff_{}'.format(t), ascending=False
                                                                      )[:n_region].index)
        else:
            _l += list(d_p[d_p['fdr_{}'.format(t)] == 0].index)
            _n_rest = n_region - d_p[d_p['fdr_{}'.format(t)] == 0].shape[0]
            _l += list(d_p[d_p['fdr_{}'.format(t)] > 0].sort_values(by='fdr_{}'.format(t))[:_n_rest].index)

        if i == 0:
            list_idx = _l
        else:
            list_idx += _l
#             print(t, len(_l))
        txt += "{t} : {l}\n".format(t=t, l=len(_l))

    txt += "# Hypo methylated regions \n"
    for i,t in enumerate(df_ref['MinorGroup'].unique()):
        _l = []
        d_n = df[df['diff_{}'.format(t)] < -th_diff]
        if d_n[d_n['fdr_{}'.format(t)] == 0].shape[0] >= n_region:
            _l += list(d_n[d_n['fdr_{}'.format(t)] == 0].sort_values(by='diff_{}'.format(t), ascending=True
                                                                      )[:n_region].index)
        else:
            _l += list(d_n[d_n['fdr_{}'.format(t)] == 0].index)
            _n_rest = n_region - d_n[d_n['fdr_{}'.format(t)] == 0].shape[0]
            _l += list(d_n[d_n['fdr_{}'.format(t)] > 0].sort_values(by='fdr_{}'.format(t))[:_n_rest].index)
        list_idx += _l
#             print(t, len(_l))
        txt += "{t} : {l}\n".format(t=t, l=len(_l))
        # list_idx = list(set(list_idx))
        
    with open('./results/{v}/ref_MinorGroup_{x}bp_{n}regions_{d}diff.txt'.format(
        v=version, x=xbp, n=n_region, d=th_diff), 'w') as f:
        f.write(txt)

    df_tis = df.loc[list_idx, ['rate_{}'.format(x) for x in df_ref.sort_values(by='order')['FileID']] + ['chr', 'start', 'end']]
    df_tis.index = df_tis['chr'] + ":" + df_tis['start'].astype(str) + "-" + df_tis['end'].astype(str)
    df_tis = df_tis.drop(['chr', 'start', 'end'], axis=1)
    df_tis.columns = df_ref.sort_values(by='order')['MinorGroup']

    plt.figure(figsize=(20,10))
    sns.heatmap(df_tis, cmap='cividis_r', vmin=0, vmax=1, yticklabels=False)
    plt.savefig('./img/{v}/heatmap_MinorGroup_{x}bp_{n}regions_{d}diff_all.pdf'.format(
            v=version, x=xbp, n=n_region, d=th_diff), bbox_inches='tight')
  
    df_tis_group = df_tis.T
    df_tis_group['group'] = df_tis_group.index
    df_tis_group = df_tis_group.groupby(by='group').mean().T
    df_tis_group = df_tis_group[df_ref['MinorGroup'].unique()]

    plt.figure(figsize=(8,5))
    sns.heatmap(df_tis_group, cmap='cividis_r', vmin=0, vmax=1, yticklabels=False)
    plt.savefig('./img/{v}/heatmap_MinorGroup_{x}bp_{n}regions_{d}diff_group.pdf'.format(
            v=version, x=xbp, n=n_region, d=th_diff), bbox_inches='tight')
        
    df_tis_group.to_csv('./results/{v}/ref_MinorGroup_{x}bp_{n}regions_{d}diff.csv'.format(
           v=version, x=xbp, n=n_region, d=th_diff))
