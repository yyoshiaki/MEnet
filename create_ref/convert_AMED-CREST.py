import os
import subprocess
import sys
import pandas as pd
from tqdm import tqdm

# bw2bedgraph = '/home/yyasumizu/Programs/hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64.v385/bigWigToBedGraph'
bw2bedgraph = "bigWigToBedGraph"

listdir = os.listdir('../data/AMED-CREST/')
list_sample = [x.split('.')[2] for x in listdir if 'signal' in x]

# df_metsig = pd.read_csv('../data/Blueprint/metadata_metsig.csv', index_col=0)
# df_metsig = pd.read_csv('../data/Blueprint/mini_metadata_metsig.csv', index_col=0)

for s in tqdm(list_sample):
    try:
        f_bs_call = '../data/AMED-CREST/' + [x for x in listdir if (s in x) and ('methylation_profile' in x)][0]
    except:
        print(s)
        sys.exit()
        # subprocess.run('wget -P ../data/Blueprint {}'.format(row['URL']) )
        # f_bs_call = '../data/Blueprint/' + [x for x in listdir if (row.Name in x) and ('bs_call' in x)][0]
    f_bs_cov = '../data/AMED-CREST/' + [x for x in listdir if (s in x) and ('signal' in x)][0]
    
    f_bs_call_bdg = '../data/AMED-CREST/' + s + 'bs_call.beg'
    f_bs_cov_bdg = '../data/AMED-CREST/' + s + 'bs_cov.beg'
    f_bs = '../data/AMED-CREST/' + s + '.bismark.cov.gz'
    
    cmd = [bw2bedgraph, f_bs_call, f_bs_call_bdg]
    subprocess.run(cmd)
    
    cmd = [bw2bedgraph, f_bs_cov, f_bs_cov_bdg]
    subprocess.run(cmd)
    
    df_bs_call = pd.read_csv(f_bs_call_bdg, sep='\t', header=None)
    df_bs_call.columns = ['chr', 'start', 'end', 'rate']
    df_bs_cov = pd.read_csv(f_bs_cov_bdg, sep='\t', header=None)
    df_bs_cov.columns = ['chr', 'start', 'end', 'cov']
    
    df_bs = pd.merge(df_bs_call, df_bs_cov, on=['chr', 'start', 'end'])
    
    df_bs['rate'] /= 100
    df_bs['meth'] = (df_bs['cov'] * df_bs['rate']).astype(int)
    df_bs['unmeth'] = df_bs['cov'] - df_bs['meth']
    df_bs['rate'] *= 100
    
    df_bs = df_bs[['chr', 'start', 'end', 'rate', 'meth', 'unmeth']]
    
    df_bs.to_csv(f_bs, sep='\t', header=None, index=None)