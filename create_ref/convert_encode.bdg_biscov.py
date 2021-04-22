import pandas as pd
from tqdm import tqdm

df_encode_meta = pd.read_csv('../data/ENCODE/metadata.tsv', sep='\t')
df_encode_meta = df_encode_meta[df_encode_meta['Output type'] == 'methylation state at CpG']

for pos,row in tqdm(df_encode_meta.iterrows(), total=df_encode_meta.shape[0]):
    f_enc = '../data/ENCODE/{}.bed.gz'.format(row['File accession'])
    f_enc_out = '../data/ENCODE/{}.bismark.cov.gz'.format(row['File accession'])
    df_enc = pd.read_csv(f_enc, sep='\t', header=None)

    df_enc[10]
    df_enc[11] = (df_enc[9] * (100 - df_enc[10]) / 100).astype(int)
    df_enc[12] = df_enc[9] - df_enc[11]

    df_enc = df_enc[[0,1,2,10,12,11]]
    df_enc.to_csv(f_enc_out, sep='\t', header=None, index=None)
