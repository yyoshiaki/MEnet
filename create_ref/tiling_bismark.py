import glob
import os
import subprocess
import pandas as pd
from tqdm import tqdm

list_biscov = glob.glob('./data/**/*bismark.cov*', recursive=True) +\
glob.glob('./data/**/*bismark.sort.cov*', recursive=True)

list_xbpwin = glob.glob('./data/**/*bpwin.bed*', recursive=True)

for winbp in tqdm(['100','500','1000']):
    for biscov in list_biscov:
        base_biscov = biscov.replace('.bismark.sort.cov', '').replace('.bismark.cov', '').replace('.gz', '')
        bissort = base_biscov + '.sort.bed'
        xbpwin = base_biscov+'.'+winbp+'bpwin.bed'
        
        if not os.path.exists(xbpwin):
            cmd = 'bedtools sort -i {bis} > {bissort}'.format(bis=biscov, bissort=bissort)
            subprocess.run(cmd, shell=True)
            cmd = 'bedtools map -a ../data/hg38.win{x}.bed -b {bis} -c 4,5,6 -o mean,sum,sum | grep -v "\.\s*\." > {o}'.format(
    x = winbp, bis=bissort, o=xbpwin)
            subprocess.run(cmd, shell=True)