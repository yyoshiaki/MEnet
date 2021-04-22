# Create the reference for MEnet.

We prepared the reference locally. This is the method to prepare the reference for the traininig of MEnet.

1. create bismark files.
2. convert each dataset to bismark by convert_xxx.py. Brain samples was processed by Mr. Matsumoto using bismark.
3. mkwindow.hg38.sh
4. tiling_bismark.py
5. createref_multi.py ex. python createref_multi.py 1000

## 1. Create bismark files.

We used WGBS, RRBS, nanopore WGS and, methylation array (450K or EPIC) as reference. We first pre-processed each file as following procedures.

### WGBS

#### WGBS PE

```
#!/bin/bash
set -xe

cat SRR_Acc_List.txt | while read line; do
if [[ ! -f "${line}_1.fastq.gz" ]]; then
 fasterq-dump $line -O ./ -e 12 -p
 pigz -p 16 ${line}_1.fastq
 pigz -p 16 ${line}_2.fastq
fi
done

cat SRR_Acc_List.txt | while read line; do
if [[ ! -f "${line}_1_val_1.fq.gz" ]]; then
trim_galore --paired ${line}_1.fastq.gz ${line}_2.fastq.gz --cores 4
fi

if [[ ! -f "${line}_1_val_1_bismark_bt2_pe.bam" ]]; then
bismark --genome ../fasta -1 ${line}_1_val_1.fq.gz -2 ${line}_2_val_2.fq.gz --multicore 4
fi

if [[ ! -f "${line}_1_val_1_bismark_bt2_pe.deduplicated.bam" ]]; then
deduplicate_bismark --bam ${line}_1_val_1_bismark_bt2_pe.bam
fi

if [[ ! -f "${line}_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz" ]]; then
bismark_methylation_extractor --gzip  --multicore 16 --bedGraph ${line}_1_val_1_bismark_bt2_pe.deduplicated.bam
fi

if [[ ! -f "${line}.bismark.cov.gz" ]]; then
cp ${line}_1_val_1_bismark_bt2_pe.deduplicated.bismark.cov.gz ${line}.bismark.cov.gz
fi

done
```

### WGBS SE

```
#!/bin/bash
set -xe

cat SRR_Acc_List.txt | while read line; do
if [[ ! -f "$line.fastq.gz" ]]; then
 fasterq-dump $line -O ./ -e 12 -p
 pigz -p 16 ${line}.fastq
fi
done

cat SRR_Acc_List.txt | while read line; do

if [[ ! -f "${line}_trimmed.fq.gz" ]]; then
trim_galore ${line}.fastq.gz --cores 4
fi

if [[ ! -f "${line}_trimmed_bismark_bt2.bam" ]]; then
bismark --genome /fasta ${line}_trimmed.fq.gz --multicore 4
fi

if [[ ! -f "${line}_trimmed_bismark_bt2.deduplicated.bam" ]]; then
deduplicate_bismark --bam ${line}_trimmed_bismark_bt2.bam
fi

if [[ ! -f "${line}_trimmed_bismark_bt2.deduplicated.bismark.cov.gz" ]]; then
bismark_methylation_extractor --gzip --multicore 16 --bedGraph ${line}_trimmed_bismark_bt2.deduplicated.bam
fi

if [[ ! -f "${line}.bismark.cov.gz" ]]; then
cp ${line}_trimmed_bismark_bt2.deduplicated.bismark.cov.gz ${line}.bismark.cov.gz
fi

done
```

### RRBS PE

```
#!/bin/bash
set -xe

cat SRR_Acc_List.txt | while read line; do
if [[ ! -f "${line}_1.fastq.gz" ]]; then
         fasterq-dump $line -O ./ -e 12 -p
          pigz -p 16 ${line}_1.fastq
          pigz -p 16 ${line}_2.fastq
fi
done

cat SRR_Acc_List.txt | while read line; do
   if [[ ! -f "${line}_1_val_1.fq.gz" ]]; then
           trim_galore --paired ${line}_1.fastq.gz ${line}_2.fastq.gz --cores 4 --rrbs
   fi

   if [[ ! -f "${line}_1_val_1_bismark_bt2_pe.bam" ]]; then
           bismark --genome ../fasta -1 ${line}_1_val_1.fq.gz -2 ${line}_2_val_2.fq.gz --multicore 4
   fi

   if [[ ! -f "${line}_1_val_1_bismark_bt2_pe.bismark.cov.gz" ]]; then
           bismark_methylation_extractor --gzip  --multicore 16 --bedGraph ${line}_1_val_1_bismark_bt2_pe.bam
   fi

   if [[ ! -f "${line}.bismark.cov.gz" ]]; then
           cp ${line}_1_val_1_bismark_bt2_pe.bismark.cov.gz ${line}.bismark.cov.gz
   fi

done
```

### RRBS SE

```
#!/bin/bash
set -xe

cat SRR_Acc_List.txt | while read line; do
if [[ ! -f "$line.fastq.gz" ]]; then
 fasterq-dump $line -O ./ -e 12 -p
 pigz -p 16 ${line}.fastq
fi
done

cat SRR_Acc_List.txt | while read line; do
trim_galore ${line}.fastq.gz --cores 4 --rrbs
bismark --genome /fasta ${line}_trimmed.fq.gz --multicore 16
bismark_methylation_extractor --gzip --multicore 16 --bedGraph ${line}_trimmed_bismark_bt2.bam
done
```

### nanopore methylation call

```
guppy_basecaller -c dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac_prom.cfg --fast5_out  --compress_fastq -i fast5_pass/ -s guppy_basecalled/ --device "cuda:0 cuda:1 "

FAST5PATH=./guppy_basecalled
REFERENCE=/home/yyasumizu/reference/Gencode_v34/GRCh38.p13.genome.fa
# REFERENCE=/home/yyasumizu/practice/guppy_medaka/methylation_example/reference.fasta
OUTBAM=meth.bam
fast5mod guppy2sam ${FAST5PATH} \
--reference ${REFERENCE} \
--recursive --recursive --io_workers 32 --workers 128 --debug \
| samtools sort -@ 32 | samtools view -b -@ 32 > ${OUTBAM}
samtools index ${OUTBAM}

BAM=meth.bam
REGIONS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 \
chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
for REG in ${REGIONS[@]}
do
fast5mod call --meth cpg ${BAM} ${REFERENCE} ${REG} meth.${REG}.tsv
cat meth.${REG}.tsv >> meth.tsv
rm meth.${REG}.tsv
done
pigz meth.tsv
python ~/yyoshiaki-git/meth_atlas/convert_fast5mod2bismark.py meth.tsv.gz meth.bis.cov.gz
```

### Methylation Array

We adopted only 450K or EPIC array as input. First, methylatiom rate (beta value) was multiplied by n_read(10) to create pseudo bismark.cov file.

```
python array2bismark.py
```