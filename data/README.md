# DATA pre-process

## Data download

* [MIDORI2 Reference](https://www.reference-midori.info/download.php) : dowload 12S reads via this [link](https://www.reference-midori.info/download/Databases/GenBank261_2024-06-15/RAW/total/MIDORI2_TOTAL_NUC_GB261_srRNA_RAW.fasta.gz) and unzip it at [data/crabs/12S repository](data/crabs/12S)
* [MitoFish](https://mitofish.aori.u-tokyo.ac.jp/) : via CRABS comand see below

## Extract teleo marker via [CRABS](https://doi.org/10.1111/1755-0998.13741)

Refer to CRABS instalation [github](https://github.com/gjeunen/reference_database_creator?tab=readme-ov-file#installing-crabs). We will need Docker, conda or manually install CRABS (tested on Linux and Mac system).

1. Dowload MitoFish : `crabs db_download --source mitofish --output "/data/mitofish/mitofish.fasta" --keep_original yes`  

2. Perform an in silico PCR :   

    * MitoFish : `crabs insilico_pcr --input data/mitofish/mitofish.fasta --output data/mitofish/teleo_fb_3/output_teleo_3.fasta --fwd ACACCGCCCGTCACTCT --rev CTTCCGGTACACTTACCATG --error 3` 

    * MIDORI2 : `crabs insilico_pcr --input data/12S/MIDORI2_TOTAL_NUC_GB261_srRNA_RAW.fasta --output data/12S/teleo_12S_3/teleo_12S_3.fasta --fwd ACACCGCCCGTCACTCT --rev CTTCCGGTACACTTACCATG --error 3`
3. Perform an PGA (Pairwise Global Alignment) : 

    * MitoFish : `crabs pga --input data/fish_base/mitofish.fasta --output data/mitofish/teleo_fb_3/output_pga_3.fasta --database data/mitofish/teleo_fb_3/output_teleo_3.fasta --fwd ACACCGCCCGTCACTCT --rev CTTCCGGTACACTTACCATG --speed medium --percid 0.95 --coverage 0.95 --filter_method strict`

    * MIDORI2 : `crabs pga --input data/12S/MIDORI2_TOTAL_NUC_GB261_srRNA_RAW.fasta --output data/12S/teleo_12S_3/output_pga_3.fasta --database data/12S/teleo_12S_3/teleo_12S_3.fasta --fwd ACACCGCCCGTCACTCT --rev CTTCCGGTACACTTACCATG --speed medium --percid 0.95 --coverage 0.95 --filter_method strict`

## Assign Taxonomy :

* MitoFish : `crabs assign_tax --input data/mitofish/teleo_fb_3/output_pga_3.fasta --output data/mitofish/teleo_fb_3/output_pga_3.tsv --acc2tax nucl_gb.accession2taxid --taxid nodes.dmp --name names.dmp --missing "data/mitofish/teleo_fb_3/teleo_missing_taxa_pga_3.tsv"`

* MIDORI2 : `crabs assign_tax --input "data/12S/teleo_12S_3/output_pga_3.fasta" --output /data/12S/teleo_12S_3/output_pga_3.tsv --acc2tax nucl_gb.accession2taxid --taxid nodes.dmp --name names.dmp --missing "/data/12S/teleo_12S_3/teleo_missing_taxa.tsv"`

The two `output_pga_3.tsv` contains ....

## Process raw Teleo (duplicates, augmentation)