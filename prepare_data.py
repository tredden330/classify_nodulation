import pandas as pd
import numpy as np
from Bio import SeqIO
import random

def assemble_data(num_samples):    #add raw fastq sequences into the nodulation sequence csv

    #write and label bulk rna-seq data into dataframe
    bulk_sequences = []
    with open("data/SRR23387605.fastq") as handle:

        for record in SeqIO.parse(handle, "fastq"):
            bulk_sequences.append(str(record.seq))

        small_bulk_sequences = random.choices(bulk_sequences, k=num_samples)
        print("bulk sequences num = ", len(small_bulk_sequences))

        data = {'genes':small_bulk_sequences, 'nod_relation':np.zeros(len(small_bulk_sequences))}

        bulk_df = pd.DataFrame(data)

    #write and label nodulation specific genes into dataframe
    nod_genes = pd.read_csv("data/nodulation_genes.csv")
    nod_sequences = nod_genes['Transcript']

    small_nod_sequences = []
    for seq in nod_sequences:
        for repeat in range(round(num_samples/len(nod_sequences))):    #randomly sample the long sequences into 300 bp segments
            if len(str(seq)) > 300:
                random_index = random.randrange(0, len(str(seq)) - 300)
                small_nod_sequences.append(seq[random_index:random_index+300])

    print("nod sequences num = ", len(small_nod_sequences))
    data = {'genes':small_nod_sequences, 'nod_relation': np.ones(len(small_nod_sequences))}

    small_nod_sequences = pd.DataFrame(data)

    #concatenate data together and save as csv
    df = pd.concat([bulk_df, small_nod_sequences])
    df.to_csv("data/genes.csv",index=False)


def encode_sequences():    #to input into neural network, sequences must be encoded as floats

    csv = pd.read_csv("data/genes.csv")

    float_sequences = []
    for gene_seq in csv['genes']:

        float_seq = []
        for base in gene_seq:
            if base == 'A' or base == 'a':
                float_seq.append(-1.0)
            if base == 'T' or base == 't':
                float_seq.append(-2.0)
            if base == 'C' or base == 'c':
                float_seq.append(1.0)
            if base == 'G' or base == 'g':
                float_seq.append(2.0)
            if base == 'N' or base == 'n':
                float_seq.append(0.0)

        if len(float_seq) != 300:
            print('there are letters unaccounted for here')
            break
        float_sequences.append(float_seq)

    csv['float_seq'] = float_sequences

    csv.to_csv("data/genes_with_float.csv", index=False)


assemble_data(1000)
encode_sequences()
