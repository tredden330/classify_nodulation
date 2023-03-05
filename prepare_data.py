import pandas as pd
import numpy as np
from Bio import SeqIO
import random

def assemble_data(num_samples):    #add raw fastq sequences into the nodulation sequence csv

    #write and label bulk rna-seq data into dataframe
    bulk_sequences = []
    with open("data/SRR23387605.fastq") as handle:
        count = 0
        for record in SeqIO.parse(handle, "fastq"):
            bulk_sequences.append(str(record.seq))
            if (count > num_samples):
                break
            count += 1

        print(len(bulk_sequences))
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


def float_encode_sequences():    #to input into neural network, sequences must be encoded as floats

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
    csv = csv.sample(frac=1)
    csv.to_csv("data/genes_with_float_million.csv", index=False)

def onehot_encode_sequences():

    csv = pd.read_csv("data/genes.csv")
    onehot_sequences = []
    for gene_seq in csv['genes']:
        onehot_sequences.append(onehote(gene_seq))
    onehot_sequences = np.array(onehot_sequences)
    onehot_sequences = onehot_sequences.reshape(len(onehot_sequences), 1200)
    print(onehot_sequences)
    print(onehot_sequences[:,0])
    
    for num in range(400):
        csv[num] = onehot_sequences[:,num]
    
    #csv = csv.sample(frac=1)
    csv.to_csv("data/genes_with_onehot.csv", index=False)

def onehote(seq):
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "T":[0., 0., 0., 1.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
    return np.array(seq2)

assemble_data(900)
onehot_encode_sequences()
#float_encode_sequences()
