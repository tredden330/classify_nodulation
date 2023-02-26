# classify_nodulation
Training a neural network to classify short (300 bp) sequences that influence nodulation in leguminous plants

## Current Program
- [x] Reads fastq file sequences, label them, and append them to a master list
- [x] Encode all sequences of the master list into floats
- [x] Separate data into test/training sets
- [x] Create a neural network architecture that can learn from the data
- [ ] Graph testing results
- [ ] Optimize hyperparameters for optimum network

### The Data
- *Medicago Truncatula* RNA-seq data has been pulled from [NCBI's Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra)
- Gene sequences that have been experimentally validated to influence nodulation were supplied from [this review of nitrogen-fixation research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6961631/)

### Output

![image](https://user-images.githubusercontent.com/88045526/221385648-414c5f02-902a-4898-b098-2623c70bd3c0.png)
