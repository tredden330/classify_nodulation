# classify_nodulation
Training a neural network to classify short (300 bp) sequences that influence nodulation in leguminous plants

## Current Program
- [x] Reads fastq file sequences, label them, and append them to a list
- [x] Encode all sequences of the list into floats
- [x] Separate data into test/training sets
- [x] Create a neural network architecture that can learn from the data
- [x] Graph testing results
- [ ] Optimize hyperparameters for optimum network

### The Data
- *Medicago Truncatula* RNA-seq data has been pulled from [NCBI's Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra)
- Gene sequences that have been experimentally validated to influence nodulation were supplied from [this review of nitrogen-fixation research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6961631/)

### Output Graphs

![percent_correct](https://user-images.githubusercontent.com/88045526/222621686-dce55aa4-adf5-4112-a762-1d7b1e4d4740.png)

![errors](https://user-images.githubusercontent.com/88045526/222621692-9d190714-fbe0-4428-b0cc-e897ba6c04e6.png)
