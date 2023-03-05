# classify_nodulation
Training a computer to classify short (300 bp) sequences that influence nodulation in leguminous plants. Currently utilizing neural networks and k-nearest-neighbor approaches

## Current Program
- [x] Reads fastq file sequences, label them, and append them to a list
- [x] Encode all sequences of the list into floats
- [x] Separate data into test/training sets
- [x] Create a neural network test-train code
- [x] Create Knn train-test code
- [x] Graph testing results
- [ ] Optimize hyperparameters and approached for optimum classification

### The Data
- *Medicago Truncatula* RNA-seq data has been pulled from [NCBI's Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra)
- Gene sequences that have been experimentally validated to influence nodulation were supplied from [this review of nitrogen-fixation research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6961631/)

### Output Graphs

![percent_correct](https://user-images.githubusercontent.com/88045526/222621686-dce55aa4-adf5-4112-a762-1d7b1e4d4740.png)

![errors](https://user-images.githubusercontent.com/88045526/222621692-9d190714-fbe0-4428-b0cc-e897ba6c04e6.png)

## References
These technical blog posts really helped this project come along:

[Simple PyTorch Neural Network using the iris dataset](https://www.kaggle.com/code/mohitchaitanya/simple-iris-dataset-classification-using-pytorch)

[One-hot DNA encoding](https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a)

