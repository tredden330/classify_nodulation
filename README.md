# classify_nodulation
Training a computer to classify short (300 bp) sequences that influence nodulation in leguminous plants. 

Currently utilizing neural networks and k-nearest-neighbor approaches

## Current Program
- [x] Reads fastq file sequences, label them, and append them to a list
- [x] Encode sequences with floats or one-hot
- [x] Separate data into test/training sets
- [x] Create a neural network test-train code
- [x] Create Knn train-test code
- [x] Graph testing results
- [ ] Optimize hyperparameters and approached for optimum classification

### The Data
- *Medicago Truncatula* RNA-seq data has been pulled from [NCBI's Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra)
- Gene sequences that have been experimentally validated to influence nodulation were supplied from [this review of nitrogen-fixation research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6961631/)

### Output Graphs -- Neural Network

![percent_correct (1)](https://user-images.githubusercontent.com/88045526/223019032-48e0ebd8-9701-417a-af97-17d2d078dea9.png)

![errors (1)](https://user-images.githubusercontent.com/88045526/223019101-3a1be5a8-ec80-4e21-81fd-1894003aa125.png)

## References
These technical blog posts really helped this project come along:

[Simple PyTorch Neural Network using the iris dataset](https://www.kaggle.com/code/mohitchaitanya/simple-iris-dataset-classification-using-pytorch)

[One-hot DNA encoding](https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a)

