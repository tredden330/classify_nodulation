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

### Output Graphs

![nn_percent_correct (1)](https://user-images.githubusercontent.com/88045526/223278367-dd9dab91-bc2e-4299-9ac0-2409eeead7fc.png)

![nn_errors](https://user-images.githubusercontent.com/88045526/223278379-6a74458c-a984-4005-ac0b-f6698d3046b2.png)

![knn_percent (1)](https://user-images.githubusercontent.com/88045526/223278443-f4fb4d39-4a51-4919-9af0-1714d16a5a63.png)

## References
These technical blog posts really helped this project come along:

[Simple PyTorch Neural Network using the iris dataset](https://www.kaggle.com/code/mohitchaitanya/simple-iris-dataset-classification-using-pytorch)

[One-hot DNA encoding](https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a)

