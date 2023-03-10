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
- [ ] Optimize hyperparameters for best classification
- [x] Apply model to discover candidate genes for knock-out

### The Data
- *Medicago Truncatula* RNA-seq data has been pulled from [NCBI's Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra)
- Gene sequences that have been experimentally validated to influence nodulation were supplied from [this review of nitrogen-fixation research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6961631/)

### Model Accuracy


![nn_percent_correct](https://user-images.githubusercontent.com/88045526/223479343-bfc8de4a-84d9-4d60-8ed1-83460cd12854.png)

![nn_errors](https://user-images.githubusercontent.com/88045526/223479576-787a505b-cb1f-478c-a950-33be89776c92.png)

![knn_percent (1)](https://user-images.githubusercontent.com/88045526/223278443-f4fb4d39-4a51-4919-9af0-1714d16a5a63.png)

### Model Application

After training the model, I apply it to the *Medicago Truncatula* genome

![nod_distribution](https://user-images.githubusercontent.com/88045526/224168154-6e34dc87-88f0-4833-aeb8-021a03d84dac.png)

![nod_distribution (1)](https://user-images.githubusercontent.com/88045526/224172599-8dc016ab-d0e7-4e47-9b3f-1b5db417108c.png)

## References
These technical blog posts really helped this project come along:

[Simple PyTorch Neural Network using the iris dataset](https://www.kaggle.com/code/mohitchaitanya/simple-iris-dataset-classification-using-pytorch)

[One-hot DNA encoding](https://elferachid.medium.com/one-hot-encoding-dna-92a1c29ba15a)

[Low-pass Filter](https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7)

