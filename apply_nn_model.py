import torch
import torch.nn as nn
import nod_neural
import prepare_data
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt

def load_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = nod_neural.AppliedNeuralNetworkClassificationModel(nod_neural.input_dim,nod_neural.output_dim).to(device)
    model.load_state_dict(torch.load("models/nn_model.pt", map_location=device))
    model.eval()
    return model

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5*fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

model = load_model()
print("model loaded...")

neg_results = []
pos_results = []
with open("data/Medicago_truncatula.MedtrA17_4.0.dna.toplevel.fa") as f:
    for record in SeqIO.parse(f, "fasta"):
        windows = prepare_data.window(str(record.seq), 300)
        
        count = 0
        for window in windows:
            encoded_window = prepare_data.onehote(window)
            tensor = torch.FloatTensor(encoded_window)
            tensor = tensor.reshape(1200)
            result = model(tensor)
            neg_results = np.append(neg_results, result[0].detach().numpy())
            pos_results = np.append(pos_results, result[1].detach().numpy())
            count += 1
            if count > 5000:
                break
        break
        
    

print(neg_results)
print(pos_results)

x = range(len(neg_results))
fig, ax = plt.subplots()
#ax.plot(x, neg_results, 'r', label="not-nodule")
ax.plot(x, pos_results, 'c', label="nodule-likeliness")
ax.plot(x, butter_lowpass_filter(pos_results, 0.2, 30, 2), 'm', label="low-pass filtered")
plt.title("Chromosome 1")
plt.xlabel("Location (base pair)")
plt.ylabel("Score")
ax.grid()
plt.legend()
fig.savefig("graphs/nod_distribution.png")


print("seems to work")