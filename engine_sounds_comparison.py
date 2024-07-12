import numpy as np
from glob import glob
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

#wav files to be tested should be in the same folder
files = list(glob("*.wav"))
files = np.sort(files)

mExact = np.zeros((len(files), len(files)))

for i, file1 in enumerate(files):
    for j, file2 in enumerate(files):
        ###change this according to names of files!!!###
        mExact[i, j] = 1 - (file1[:6] == file2[:6])

#Spectral Centroid
def compareSimilarity(file1, file2):
    audio1, sr1 = librosa.load(file1, mono=False)
    audio2, sr2 = librosa.load(file2, mono=False)

    channel1_audio1 = audio1[0]
    channel1_audio2 = audio2[0]

    spectral_centroid1 = librosa.feature.spectral_centroid(y=audio1, sr=sr1)
    spectral_centroid1 = np.mean(spectral_centroid1)

    spectral_centroid2 = librosa.feature.spectral_centroid(y=audio2, sr=sr2)
    spectral_centroid2 = np.mean(spectral_centroid2)

    centroid_distance = np.sqrt(np.sum((spectral_centroid1 - spectral_centroid2) ** 2))

    distance = centroid_distance

    return distance

index = np.arange(len(files))

m = np.zeros((len(index), len(index)))

for i, ind0 in tqdm(enumerate(index), total=len(index)):
    for j, ind1 in enumerate(index):
        m[i, j] = compareSimilarity(files[ind0], files[ind1])

index2 = np.argsort(files)

acc = lambda s: np.mean(np.float64(m[index2, :][:, index2] > s) == mExact[index2, :][:, index2])

#This allows to find the distance giving best accuracy
s = np.linspace(0, 1000, 1000)
accArr = [acc(s0) for s0 in s]
print(np.argmax(accArr))
print(np.max(accArr))
#plt.plot(s, accArr)

##############################

#uncomment below to plot the different distances
#fig = plt.figure(figsize=(14, 14))
#colorim = plt.imshow(m[index2,:][:,index2])
#plt.colorbar(colorim)
#plt.xticks(np.arange(len(index2)), [files[i] for i in index2], rotation=90)
#plt.yticks(np.arange(len(index2)), [files[i] for i in index2])

##############################

#uncomment below to plot the classification based on the best distance

#arr_red = np.where(m <= np.argmax(accArr), 1, 0)

#cmap_colors = ['black', 'green']
#cmap_bounds = [0, 0.5, 1]
#cmap = colors.ListedColormap(cmap_colors)
#norm = colors.BoundaryNorm(cmap_bounds, cmap.N)

#fig, ax = plt.subplots(figsize=(14, 14))
#im = ax.imshow(arr_red, cmap=cmap, norm=norm)

#cbar = ax.figure.colorbar(im)

#plt.xticks(np.arange(len(index2)), [files[i] for i in index2], rotation=90)
#plt.yticks(np.arange(len(index2)), [files[i] for i in index2])
#plt.show()

###############################

#uncomment below to show similarity matrix instead of classification matrix
#robot_instances = [f.split('-')[0] for f in files]

#fig, ax = plt.subplots(figsize=(14, 14))
#im = ax.imshow(m, cmap='gray_r', vmin=0, vmax=450)
#cbar = ax.figure.colorbar(im, ax=ax)
#ax.set_xticks(np.arange(len(index2)))
#ax.set_xticklabels([robot_instances[i] for i in index2], rotation=90)
#ax.set_yticks(np.arange(len(index2)))
#ax.set_yticklabels([robot_instances[i] for i in index2])
#plt.show()


###############################
#uncomment below to show distance bars of all samples

#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 20))
#for i in range(len(m)//2):
#    ax1.bar(np.arange(len(index2)) + i*0.1, m[i], width=0.1, label=f"{files[i][:9]}")

#ax1.set_xticks(np.arange(len(index2)) + 2*0.1)
#ax1.set_xticklabels([files[i] for i in index2], rotation=90)
#ax1.legend()

#for i in range(len(m) // 2, len(m)):
#    ax2.bar(np.arange(len(index2)) + (i-5)*0.1, m[i], width=0.1, label=f"{files[i][:9]}")
#ax2.set_xticks(np.arange(len(index2)) + 2*0.1)
#ax2.set_xticklabels([files[i] for i in index2], rotation=90)
#ax2.legend()
#plt.show()
