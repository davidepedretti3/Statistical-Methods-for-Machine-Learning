# -*- coding: utf-8 -*-

# importing libraries
import librosa
from librosa import feature
from glob import glob
import numpy as np

# path to the different folders
fold1 = glob(r"UrbanSound8K\audio\fold1\*.wav")
fold2 = glob(r"UrbanSound8K\audio\fold2\*.wav")
fold3 = glob(r"UrbanSound8K\audio\fold3\*.wav")
fold4 = glob(r"UrbanSound8K\audio\fold4\*.wav")
fold6 = glob(r"UrbanSound8K\audio\fold6\*.wav")
fold5 = glob(r"UrbanSound8K\audio\fold5\*.wav")
fold7 = glob(r"UrbanSound8K\audio\fold7\*.wav")
fold8 = glob(r"UrbanSound8K\audio\fold8\*.wav")
fold9 = glob(r"UrbanSound8K\audio\fold9\*.wav")
fold10 = glob(r"UrbanSound8K\audio\fold10\*.wav")

# datasets for training and testing
dataset_path_train = fold1+fold2+fold3+fold4+fold6
dataset_path_test = [fold5,fold7,fold8,fold9,fold10]

# function that aims at loading the audio files and extracting some features
def get_feature_vector(y): 
    
    # loading the audio file
    sound_file, sample_rate = librosa.load(y, sr=None)
    
    # extracting mfccs and calculating mean, median, max and min 
    mfccs = feature.mfcc(sound_file,sample_rate,n_mfcc=12)
    # calculating mean, median, max and min 
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_median = np.median(mfccs.T,axis=0)
    mfccs_max = np.max(mfccs.T,axis=0)
    mfccs_min = np.min(mfccs.T,axis=0)
    # concatenating all the statistics calculated on MFCCs
    mfccs = np.concatenate((mfccs_mean, mfccs_median, mfccs_max,mfccs_min),axis=0)   
    
    # extract chroma stft and calculating mean, median, max and min 
    chroma = feature.chroma_stft(sound_file, sample_rate,n_chroma=12)    
    chroma_mean = np.mean(chroma.T,axis=0)
    chroma_median = np.median(chroma.T,axis=0)
    chroma_max =np.max(chroma.T,axis=0)
    chroma_min = np.min(chroma.T,axis=0)    
    # concatenating all the statistics calculated on chroma stft                
    chroma = np.concatenate((chroma_mean,chroma_median, chroma_max,chroma_min),axis=0)  
    
    # extracting rms and calculating mean, median, max and min 
    rms = feature.rms(sound_file, sample_rate)
    rms_mean = np.mean(rms)
    rms_median = np.median(rms)
    rms_max = np.max(rms)
    rms_min = np.min(rms)
    # concatenating all the statistics calculated on rms
    rms = np.array([rms_mean, rms_median, rms_max, rms_min])      
    
    feature_vector = np.concatenate((mfccs, chroma, rms), axis=0)
    
    return feature_vector

# main
if __name__ == "__main__":    
    
    # name of the file that will be generated
    first_output = 'features_for_training.csv'

    audio_features = []
    for file in dataset_path_train: 
        feature_vector = get_feature_vector(file)
        audio_features.append(feature_vector)     
    audio_features = np.array(audio_features)

    # label for each file, corresponding to the class
    labels = [int((x.split("\\")[-1]).split('-')[1]) for x in dataset_path_train]
    
    features_def = np.insert(audio_features, 0, labels, axis=1)

    # saving features 
    np.savetxt(first_output, features_def, delimiter=",")
    
    # the same for the testing part
   
    for fold in dataset_path_test: 
        audio_features_testing = []
        for file in fold: 
            feature_vector_testing = get_feature_vector(file)
            audio_features_testing.append(feature_vector_testing) 
        
        audio_features_testing = np.array(audio_features_testing)
    
        # label for each file, corresponding to the class
        labels_test = [int((x.split("\\")[-1]).split('-')[1]) for x in fold]
        # fold number for each file
        fold_number = int(np.unique([x.split("\\")[-2][4:] for x in fold]))
        
        features_def_testing = np.insert(audio_features_testing, 0, labels_test, axis=1)
        
        # saving features
        np.savetxt("test" + str(fold_number) + ".csv", features_def_testing, delimiter=",")
