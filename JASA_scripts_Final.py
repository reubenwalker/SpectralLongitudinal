
'''
PSEUDOCODE
1. Define frequency constants for vowels, sustained tones, and repertoire.
2. Gather all .wav files in the current directory.
3. Classify files based on filename code:
   - Vowel glides
   - Triads
   - Messa di voce
   - Vaccai exercises: Manca, Avezzo, Come
4. Extract quietest 30ms segment from messa di voce recordings.
5. Select vowel, triad, and Avezzo files for analysis.
6. For each selected file:
   a. Load audio and extract metadata (duration, trial type, ID, date).
   b. Skip if duration is too short.
   c. Check for matching messa di voce data; skip if missing.
   
   e. Use middle portion of recordings with repetitions, otherwise use full.
   f. Set pitch frequency range based on trial type.
   g. Calculate pitch contour; skip if no valid pitch contour.
   h. Estimate background noise level from "silent" sections.
   i. If triad exercise, extract high sustained tone for analysis.
   j. Extract pitch stats: min, max, median, mean.
   k. Calculate separate LTAS for alpha ratio and H1H2LTAS.
   l. Calculate metrics (CPPs, alpha ratio, H1H2LTAS).
   m. Estimate SNR from "background noise"
   n. Calculate relative SPL compared to messa di voce.
   o. Compile all results into a dataframe.
7. Save all results to a file for merging with categorical database.
'''

#Necessary Libraries:
from parselmouth.praat import call
from parselmouth import Sound 
import scipy
from scipy import stats
import librosa
from scipy.signal import hilbert, butter, sosfilt, find_peaks, resample, find_peaks_cwt
from scipy.io.wavfile import read, write
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acovf, acf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pickle
from scipy import signal
from librosa.feature import rms
import os
import glob
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
import math
import random
import numpy as np
from matplotlib.lines import Line2D
import speechpy
import pyloudnorm as pyln

def extractMetadata(wavFilename, data, samplerate):
    #All filenames were moved to a folder in form of 0000&2025_10_01&test1
    duration = len(data)/samplerate #seconds
    trialNum = wavFilename.split('\\')[-1][-5]
    idNum = wavFilename.split('\\')[-1].split('&')[0]
    date = wavFilename.split('\\')[-1].split('&')[1]
    return duration, trialNum, idNum, date

def stereoCheck(data):
    if len(data.shape) > 1: #Convert to mono
        data = data[:,0]
    return data

def get_nonzero_bounds(arr):
#Return first and last nonzero indices of pitched segment
    arr = np.asarray(arr)
    nonzero_indices = np.where(arr != 0)[0]
    if nonzero_indices.size == 0:
        return None, None  # or raise an error if needed
    return nonzero_indices[0], nonzero_indices[-1]


def get_quietest_edge_rms(audio, sr, segment_duration=0.1):
    ###Normalization to quietest portion of messa di voce segment
    segment_samples = int(segment_duration * sr)
    start_segment = audio[:segment_samples]
    end_segment = audio[-segment_samples:]

    start_rms = np.sqrt(np.mean(start_segment**2))
    end_rms = np.sqrt(np.mean(end_segment**2))

    return min(start_rms, end_rms)
    
def normalize_to_target_rms(audio, target_rms):
    audio_rms = np.sqrt(np.mean(audio**2))
    if audio_rms == 0:
        return audio
    return audio * (target_rms / audio_rms)

 
def loudness_normalize(audio, sr, target_lufs=-23.0):
    #Potential alternative to messa di voce normalization
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
    return normalized_audio, loudness

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def selectMiddleTrial(wavFilename):
    #For exercises 1-4 where multiple repetitions of task are present
    #load wav file.
    samplerate, data = read(wavFilename)
    #Load audio file
    #If audio is stereo, take the left channel
    try:
        if data.shape[1] > 1:
            data = data[:,0]
    except:
        pass
    #There are some local zeros. Take rolling average and find max closest to midpoint
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    #Need a big moving average to not cut out mid-trial lows.
    #Let's use a half-second moving average (samplerate/2)
    a = moving_average(amplitude_envelope, n=round(samplerate/2))
    #Try to find larger peaks
    #Set all values where the amplitude_envelope is a factor of 2^3 lower to zero (-3dB)
    #Why -3 dB? Had one error, so let's lower the threshold to -4 dB
    a1 = np.where(a < a.max()/np.power(2,3), 0, a) 
    #Could be a problem with clipping^
    
    ###Find the amplitude envelope closest to the midpoint.
        #Could be a nonzero section with dB >-3 of max or a section closest to it.
        #If you find the sections where amplitude envelope != 0, you have an array of x tests
        #Choose the middle one.
    peaks = find_peaks(a1)
    #Calculate the distance from each to the midpoint
    distance_from_midpoint = abs(peaks[0] - round(len(a1)/2))
    minIndex = peaks[0][np.argmin(distance_from_midpoint)]
    #Ok, that gives us the local max closest to the midpoint.
    #Set zeros of audio intensity to -4 dB below this point.
    a = np.where(a < a[minIndex]/np.power(2,4), 0, a)
    #How do we find the section of the signal around that local max?
    #First, find zero prior to that point.
    if np.where(a[:minIndex] == 0)[0].size != 0:
        startMiddleAttempt = np.where(a[:minIndex]==0)[0][-1]
    else:
        startMiddleAttempt = 0  
    
    #Then find zero after that point. 
        #In one case, the recording ends before a zero.
    if np.where(a[minIndex:] == 0)[0].size != 0:
        finishMiddleAttempt = minIndex + np.where(a[minIndex:]==0)[0][0]
    else:
        finishMiddleAttempt = len(data)
    selectedMiddleTrial = data[startMiddleAttempt:finishMiddleAttempt]
    #visualCheckSelection(data, startMiddleAttempt, finishMiddleAttempt)
    #prompt = input("Press Enter to continue...")
    #if #prompt == 'q':
    #    break
    return samplerate, selectedMiddleTrial    

def contourCheck(pitch_contour, f_s_contour):
    if type(pitch_contour) == float:
        return np.nan, np.nan
    pitchDF = pd.DataFrame(pitch_contour)
    if pitchDF[pitchDF[0].notna()].shape[0]/f_s_contour < 2:
        return np.nan, np.nan
    else:
        return pitch_contour, pitchDF

def returnPitchedSegment(data, samplerate):
    sound = Sound(data, samplerate)
    #Creates PRAAT sound file from .wav array
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #create a praat pitch object
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency

    #Select only the pitched portion of the audio:
    start, end = get_nonzero_bounds(pitch_contour)
    i_0 = int(np.ceil(start*samplerate/f_s_contour))
    i_f = int(np.floor(end*samplerate/f_s_contour))
    pitched_messa = data[i_0:i_f]
    return pitched_messa

def estimateNoise(data, samplerate, pitchDF, f_s_contour):
    #Find longest non-pitched segment at beginning/end of pitch contour
    i_nonZero = (pitchDF[0].notna()).argmax()
    #Check the end of the file as well
    i_flipped = (np.flip(pitchDF[0]).notna()).argmax()
    beg0 = int(np.floor(i_nonZero*samplerate/f_s_contour*0.25))
    begF = int(np.floor(i_nonZero*samplerate/f_s_contour*0.75))
    begFrame = int(np.floor(i_nonZero*samplerate/f_s_contour))
    
    end0 = int(np.floor(i_flipped*samplerate/f_s_contour*0.25))
    endF = int(np.floor(i_flipped*samplerate/f_s_contour*0.75))
    endFrame = int(np.floor(i_flipped*samplerate/f_s_contour))
    #Find "quieter" noise (trying to avoid clipping)
    
    #Compare signal power
    begPow = np.mean(data[beg0:begF]**2)
    endPow = np.mean(data[-endF:-end0]**2)

    if ((begPow > endPow) & (endFrame != 0)):
        noisy_part = data[-endF:-end0]
        # plt.close()
        # plt.plot(np.arange(len(data)),data)
        # plt.plot(np.arange(len(data)-endF,len(data)-end0),data[-endF:-end0] )
    elif ((begPow < endPow) & (begFrame != 0)):
        noisy_part = data[beg0:begF]
        # plt.close()
        # plt.plot(np.arange(len(data)), data)
        # plt.plot(np.arange(beg0,begF), data[beg0:begF])
    elif i_nonZero > i_flipped:
        noisy_part = data[beg0:begF] 
        # plt.close()
        # plt.plot(np.arange(len(data)), data)
        # plt.plot(np.arange(beg0,begF), data[beg0:begF])
    else:
        noisy_part = data[-endF:-end0]
        # plt.close()
        # plt.plot(np.arange(len(data)),data)
        # plt.plot(np.arange(len(data)-endF,len(data)-end0),data[-endF:-end0] )

    duration_n = len(noisy_part)/samplerate
    return noisy_part, duration_n


def skCategorize(df_notna, vowelNum=5):
    #Categorize vowels based on formants 1-3 using k nearest means
    prep = MaxAbsScaler()
    kmeans = KMeans(n_clusters=vowelNum, random_state=0)

    scaled_data = prep.fit_transform(df_notna)
    kmeans.fit(scaled_data)

    df_notna['label'] = kmeans.labels_
    labelOrder = df_notna.groupby('label')['t'].median().sort_values().index
    if vowelNum == 5:
        vowels = ['a', 'e', 'i', 'o', 'u']

    if vowelNum == 3:
        vowels = ['a', 'e_i', 'o_u']
    vowelDict = {}        
    for i in range(vowelNum):
        vowelDict[labelOrder[i]] = vowels[i]
    df_notna['vowel'] = df_notna['label'].apply(lambda x: vowelDict[x])
    return df_notna


def skVowelSD(data, samplerate, freqLow, freqHigh):
    # Calculate formants 1-3 for vowel glide 
    sound = Sound(data, samplerate)

    f0min=freqLow
    f0max=freqHigh
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    t_list = []

    for point in range(0, numPoints):
        point +=1
        t = call(pointProcess, "Get time from index", point)
        t_list.append(t)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    df_f = pd.DataFrame(np.array([t_list, f1_list, f2_list, f3_list]).T, columns=['t', 'f1', 'f2', 'f3'])
    df_f['f1'] = df_f['f1'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    df_f['f2'] = df_f['f2'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    df_f['f3'] = df_f['f3'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    
    #Remove nans
    df_f = df_f[(df_f['f1'].notna()) & (df_f['f2'].notna()) & (df_f['f3'].notna())]
    
    # Categorize sections /a/, /ei/ and /ou/
    df_f = skCategorize(df_f, vowelNum=3)

    #Goal: find three longest unbroken segments of a, e_i, and o_u
    resultDF = pd.DataFrame({}, columns=['a', 'e_i', 'o_u'],index=['Max', 'Index'])
    for i in ['a', 'e_i', 'o_u']:
        #print(i)
        mask = df_f['vowel'] == i
        try:
        #Find longest stretch of given vowel
            resultDF.loc['Max', i] = (~mask).cumsum()[mask].value_counts().max()
            resultDF.loc['Index',i] = (~mask).cumsum()[mask].value_counts().index[0]
        except IndexError:
            #print('oh no!')
            return np.nan, np.nan, np.nan
    i_a0 = resultDF.loc['Index', 'a']
    i_af = i_a0 + resultDF.loc['Max', 'a']
    quartSpan_a = round((i_af - i_a0)/4)
    i_a0 = i_a0 + quartSpan_a
    i_af = i_af - quartSpan_a
    i_ei0 = resultDF.loc['Index', 'e_i']
    i_eif = i_ei0 + resultDF.loc['Max', 'e_i']
    quartSpan_ei = round((i_eif - i_ei0)/4)
    i_ei0 = i_ei0 + quartSpan_ei
    i_eif = i_eif - quartSpan_ei
    i_ou0 = resultDF.loc['Index', 'o_u']
    i_ouf = i_ou0 + resultDF.loc['Max', 'o_u']
    quartSpan_ou = round((i_ouf - i_ou0)/4)
    i_ou0 = i_ou0 + quartSpan_ou
    i_ouf = i_ouf - quartSpan_ou

    #audio frames
    x_a0 = round(df_f.loc[i_a0,'t']*samplerate)
    x_af = round(df_f.loc[i_af, 't']*samplerate)
    x_ei0 = round(df_f.loc[i_ei0,'t']*samplerate)
    x_eif = round(df_f.loc[i_eif,'t']*samplerate)
    x_ou0 = round(df_f.loc[i_ou0,'t']*samplerate)
    x_ouf = round(df_f.loc[i_ouf,'t']*samplerate)
    
    data_a = data[x_a0:x_af]
    
    # plt.close('all')
    # for i in range(1,4):
        # plt.plot(df_f['t'], df_f['f'+str(i)], color='black')

    # colors = ['blue', 'green', 'orange', 'purple', 'yellow']
    # vowels = df_f.groupby('vowel')['label'].first().index
    # for i in range(3):
        # mask = df_f['vowel'] == vowels[i]
        # for j in range(1,4):
            # plt.plot(df_f.loc[mask, 't'], df_f.loc[mask, 'f'+str(j)], 'o' , color=colors[i])
            # plt.text(df_f.loc[mask, 't'].median(), 1500, vowels[i], horizontalalignment='center')
    # for k in range(1,4): 
        # plt.plot(df_f.iloc[i_a0:i_af]['t'], df_f.iloc[i_a0:i_af]['f'+str(k)], 'o', color='red')
    # plt.show()
    
    return data_a
    

def calcPitchContour(data, samplerate, freqLow=60, freqHigh=700,gender=np.nan):
    #Reutnr pitch contour and effective sampling frequency of pitch contour
    sound = Sound(data, samplerate)
    pitch = call(sound, "To Pitch", 0.1, freqLow, freqHigh) #c4-g5 
    pitch_contour = pitch.selected_array['frequency']
    if pitch_contour[pitch_contour != 0].size == 0:
        return np.nan, np.nan#, np.nan, np.nan
    pitchContLength = pitch.selected_array['frequency'].size
    wavLength = len(data)
    f_s_contour = pitchContLength/wavLength*samplerate
    ###Pitch contour correction
        #From https://praat-users.yahoogroups.co.narkive.com/swLrgWcR/eliminate-octave-jumps-and-annotate-pitch-contour
    q1 = call(pitch, "Get quantile", 0.0, 0.0, 0.25, "Hertz")
    q3 = call(pitch, "Get quantile", 0.0, 0.0, 0.75, "Hertz")
    floor = q1*0.75
    ceiling = q3*2
    pitchCorrect = call(sound, "To Pitch", 0.1, floor, ceiling) #c4-g5 
    contourCorrect = pitchCorrect.selected_array['frequency']
    #plt.plot(pitch_contour)
    #plt.plot(contourCorrect)
    contourCorrect[contourCorrect == 0] = np.nan
    return contourCorrect, f_s_contour
    
def isolateHighestPitch2024(f_s_Audio, selectedMiddleTrial, pitch_contour, f_s_contour):
    #So we have an interval of a minor third between the highest note and the middle note. 
    ###Ok, now we want to find the corresponding interval in the pitch_contour array    
        #That are within a minor third of the maximum pitch. 
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)  
    #Let's take the middle fifty percent of this interval.

    begin50 = beginInterval + round((endInterval - beginInterval)*.25)
    end50 =  beginInterval + round((endInterval - beginInterval)*.75)

    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt

    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt  
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return f_s_Audio, middleFiftyPercentHighestPitch, maxFreq, meanFreq
    
def LTAS_5000(data, samplerate, freqResolution=400):
    #Return long-term average spectrum from 0-5000 Hz
    #We want a frequency resolution of 400 Hz
    #This is 400 Hz = samplerate/nperseg

    n = round(samplerate/freqResolution)

    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', nperseg=n, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)

    idx1 = (f<5000)
    mask = np.where(idx1)
    LTASarray = np.array([f[mask],RMS[mask]])

    return LTASarray
    
def freqExtract(data, samplerate, freqLow=60, freqHigh=700):
    # Extract frequency min, max, median, and mean
    sound = Sound(data, samplerate)
    pitch = call(sound, "To Pitch", 0.1, freqLow, freqHigh) #c4-g5 
    pitch_contour = pitch.selected_array['frequency']
    if pitch_contour[pitch_contour != 0].size == 0:
        return np.nan, np.nan, np.nan, np.nan
    pitchContLength = pitch.selected_array['frequency'].size
    wavLength = len(data)
    f_s_contour = pitchContLength/wavLength*samplerate
    ###Pitch contour correction
        #From https://praat-users.yahoogroups.co.narkive.com/swLrgWcR/eliminate-octave-jumps-and-annotate-pitch-contour
    q1 = call(pitch, "Get quantile", 0.0, 0.0, 0.25, "Hertz")
    q3 = call(pitch, "Get quantile", 0.0, 0.0, 0.75, "Hertz")
    floor = q1*0.75
    ceiling = q3*2
    pitchCorrect = call(sound, "To Pitch", 0.1, floor, ceiling) #c4-g5 
    contourCorrect = pitchCorrect.selected_array['frequency']
    #plt.plot(pitch_contour)
    #plt.plot(contourCorrect)
    contourCorrect[contourCorrect == 0] = np.nan
    neighbor = contourCorrect[1:]/contourCorrect[:-1]
    idx = neighbor > 1.5
    mask = np.where(idx)
    #Eliminate octave jumps?
    # contourCorrect[1:][mask] = contourCorrect[1:][mask]/2
    #plt.plot(contourCorrect)
    f_0_min = np.nanmin(contourCorrect)
    f_0_max = np.nanmax(contourCorrect)
    f_0_mean = np.nanmean(contourCorrect)
    f_0_median = np.nanmedian(contourCorrect)
    return f_0_min, f_0_max, f_0_median, f_0_mean
    
def calcAlphaRatio(f, RMS):
    # Calculate ratio in decibels of summed energy in LTAS above 1000 Hz to below 1000 Hz
    idx0 = (f<1000)
    mask0 = np.where(idx0)
    idx1 = (f>1000)
    mask1 = np.where(idx1)
    #Alpha Ratio is high/low energy
    #This needs to be converted to decibels
    alphaRatio = 10*np.log10(RMS[mask1].sum()/RMS[mask0].sum())
    return alphaRatio # decibels

def calcH1H2LTAS_thresh(f, RMS, f_0_min, f_0_max):
    #Comparing energy in the energy band of the f_0 range
        #to the energy an octave above
    #Let's just compare frequency bands from [f_0_max/2, f_0_max]
    if f_0_min < f_0_max/2:
        f_0_min = f_0_max/2
    idx0 = (f>f_0_min)*(f<f_0_max)
    mask0 = np.where(idx0)
    idx1 = (f>f_0_min*2)*(f<f_0_max*2)
    mask1 = np.where(idx1)
    #Since we're converting Root-Mean-Squared, a power measure, we use 10*log
    #Need to use the AVERAGE of the levels in these bands, not the sum
    H1H2LTAS = 10*np.log10(RMS[mask0].mean()/RMS[mask1].mean())
    return H1H2LTAS # dB

def calcF1_Proportion(f, RMS, f_0_min, f_0_max):
    #Comparing energy in the energy band of the f_1 range
        #to all other bands
    #Let's just compare frequency bands from [f_0_max/2, f_0_max]
    if f_0_min < f_0_max/2:
        f_0_min = f_0_max/2
    idx0 = (f>f_0_min)*(f<f_0_max)
    mask0 = np.where(idx0)
    # idx1 = (f>f_0_min*2)*(f<f_0_max*2)
    mask1 = np.where(~idx0)
    #Since we're converting Root-Mean-Squared, a power measure, we use 10*log
    #Need to use the AVERAGE of the levels in these bands, not the sum
    H1H2LTAS = 10*np.log10(RMS[mask0].mean()/RMS[mask1].mean())
    return H1H2LTAS # dB

def sustainedH1H2LTAS_CPPs(pitch_contour, f, RMS, medComp, data, samplerate):
    # Calculate octaves of H1H2LTAS around sustained F0, then calculate H1H2LTAS and CPPs
    #Calculate whether pitch is closer to possible pitches in medComp
    #Pitch contour has to be np.nan cleaned
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]
    medArray = np.abs(medComp - np.median(cleanedContour))
    keyGuess = medComp[medArray.argmin()]
    # print(keyGuess)
    root2 = np.sqrt(2)
    #Calculate the octave around the pitch
    f0_min = keyGuess/root2
    f0_max = keyGuess*root2
    H1H2LTAS = calcH1H2LTAS_thresh(f,RMS, f0_min, f0_max)#keyGuess, 2*keyGuess)
    if medArray.argmin() < 2:
        f1_guess = f0_min*2
    else:
        f1_guess = f0_min
    f1_prop = calcF1_Proportion(f,RMS, f1_guess, 2*f1_guess)
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
   
    return H1H2LTAS, cpps, keyGuess, f1_prop

def avezzoH1H2LTAS_CPPs(pitch_contour, f, RMS, freqComp, data, samplerate):
    #Calculate which possible octave fits the pitch contour best, then calculate H1H2LTAS and CPPs
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]
    d, dis, d1, dis1, d2, dis2 = freqComp
    
    #See which portion of contour is found between pitches
    d_mask = (cleanedContour > d) & (cleanedContour < d1)
    d1_mask = (cleanedContour > d1) & (cleanedContour < d2)
    dis_mask = (cleanedContour > dis) & (cleanedContour < dis1)
    dis1_mask = (cleanedContour > dis1) & (cleanedContour < dis2)
    pLen = cleanedContour.shape[0]
    #Calculate percentage of contour found between those pitches
    dPerc = cleanedContour[d_mask].shape[0]/pLen
    d1Perc = cleanedContour[d1_mask].shape[0]/pLen
    disPerc = cleanedContour[dis_mask].shape[0]/pLen
    dis1Perc = cleanedContour[dis1_mask].shape[0]/pLen

    keyArray = np.array([dPerc, disPerc, d1Perc, dis1Perc])
    freqArray = np.array([d, dis, d1, dis1])
    keyGuess = freqArray[keyArray.argmax()]

    f0_min = keyGuess
    f0_max = keyGuess*2
    H1H2LTAS = calcH1H2LTAS_thresh(f,RMS, keyGuess, 2*keyGuess)
    if keyArray.argmax() < 2:
        f1_guess = f0_min*2
    else:
        f1_guess = f0_min
    f1_prop = calcF1_Proportion(f,RMS, f1_guess, 2*f1_guess)
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    
    return H1H2LTAS, cpps, keyGuess, f1_prop

def calcRelativeSPL(data, target_RMS):
    # Calculating dSPL difference from quietest sung segment of messa di voce
    RMS_data = np.sqrt(np.mean(data ** 2))
    dSPL = 20*np.log10(RMS_data/target_RMS)
    return dSPL

def calcSNR_simplest(signal, noise):
    #Simple SNR estimate
    powS = np.mean(signal**2)
    powN = np.mean(noise**2)
    return 10*np.log10((powS-powN)/powN)

def compileResultDict(
    idNum, date, trialNum, duration, samplerate,
    duration_n=np.nan,
    f_0_max=np.nan, f_0_mean=np.nan, f_0_min=np.nan, f_0_median=np.nan,
    keyGuess=np.nan, CPPs=np.nan, alphaRatio=np.nan, H1H2LTAS=np.nan,
    SNR=np.nan, dSPL=np.nan, f1_prop=np.nan, no_Messa=False, no_Pitch=False):
    # Return calculated values in dataframe form
    return {
        'id': idNum,
        'date': date,
        'trialNum': trialNum,
        'duration': duration,
        'sr': samplerate,
        'duration_noise': duration_n,
        'pitchMax': f_0_max,
        'pitchMean': f_0_mean,
        'pitchMin': f_0_min,
        'pitchMed': f_0_median,
        'keyGuess': keyGuess,
        'CPPs': CPPs,
        'alphaRatio': alphaRatio,
        'H1H2LTAS': H1H2LTAS,
        'SNR': SNR,
        'dSPL': dSPL,
        'f1_prop':f1_prop,
        'no_Messa':no_Messa,
        'no_Pitch':no_Pitch
    }

def extractQuietestSamples(messaSamples, df):
    for wavFilename in messaSamples:
        data, samplerate = librosa.load(wavFilename, sr=None, mono=False)
        
        #Extract metadata from filename
        duration, trialNum, idNum, date = extractMetadata(wavFilename, data, samplerate)

        if duration < 2:
            duration = np.nan
            resultDict = {'id':idNum, 
                          'date':date,
                          'trialNum':trialNum,
                          'duration':duration
                          }
            df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
            continue
        
        data = stereoCheck(data)
        
        
        ### Test: Loudness normalize data to -23 Lufs
        data, __ = loudness_normalize(data, samplerate)
        
        #Select Middle Trial if there are repetitions of 
        if (duration > 20) & (trialNum != '6'):
            samplerate, middleTrial = selectMiddleTrial(wavFilename)
        else:
            middleTrial = data

        #Return sung segment of messa di voce
        pitched_messa = returnPitchedSegment(middleTrial, samplerate)

        #Select the quietest 30 ms portion of the messa di voce
        target_rms = get_quietest_edge_rms(pitched_messa, samplerate, segment_duration=0.03) # 30 ms 
        
        resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      'date':wavFilename.split('\\')[-1].split('&')[1],
                      'trialNum':wavFilename.split('\\')[-1][-5],
                      'ref_RMS':target_rms
                     }
        #df = df.append(resultDict, ignore_index=True)
        df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
    return df

def main():

    # Define necessary constants:
    #What are our frequency thresholds for avezzo?
    #DDur / DisDur
    d = 146.8324
    dis = 155.5635
    d1 = 293.6648
    dis1 = 311.1270
    d2 = 587.3295
    dis2 = 622.2540

    #For sustained fifth
    e1 = 329.6276
    e2 = 659.2551

    #For vowel
    g = 195.9977
    g1 = 391.9954
    a = 220.0000
    a1 = 440

    root2 = np.sqrt(2)

    mediumMedianFreq = np.array([g,a,g1,a1])

    highMedianFreq = np.array([d1,e1,d2,e2])
    
    repertoireFreq = np.array([d,dis,d1,dis1,d2,dis2])

    #All wavFiles are stored in a single folder with metadata in the filename
    path = os.getcwd()
    wav_files = glob.glob(os.path.join(path, "*.wav*"))#"*.xlsx"))

    vokalFiles = []
    dreiklangFiles = []
    messaFiles = []
    mancaFiles = []
    avezzoFiles = []
    comeFiles = []

    for i in wav_files:
        if i[-5] == '1':
            #Vowel Glide
            vokalFiles.append(i)
        if i[-5] == '2':
            #Ascending-Descending Triad
            dreiklangFiles.append(i)
        if i[-5] == '4':
            #Messa di voce
            messaFiles.append(i)
        if i[-5] == '5':
            #Vaccai Manca sollecita
            mancaFiles.append(i)
        if i[-5] == '6':
            #Vaccai Avezzo a vivere
            avezzoFiles.append(i)
        if i[-5] == '7':
            #Vaccai come il candore
            comeFiles.append(i)
            
    ### Here we collect all messa di voce files (crescendo-decrescendo)
    messaSamples =  messaFiles 
    #indexArray = df.index
    df = pd.DataFrame({})

    ### Store the quietest 30ms RMS amplitude of messa di voce phonation in our dataframe:
    df = extractQuietestSamples(messaSamples, df)

    ### We want to analze medium sustained phonation, high sustained phonation and the repertoire sample.
    analysisSamples =  vokalFiles + dreiklangFiles + avezzoFiles #

    for wavFilename in analysisSamples:
        data, samplerate = librosa.load(wavFilename, sr=None, mono=False)
        
        duration, trialNum, idNum, date = extractMetadata(wavFilename, data, samplerate)
        if duration < 2:
            duration = np.nan
            resultDict = compileResultDict(idNum, date, trialNum, duration, samplerate)
            df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
            continue

        data = stereoCheck(data)
        
        ###Let's amplitude normalize:
        #If we don't have an RMS from the messa di voce, we'll exclude the sample
        if df[((df['id'] == idNum) & (df['date'] == date) & (df['trialNum'] == '4'))].size == 0:
            resultDict = compileResultDict(idNum, date, trialNum, duration, samplerate, no_Messa=True)
            df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
            continue

        ### Calculate quietest RMS of messa di voce:
        target_RMS = df[((df['id'] == idNum) & (df['date'] == date) & (df['trialNum'] == '4'))]['ref_RMS'][0]
        # data = normalize_to_target_rms(data, target_RMS)
        
        ### Test: Loudness normalize data to -23 Lufs
        # data, __ = loudness_normalize(data, samplerate)
        
        #Select Middle Trial if medium or high sustained recording has repetitions:
        if (duration > 20) & (trialNum != '6'):
            samplerate, middleTrial = selectMiddleTrial(wavFilename)
        else:
            middleTrial = data

        # Identify high and low thresholds for pitch contours within half octave of min/max possible
        if trialNum == '1':
            fLow = g/root2
            fHigh = a1*root2
        if trialNum == '2':
            # fLow = d1/root2
            fLow = g/root2
            fHigh = e2*root2

        if trialNum == '6':
            # fLow = d1/root2
            fLow = 60
            fHigh = e2*root2

        #Calculate pitch contour
        pitch_contour, f_s_contour = calcPitchContour(middleTrial, samplerate)
        

        # If no pitch contour is found, return np.nan, otherwise convert to dataframe
        pitch_contour, pitchDF = contourCheck(pitch_contour, f_s_contour)
        if type(pitch_contour) == float:
            resultDict = compileResultDict(idNum, date, trialNum, duration, samplerate, no_Pitch=True)
            df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
            continue                      

        #Estimate noise from beginning and end of recording
        noisy_part, duration_n = estimateNoise(data, samplerate, pitchDF, f_s_contour)
        
        if trialNum == '1':
            #Analyze data as-is.
            data = middleTrial
        if trialNum == '2':
            # Extract middle 50% of sustained fifth 
            samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch2024(samplerate, middleTrial, pitch_contour, f_s_contour)
            data = highestPitch
            if trialNum == '2':
                fLow = d1/root2
                fHigh = e2*root2
            pitch_contour, f_s_contour = calcPitchContour(data, samplerate, freqLow=fLow, freqHigh=fHigh)
            pitch_contour, pitchDF = contourCheck(pitch_contour, f_s_contour)
            
            if type(pitch_contour) == float:
                resultDict = compileResultDict(idNum, date, trialNum, duration, samplerate, no_Pitch=True)
                df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
                continue      
        
        #Extract min/max/median/mean freq info
        f_0_min, f_0_max, f_0_median, f_0_mean = freqExtract(data, samplerate, freqLow=fLow, freqHigh=fHigh)
        
        # Calculate LTAS for alpha ratio (400 Hz freq bands)
        LTASarray = LTAS_5000(data, samplerate, freqResolution=400)
        f = LTASarray[0]
        RMS = LTASarray[1]
        # Calculate LTAS for H1H2LTAS (30 Hz freq bands)
        LTASarray_H1H2 = LTAS_5000(data, samplerate, freqResolution=30)
        f_H1H2 = LTASarray_H1H2[0]
        RMS_H1H2 = LTASarray_H1H2[1]
        
        # Calculate Alpha Ratio
        alphaRatio = calcAlphaRatio(f, RMS)
        
        if trialNum == '2':
            #Calculate H1H2LTAS and CPPs for High Sustained Phonation
            H1H2LTAS, CPPs, keyGuess, f1_prop = sustainedH1H2LTAS_CPPs(pitch_contour, f_H1H2, RMS_H1H2, highMedianFreq, data, samplerate)
            
        if trialNum == '1':
            #Calculate H1H2LTAS and CPPs for Medium Sustained Phonation
            H1H2LTAS, CPPs, keyGuess, f1_prop = sustainedH1H2LTAS_CPPs(pitch_contour, f_H1H2, RMS_H1H2, mediumMedianFreq, data, samplerate)

        if trialNum == '6':
            #Calculate H1H2LTAS and CPPs for Repertoire Sample
            H1H2LTAS, CPPs, keyGuess, f1_prop = avezzoH1H2LTAS_CPPs(pitch_contour, f_H1H2, RMS_H1H2, repertoireFreq, data, samplerate)
        
        #Estimate SNR using extracted "silence"
        SNR = calcSNR_simplest(data, noisy_part)
        
        # Calculate relative SPL to quietest 30s from messa di voce
        dSPL = calcRelativeSPL(data, target_RMS)

        resultDict = compileResultDict(idNum, date, trialNum, duration, samplerate,
                                       duration_n,
                                       f_0_max, f_0_mean, f_0_min, f_0_median,
                                       keyGuess, CPPs, alphaRatio, H1H2LTAS,
                                       SNR, dSPL, f1_prop)

        df = pd.concat([df, pd.DataFrame.from_records([resultDict])])

        plt.close('all')
        df.to_pickle('finalJASA20250616.pkl')
        df.to_csv('finalJASA20250616.csv')

# Standard Python entry point:
if __name__ == "__main__":
    main()