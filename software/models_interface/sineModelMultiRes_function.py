# function to call the main analysis/synthesis functions in software/models/sineModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sineModel as SM

def main(inputFile='../../sounds/orchestra.wav',
  w1='hamming', w2='blackmanharris', w3='blackmanharris',
  M1=4095, M2=513, M3=513,
  N1=4096, N2=1024, N3=1024,
  B1=0, B2=1000, B3=5000,
  t=-80):
  """
  Perform analysis/synthesis using the sinusoidal model
  inputFile: input sound file (monophonic with sampling rate of 44100)
  window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
  M1, M2, M3: analysis window sizes (low, mid, high bands);
  N1, N2, N3: fft sizes per band (powers of two, bigger or equal than Ms);
  B1, B2, B3: lower freq bound per band;
  t: magnitude threshold of spectral peaks
  """

  # read input sound
  fs, x = UF.wavread(inputFile)

  # compute analysis windows
  w1 = get_window(window, M1)
  w2 = get_window(window, M2)
  w3 = get_window(window, M3)

  # analyze/synthesize
  y = SM.sineModelMultiRes(x, fs, t, w1, w2, w3, N1, N2, N3, B1, B2, B3)

  # output sound file name
  outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModelMultiRes.wav'

  # write the synthesized sound obtained from the sinusoidal synthesis
  UF.wavwrite(y, fs, outputFile)

  # create figure to show plots
  plt.figure(figsize=(12, 9))

  # frequency range to plot
  maxplotfreq = 5000.0

  # plot the input sound
  plt.subplot(3,1,1)
  plt.plot(np.arange(x.size)/float(fs), x)
  plt.axis([0, x.size/float(fs), min(x), max(x)])
  plt.ylabel('amplitude')
  plt.xlabel('time (sec)')
  plt.title('input sound: x')

  # plot the output sound
  plt.subplot(3,1,3)
  plt.plot(np.arange(y.size)/float(fs), y)
  plt.axis([0, y.size/float(fs), min(y), max(y)])
  plt.ylabel('amplitude')
  plt.xlabel('time (sec)')
  plt.title('output sound: y')

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()
