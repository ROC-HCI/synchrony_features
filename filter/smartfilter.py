#!/usr/bin/env python
"""
------------------------------------------------------------------------
  loads a file using csv (each row represents time)
  applies a bandpass filter for each of specified columns
  generates a new .csv file fname.filtered.csv

  Tries to "intelligently" handle regions of nan.
  Some graphical debug utilities.

  --------------------------
  necessary csv file format:
     -the first row should be a header
     -each time sample should corresond to a row
     -each column represents a different dimension
     
  portions copied from:  
  https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
------------------------------------------------------------------------
"""
import glob
import os

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, lfilter
from scipy.signal import freqz

from enum import Enum    
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)
'''
If you want to set the logging level from a command-line option such as:
  --log=INFO
'''

#------------------
class SmartFilter:
    """ Wrapper class for easy filtering of signal including:
            low, high, bandpass, moving average, and threshold.
            Can process a file or directory.
    """
    
        
    def __init__(self):
        b = []
        a = []
        self.filter_type = ''
        self.apply_map = {
            'low'    :  self.apply_IIR,
            'high'   :  self.apply_IIR,
            'band'   :  self.apply_IIR,
            'ave'    :  self.apply_moving_ave,
            'thresh' :  self.apply_thresh
        }
    
    def apply(self,data):
        filtered_data = self.apply_map[self.filter_type](data)
        return filtered_data

    def init_band(self, lowcut_hz, highcut_hz, rate_hz, order=5):
        """ generate filter coefficients            
            lowcut, highcut = frequencies are in Hz.
            fs - samples per second 
        """        
        self.filter_type = 'band'
        self.rate_hz = rate_hz
        self.order = order
        nyq = 0.5 * self.rate_hz
        low = lowcut_hz / nyq
        high = highcut_hz / nyq
        self.b, self.a = butter(order, [low, high], btype='band')        

    def init_low(self, cutoff_frequency, fs, order=5):
        """ generate filter coefficients            
            lowcut, highcut = frequencies are in Hz.
            fs - samples per second 
        """        
        self.filter_type = 'low'
        self.rate_hz = fs
        self.order = order
        nyq = 0.5 * fs
        cut = cutoff_frequency / nyq
        self.b, self.a = butter(order, cut, btype='low')        
        
    def init_moving_ave(self, n):
        self.filter_type = 'ave'
        self.n = n

    def init_thresh(self, thresh):
        self.filter_type = 'thresh'
        self.thresh = thresh
        
    def apply_IIR(self, data):
        """ applies low, high, bandpass filter """
        filtered_data = lfilter(self.b, self.a, data)
        return filtered_data
        
    def apply_moving_ave(self, data):
        d = np.cumsum(data, dtype=float)
        d[self.n:] = d[self.n:] - d[:-self.n]
        filtered_data = d[self.n - 1:] / self.n            
        return filtered_data

    def apply_thresh(self, data):
        filtered_data =  (data > self.thresh).astype(float)           
        return filtered_data

    def plot(self):
        if(self.filter_type in ['low', 'high', 'band']):
            # Plot the frequency response for a few different orders.
            plt.figure(1)
            plt.clf()
            w, h = freqz(self.b, self.a, worN=2000)
    
            plt.subplot(2,1,1)
            plt.plot((self.rate_hz * 0.5 / np.pi) * w, abs(h), 
                     label="order = %d" % self.order)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')  
            
            angles = np.unwrap(np.angle(h))        
            plt.subplot(2,1,2)
            plt.plot(w, angles, 'g')
            plt.ylabel('Angle (radians)', color='g')
            plt.grid()
            plt.axis('tight')
    
            #plt.show()        
            
    def filter_file(self,fname):
        """ load a csv file """

    def __str__(self):
        s = "SmartFilter type: "
        s += self.filter_type
        return s

#============================================================================
class TestSmartFilter(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(self):
        """ runs once before ALL tests """
        print("\n...........unit testing class SmartFilter..................")
        #self.my_Crf = Crf()

    def setUp(self):
        """ runs once before EACH test """
        pass

    @unittest.skip
    def test_init(self):
        print("\n...testing init(...)")
        pass
    
    @unittest.skip
    def test_band(self):
        print("\n...testing bandpass filter(...)")
        
        my_filt = SmartFilter()

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 5000.0
        lowcut = 500.0
        highcut = 1250.0
        order =3
        
        # Plot the frequency response for a few different orders.
        plt.figure(1)
        plt.clf()
        my_filt.init_band(lowcut, highcut, fs, order)
        my_filt.plot()
        plt.show()
        
        # Filter a noisy signal.
        T = 0.05
        nsamples = T * fs
        t = np.linspace(0, T, nsamples, endpoint=False)
        a = 0.02
        f0 = 600.0
        x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
        x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
        x += a * np.cos(2 * np.pi * f0 * t + .11)
        x += 0.03 * np.cos(2 * np.pi * 2000 * t)
        plt.figure(2)
        plt.clf()
        plt.plot(t, x, label='Noisy signal')        

        y = my_filt.apply(x)
        plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
        plt.xlabel('time (seconds)')
        plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
    
        plt.show()
        
        pause = input('Press enter when complete: ')

    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class SimpleFilter complete..............\n")
        
#------------------------------------------------------------------------
def load_file(fname):
    """ loads a csv file and places into a 2D np string array.
        RETURNS:
           header
           data
    """
    
    logging.info('  loading file:' + fname)
    f = open(fname, 'rt')
    data = []
    try:
        reader = csv.reader(f)
        header = next(reader)[:-1] # Affectiva has a extra empty column
        for row in reader:
            # deal with Affectiva 'bug' - no space between time and first nan
            # eg: '0.0000nan' --> 0.0000, 'nan'
            if('nan' in row[0]):
                row.insert(1,'nan')
                row[0] = row[0][:-3] # chop off last three chars
            data.append(np.array(row[:-1])) # Affectiva has a extra empty column
            #data.append(row)
    finally:
        f.close()
    
    # data[time, datafield] # all data is strings, even numbers
    data = np.array(data) 
    print(data.shape)
    print(header)
    
    return header,data

#------------------------------------------------------------------------
def filter_dir():
    """ applies filter to all csv files in a directory """
    # TODO
    pass

#------------------------------------------------------------------------
def filter_file(fname):
    """ applies filter to given file and generates a new csv """

    my_filter = SmartFilter()
    lowcut  = 1
    highcut = 2
    fs = 17
    order = 6
    #my_filter.init_band(lowcut, highcut, fs, order)
    #my_filter.init_low(highcut, fs, order)
    my_filter.init_thresh(30)
    #my_filter.init_moving_ave(5)
    my_filter.plot()
    logging.info('filtering file:' + fname)
    header_list, data_string_ary_nd = load_file(fname)
    
    # CALL FILTER CODE FOR SPECIFIED COLUMNS
    for col in [20]: # 8 - pitch
        x = np.array(data_string_ary_nd[:,col],float)
        x2 = np.nan_to_num(x)
        t = np.array(data_string_ary_nd[:,0],float)
        # Sample rate and desired cutoff frequencies (in Hz).
        
        y = my_filter.apply(x2)

        start,stop = 195*15,195*15+400
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t[start:stop], x2[start:stop], label='original signal (%g Hz)')
        plt.xlabel('time (seconds)')
        #plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        
        plt.subplot(2,1,2)
        plt.plot(t[start:stop], y[start:stop], label='Filtered signal (%g Hz)')
        plt.xlabel('time (seconds)')
        #plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        
    
        plt.show()        

#------------------------------------------------------------------------
def do_all():
    pass

#=============================================================================
if __name__ == '__main__':
    filter_file('example/2016-03-16_10-05-49-922-annabanana.new.csv')
    
    #filter_file('example/out_avg.csv')