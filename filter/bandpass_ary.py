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
------------------------------------------------------------------------
"""
import glob
import os

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, lfilter

import logging

logging.basicConfig(level=logging.DEBUG)
'''
If you want to set the logging level from a command-line option such as:
  --log=INFO
'''



#-------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#------------------
class SmartFilter:
    """ """
    def __init__(self):
        self.D = []     # np.array of data
        self.fname = '' # datafile
        
    def __str__(self):
        return "SmartFilter"    

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

    logging.info('filtering file:' + fname)
    header_list, data_string_ary_nd = load_file(fname)
    
    # CALL FILTER CODE
    for col in [20]: # 8 - pitch
        x = np.array(data_string_ary_nd[:,col],float)
        x2 = np.nan_to_num(x)
        t = np.array(data_string_ary_nd[:,0],float)
        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 20.0
        lowcut = 0.5
        highcut = 5.0
        
        y = butter_bandpass_filter(x2, lowcut, highcut, fs, order=6)

        start,stop = 195*15,195*15+400
        plt.subplot(1,2,1)
        plt.plot(t[start:stop], x2[start:stop], label='original signal (%g Hz)')
        plt.xlabel('time (seconds)')
        #plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        
        plt.subplot(1,2,2)
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