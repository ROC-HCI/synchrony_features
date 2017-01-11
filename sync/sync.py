#!/usr/bin/env python
"""
------------------------------------------------------------------------
  class for calculating the threshold time shift synchrony (TTSS).
  
  Tries to "intelligently" handle regions of nan.
  Some graphical debug utilities.
     
------------------------------------------------------------------------
"""
import glob
import os

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from enum import Enum    
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)
'''
If you want to set the logging level from a command-line option such as:
  --log=INFO
'''

#------------------
class Sync:
    """ Synchrony calculation  
    
    
    
    """
    
    def __init__(self, X_, Y_):
        self.SAMPLE_RATE = 15
        self.X = np.array(X_)
        self.Y = np.array(Y_)
        
        assert (len(X_) == len(Y_)), 'array sizes are different'
        
        self.X_thresh = []
        self.Y_thresh = []
        self.synchrony = 0.    
        self.x_rise_cnt = 0
        self.THRESH = 50.
    
    def calculate_sync(self, max_time_shift_=37):

        synchrony_value = 0.0
        self.x_rise_cnt = 0 # the number of rising edges in X_thresh
        
        # threshold X and Y
        #self.X_thresh = (X_ > THRESH_).astype(int)
        #self.Y_thresh = (Y_ > THRESH_).astype(int)

        self.X_thresh = np.zeros_like(self.X, float)
        for t, x in enumerate(self.X):
            if np.isnan(x): 
                self.X_thresh[t] = np.nan
            else:
                self.X_thresh[t] = x > self.THRESH
                
        
        self.Y_thresh = np.zeros_like(self.Y, float)
        for t, y in enumerate(self.Y):
            if np.isnan(y): 
                self.Y_thresh[t] = np.nan
            else:
                self.Y_thresh[t] = y > self.THRESH
                
        # cycle over rising edges in X
        x_last = 0
        for t, x in enumerate (self.X_thresh):
            # is it a rising edge
            if ( (x_last == 0) and (x == 1) ):
                self.x_rise_cnt += 1
 
                logging.debug('rising edge at t :' + str(t))
                logging.debug('rising edge cnt:' + str(self.x_rise_cnt))
            x_last = x
            
        # if there is positive region in Y increase count
        


        return synchrony_value

    def plot(self):
        # Sample rate and desired cutoff frequencies (in Hz).
        plt.figure(1)
        plt.clf()

        time = np.array(range(len(self.X)),float) / self.SAMPLE_RATE
        
        plt.subplot(221)
        plt.plot(time, self.X)
        plt.xlabel('Time (sec)')
        plt.title('X')
        plt.axhline(y=self.THRESH)
        plt.grid()

        plt.subplot(222)
        plt.plot(time, self.X_thresh)
        plt.fill_between(time, 0, self.X_thresh, where=self.X_thresh >= 0, facecolor='green', interpolate=True)
        plt.title('X_thresh')
        plt.grid()

        plt.subplot(223)
        plt.plot(time, self.Y)
        plt.xlabel('Time (sec)')
        plt.title('Y')
        plt.axhline(y=self.THRESH)
        plt.grid()

        plt.subplot(224)
        plt.plot(time, self.Y_thresh)
        plt.fill_between(time, 0, self.Y_thresh, where=self.Y_thresh >= 0, facecolor='green', interpolate=True)
        plt.title('Y_thresh')
        plt.grid()
        
        plt.tight_layout()
        plt.show()      


    def __str__(self):
        s = "Syncrony: "
        s += str(self.synchrony)
        return s

#============================================================================
class TestSync(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(self):
        """ runs once before ALL tests """
        print("\n...........unit testing class Sync..................")
        #self.my_Crf = Crf()

    def setUp(self):
        """ runs once before EACH test """
        pass

    @unittest.skip
    def test_init(self):
        print("\n...testing init(...)")
        pass
    
    def test_calculate_sync_simple(self):
        print("\n...testing calculate_sync(...)")
        
        X = np.array([0,.5, 60, 1, 5, 100, 23, 10])
        Y = np.array([0,.5, .7, 1, 5, 45, 23, 100])
        time = np.array([0,1,2,3,4,5,6,7])
        
        my_sync = Sync()

        
        synchrony = my_sync.calculate_sync(X, Y,THRESH_=5)
        my_sync.plot()
        print(my_sync)
        
        #print('Synchrony: ', synchrony)
        #pause = input( synchrony)
        
    def test_calculate_sync(self):
        print("\n...testing calculate_sync(...)")
        
        
        col_list = [20] # 8 - pitch; 20 = smile?

        header_list, I_data_str_ary_nd = load_file(
            'example/2016-03-16_10-05-49-922-I-T-annabanana.csv')
        X = np.array(I_data_str_ary_nd[:,20], float)        

        header_list, W_data_str_ary_nd = load_file(
            'example/2016-03-16_10-05-49-922-W-T-tarples.csv')

        Y = np.array(W_data_str_ary_nd[:,20],float)
        smaller_N = np.minimum(len(X), len(Y)) 
        
        my_sync = Sync(X[:smaller_N],Y[:smaller_N])
        my_sync.THRESH = 5
        
        synchrony = my_sync.calculate_sync()
        my_sync.plot()
        print(my_sync)
        
        #print('Synchrony: ', synchrony)
        #pause = input( synchrony)



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
           data[time, datafield]
    """
    
    logging.info('  loading file:' + fname)
    f = open(fname, 'rt')
    data = []
    try:
        reader = csv.reader(f)
        header = np.array(next(reader)[:-1]) # Affectiva has a extra empty column
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
def do_all():
    my_test = TestSync()
    my_test.test_calculate_sync()

#=============================================================================
if __name__ == '__main__':
    print('running main')
    do_all()