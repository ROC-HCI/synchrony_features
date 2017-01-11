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
 
        self.THRESH = 50.
    
    def calculate_sync(self, max_time_shift_=37):

        
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
                
        # find rising edges in X
        x_rise_cnt = 0 # the number of rising edges in X_thresh        
        x_rising_edges = [] # list of times where there is a rising edge
        x_last = 0
        for t, x in enumerate (self.X_thresh):
            # is it a rising edge
            if ( (x_last == 0) and (x == 1) ):
                x_rise_cnt += 1
                x_rising_edges.append(t)
 
                logging.debug('rising edge at t :' + str(t))
                logging.debug('rising edge cnt:' + str(x_rise_cnt))
            x_last = x
            
        # find rising edges in Y
        y_rise_cnt = 0
        y_rising_edges = []
        y_last = 0
        for t, y in enumerate (self.Y_thresh):
            # is it a rising edge
            if ( (y_last == 0) and (y == 1) ):
                y_rise_cnt += 1
                
                y_rising_edges.append(t)
 
                logging.debug('y rising edge at t :' + str(t))
                logging.debug('y rising edge cnt:' + str(y_rise_cnt))
            y_last = y
            
        # count y follows x
        shared_cnt = 0
        for t in x_rising_edges:
            window = self.Y_thresh[t:t+max_time_shift_]
            window_is_one = (window == 1) # just to make sure that nan does not give positive
            if np.any(window_is_one):
                shared_cnt += 1

        y_sync = float(shared_cnt) / x_rise_cnt


        # count x follows y
        shared_cnt = 0
        for t in y_rising_edges:
            window = self.X_thresh[t:t+max_time_shift_]
            window_is_one = (window == 1) # just to make sure that nan does not give positive
            if np.any(window_is_one):
                shared_cnt += 1

        x_sync = float(shared_cnt) / y_rise_cnt

        
        return x_sync, y_sync

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

        # TEST 1
        # TODO - make slightly more complex, add a few nans, verify that sync is correct
        X = np.array([0,.5, 60, 1, 5, 100, 23, 10])
        Y = np.array([0,.5, .7, 1, 5, 45, 23, 100])
        time = np.array([0,1,2,3,4,5,6,7])
        
        my_sync = Sync(X, Y)

        my_sync.THRESH = 5
        x_sync, y_sync = my_sync.calculate_sync()
        my_sync.plot()
        print(my_sync)
        
        #assert(x_sync == expected value)

        # TEST 2
        
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
    my_test.test_calculate_sync_simple()
    #my_test.test_calculate_sync()

#=============================================================================
if __name__ == '__main__':
    print('running main')
    do_all()