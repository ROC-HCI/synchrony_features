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
    
    def __init__(self):
        self.SAMPLE_RATE = 15
        self.X = []
        self.Y = []
        self.X_thresh = []
        self.Y_thresh = []
        self.synchrony = 0.    
        self.x_rise_cnt = 0
    
    
    def calculate_sync(self,X_, Y_, THRESH_=50, max_time_shift_=37):

        synchrony_value = 0.0
        self.x_rise_cnt = 0 # the number of rising edges in X_thresh
        self.X = np.array(X_)
        self.Y = np.array(Y_)
        
        # threshold X and Y
        self.X_thresh = (X_ > THRESH_).astype(int)
        self.Y_thresh = (Y_ > THRESH_).astype(int)
                
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
        
        plt.subplot(211)
        plt.plot(time, self.X)
        plt.xlabel('Time (sec)')
        plt.title('X')
        plt.grid()

        plt.subplot(212)
        plt.scatter(time, self.X_thresh)
        plt.title('X_thresh')
        plt.grid()
        
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
    
    def test_calculate_sync(self):
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


    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class SimpleFilter complete..............\n")
        
#------------------------------------------------------------------------

    
    
#------------------------------------------------------------------------
def do_all():
    my_test = TestSync()
    my_test.test_calculate_sync()

#=============================================================================
if __name__ == '__main__':
    print('running main')
    do_all()