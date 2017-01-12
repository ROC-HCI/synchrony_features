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
        Class for storing raw data and synchrony based results. Contains methods 
        for multiple algorithms.
    
    
    """
    
    #-------------------------------
    def __init__(self, X_, Y_):
        self.SAMPLE_RATE = 15
        self.X = np.array(X_)
        self.Y = np.array(Y_)
        
        assert (len(X_) == len(Y_)), 'X and Y array sizes are different'
        
        self.X_quant = []
        self.Y_quant = []
 
        self.THRESH = 0
        
    #-------------------------------
    def quantize_data(self, THRESH_):
        """ for X and Y, determines whether each value is > THRESH, 
            and puts result into self.X_quant and self.Y_quant
        """
        self.THRESH = THRESH_
        
        self.X_quant = np.zeros_like(self.X, float)
        for t, x in enumerate(self.X):
            if np.isnan(x): 
                self.X_quant[t] = np.nan
            else:
                self.X_quant[t] = x > self.THRESH
                
        
        self.Y_quant = np.zeros_like(self.Y, float)
        for t, y in enumerate(self.Y):
            if np.isnan(y): 
                self.Y_quant[t] = np.nan
            else:
                self.Y_quant[t] = y > self.THRESH
       
    #-------------------------------
    def calculate_corr_sync(self, slot_size_=22, max_time_shift_=37):
        """ Splits X into slots. For each X slot, calculates the correlation
            with every Y slots following the X slot time up to max_time_shift.
            -------------------------------
            return value: 
               x2y_sync, y2x_sync            
        """
        pass
    
    #-------------------------------    
    def calc_edge_trig_sync(self, THRESH_, max_time_shift_=37):
        """ given raw data stored in X and Y, calculates synchrony using
            the "edge triggered algorithm.  In turn treats X as the sender
            and Y as the receiver, then Y as the sender, X as the receiver.
            For each sender rising edge, checks if there is any positive 
            in Y in the max_time_shift following the time at X rising edge.
            
            Maybe it is better to set up an ignore region in the sender
            after each share match (if y_last > t, ignore t)?
            --------------------------
            return value: 
               x2y_sync, y2x_sync
        """
        
        self.quantize_data(THRESH_)
                
        # find rising edges in X
        x_rising_edges = [] # list of times where there is a rising edge
        x_last = 0
        for t, x in enumerate (self.X_quant):
            # is it a rising edge
            if ( (x_last == 0) and (x == 1) ):
                x_rising_edges.append(t)
 
                logging.debug('x rising edge at t :' + str(t))
                logging.debug('x rising edge cnt:' + str(len(x_rising_edges)))
            x_last = x
            
        # find rising edges in Y
        y_rising_edges = []
        y_last = 0
        for t, y in enumerate (self.Y_quant):
            # is it a rising edge
            if ( (y_last == 0) and (y == 1) ):                
                y_rising_edges.append(t)
 
                logging.debug('y rising edge at t :' + str(t))
                logging.debug('y rising edge cnt:' + str(len(y_rising_edges)))
            y_last = y
            
        # count y follows x
        x2y_shared_cnt = 0
        y_last_abs = 0
        for t in x_rising_edges:
            # make sure to ignore previous y==1 in the windows start position
            start_index = np.maximum(t,y_last_abs+1) 
            window = self.Y_quant[start_index:t+max_time_shift_]
            window_is_one = (window == 1) # just to make sure that nan does not give positive
            if np.any(window_is_one):
                x2y_shared_cnt += 1
                y_last_abs = t + np.where(window_is_one == 1)[0][0]

        logging.debug('x2y_shared_cnt :' + str(x2y_shared_cnt))

        x2y_sync = float(x2y_shared_cnt) / len(x_rising_edges)


        # count x follows y
        y2x_shared_cnt = 0
        x_last_abs = 0
        for t in y_rising_edges:
            # make sure to ignore previous x==1 in the windows start position
            start_index = np.maximum(t,x_last_abs+1) 
            window = self.X_quant[start_index:t+max_time_shift_]
            window_is_one = (window == 1) # just to make sure that nan does not give positive
            if np.any(window_is_one):
                y2x_shared_cnt += 1
                x_last_abs = t + np.where(window_is_one == 1)[0][0]

        logging.debug('x2y_shared_cnt :' + str(y2x_shared_cnt))
        y2x_sync = float(y2x_shared_cnt) / len(y_rising_edges)

        sync_data = (x2y_sync, y2x_sync, x_rising_edges, y_rising_edges, 
                     x2y_shared_cnt, y2x_shared_cnt)
        return sync_data

    #-------------------------------
    def plot(self):
        # Sample rate and desired cutoff frequencies (in Hz).
        plt.figure(1)
        plt.clf()

        time = np.array(range(len(self.X)),float) / self.SAMPLE_RATE
        
        plt.subplot(221)
        plt.plot(time, self.X)
        plt.xlabel('Time (sec)')
        plt.title('X')
        plt.axhline(y=self.THRESH, color='r')
        plt.grid()

        plt.subplot(222)
        plt.plot(time, self.X_quant)
        plt.fill_between(time, 0, self.X_quant, where=self.X_quant >= 0, 
                         facecolor='green', interpolate=True)
        plt.title('X_quant')
        plt.grid()

        plt.subplot(223)
        plt.plot(time, self.Y)
        plt.xlabel('Time (sec)')
        plt.title('Y')
        plt.axhline(y=self.THRESH, color='r')
        plt.grid()

        plt.subplot(224)
        plt.plot(time, self.Y_quant)
        plt.fill_between(time, 0, self.Y_quant, where=self.Y_quant >= 0, 
                         facecolor='green', interpolate=True)
        plt.title('Y_quant')
        plt.grid()
        
        plt.tight_layout()
        plt.show()      
        
    def print_sync_data(self, sync_data):

        (x2y_sync, y2x_sync, x_rising_edges, y_rising_edges, 
                             x2y_shared_cnt, y2x_shared_cnt) = sync_data       
        s = ''
        s += 'synchrony data:'
        s += '\n\tx2y_sync: ' + str(x2y_sync)
        s += '\n\ty2x_sync: ' + str(y2x_sync)
        s += '\n\tx_rising_edges: ' + str(len(x_rising_edges))
        s += '\n\ty_rising_edges: ' + str(len(y_rising_edges)) 
        s += '\n\tx2y_shared_cnt: ' + str(x2y_shared_cnt)
        s += '\n\ty2x_shared_cnt: ' + str(y2x_shared_cnt)
        return s

    #-------------------------------
    def __str__(self):
        # TODO
        s = "Syncrony: "
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

        sync_data = my_sync.calc_edge_trig_sync(THRESH_=50,  
                                                max_time_shift_=37)
        #my_sync.plot()
        print(my_sync.print_sync_data(sync_data)) 
        (x2y_sync, y2x_sync, x_rising_edges, y_rising_edges, 
                             x2y_shared_cnt, y2x_shared_cnt) = sync_data
        self.assertAlmostEqual(x2y_sync,0.5, places = 2)
        self.assertAlmostEqual(y2x_sync,0.0, places = 2)
        self.assertEqual(len(x_rising_edges),2)
        self.assertEqual(len(y_rising_edges),1)
        self.assertEqual(x2y_shared_cnt,1)
        self.assertEqual(y2x_shared_cnt,0)
        
        
        print(my_sync)
        

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
        
        sync_data = my_sync.calc_edge_trig_sync(THRESH_=10,  
                                                max_time_shift_=37)
        print(my_sync.print_sync_data(sync_data)) 
        #my_sync.plot()
        

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
        
        sync_data = my_sync.calc_edge_trig_sync(THRESH_=10,  
                                                max_time_shift_=37)
        print(my_sync.print_sync_data(sync_data)) 
        my_sync.plot()
        


#=============================================================================
if __name__ == '__main__':
    print('running main')
    unittest.main()
    do_all()