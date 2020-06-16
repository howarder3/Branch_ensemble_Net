import tensorflow as tf
import numpy as np
import time
import datetime
import os

from glob import glob
from six.moves import xrange

# --------- self define function ---------
# # ops: layers structure
# from ops import *
# # vis_util: visualization
# from vis_util import *
# # utils: for loading data, model
# from utils import *

class Branch_ensemble_Net(object):
	def __init__(self, sess,
				start_epoch = 1, end_epoch = 300,
				batch_size = 120, trials = 32, 
				merge_strategy = 2,
				ema_decay_rate = 0.999, 	
				data_dir = 'data_dir', 
				log_dir = 'log_dir'):

		self.sess = sess
		self.start_epoch = start_epoch		
		self.end_epoch = end_epoch
		self.batch_size = batch_size
		self.trials = trials
		self.merge_strategy = merge_strategy
		self.ema_decay_rate = ema_decay_rate
		self.data_dir = data_dir
		self.log_dir = log_dir


	def build_model(self):
		pass

	def train(self,sess):
		pass
