
import tensorflow as tf
import os
from datetime import datetime

from model import Branch_ensemble_Net

 

# parameters setting - integer
tf.app.flags.DEFINE_integer('start_epoch',1,'start epoch')
tf.app.flags.DEFINE_integer('end_epoch',300,'end epoch')
tf.app.flags.DEFINE_integer('batch_size',120,'batch size')
tf.app.flags.DEFINE_integer('trials',32,'trials')


# parameters setting - float
tf.app.flags.DEFINE_float('merge_strategy',2,'merge_strategy')
tf.app.flags.DEFINE_float('ema_decay_rate',0.999,'ema_decay_rate') 


# folder position
tf.app.flags.DEFINE_string('data_dir', 'data_dir', 'data directory') 
tf.app.flags.DEFINE_string('log', 'log', 'log directory')


FLAGS = tf.app.flags.FLAGS

def main(_):
	data_dir = FLAGS.data_dir
	log_dir = FLAGS.log


	with tf.compat.v1.Session() as sess:
		if tf.io.gfile.exists(data_dir):
			tf.io.gfile.rmtree(data_dir)
		tf.io.gfile.makedirs(data_dir)


		if tf.io.gfile.exists(log_dir):
			tf.io.gfile.rmtree(log_dir)
		tf.io.gfile.makedirs(log_dir)

		for i in range(FLAGS.trials):
			rn = datetime.now().strftime("%Y%m%d%H%M%S")
				
			model = Branch_ensemble_Net(sess, 
					start_epoch = FLAGS.start_epoch, end_epoch = FLAGS.end_epoch,
					batch_size = FLAGS.batch_size, trials = FLAGS.trials, 
					merge_strategy = FLAGS.merge_strategy,
					ema_decay_rate = FLAGS.ema_decay_rate,	
					data_dir=FLAGS.data_dir, log_dir =FLAGS.log_dir)

			model.train(sess)



# program start here
if __name__ == '__main__':
	 tf.compat.v1.app.run()
