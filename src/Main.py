import os, shutil
import threading
from time import sleep

import tensorflow as tf
import multiprocessing
from src.NNHz1.ChessPuzzle import ChessPuzzle
from src.NNHz1.NNHz1 import NNHz1
from src.NNHz1.Worker import Worker
from src.parameters import *

tf.reset_default_graph()


if os.path.exists(app_path):
    shutil.rmtree(app_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(img_path):
    os.makedirs(img_path)

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = NNHz1(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(ChessPuzzle(chess_size),i,s_size,a_size,trainer,model_path,global_episodes,sess))
        
    coord = tf.train.Coordinator()
    if load_model:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=worker_work)
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
