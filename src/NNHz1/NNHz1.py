
import tensorflow.contrib.slim as slim
from src.helper import *
from src.Utils import *


class NNHz1:

    lr = 1e-2

    def __init__(self,s_size,a_size,scope,trainer,writer):
        with tf.variable_scope(scope):


            #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
            self.state_in= tf.placeholder(shape=[None,s_size*s_size],dtype=tf.float32)
            hidden = slim.fully_connected(self.state_in,8,biases_initializer=None,activation_fn=tf.nn.relu)
            self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
            self.chosen_action = tf.argmax(self.output,1)


            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':

                #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
                #to compute the loss, and use it to update the network.
                self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
                self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

                self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
                self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

                self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                #for w in local_vars:
                #    variable_summaries(writer, w)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)


                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

