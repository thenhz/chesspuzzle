#from src.LMST_Network import LMST_Network
from src.NNHz1.NNHz1 import NNHz1
from src.parameters import *
from src.Utils import *


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes, sess):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_queens_num = []
        self.nQueensPrint = s_size -1
        self.nQueensConsole = s_size -2
        self.summary_writer = tf.summary.FileWriter(tensorboard_path + "/train_" + str(self.number), sess.graph)

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = NNHz1(s_size, a_size, self.name, trainer,self.summary_writer)
        self.update_local_ops = update_target_graph('global', self.name)

        # !!
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()

        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save

        feed_dict={
            self.local_AC.reward_holder:discounted_rewards,
            self.local_AC.action_holder:actions,
            self.local_AC.state_in:np.vstack(np.stack(observations))
        }

        g_n, v_n, _ = sess.run([self.local_AC.grad_norms,
                                self.local_AC.var_norms,
                                self.local_AC.apply_grads],
                                feed_dict=feed_dict)
        return g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.reset()
                s = self.env.state.flatten()
                episode_frames.append(s)
                #s = process_frame(s)

                while not self.env.done:
                    # Take an action using probabilities from policy network output.
                    a_dist = sess.run(self.local_AC.output,feed_dict={self.local_AC.state_in:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d, _, placedQueens = self.env.step(a)

                    if d == False:
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d])

                    episode_reward += r
                    s = s1
                    episode_step_count += 1

                    if d == True:
                        break

                if placedQueens>=self.nQueensConsole:
                    print('Game Over!!! Placed Queens:%s Reward:%s'%(placedQueens,episode_reward))
                if placedQueens>=self.nQueensPrint:
                    # Convert PNG buffer to TF image
                    image = tf.image.decode_png(self.env.render().getvalue(), channels=4)
                    # Add the batch dimension
                    self.summary_writer.flush()

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_queens_num.append(placedQueens)

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_queens = np.mean(self.episode_queens_num[-5:])

                    summary = tf.Summary()

                    weights = sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)])
                    for w in weights[0]:
                        variable_summaries2(summary,w)


                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/NumQueens', simple_value=float(mean_queens))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))


                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
