from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym.spaces
import tensorflow as tf
from reinforce.models.visionnet import vision_net
from reinforce.models.fcnet import fc_net


class ProximalPolicyLoss(object):

  def __init__(
      self, observation_space, action_space, preprocessor,
      observations, advantages, actions, prev_logits, logit_dim,
      kl_coeff, distribution_class, config, sess, report_metrics):
    assert (isinstance(action_space, gym.spaces.Discrete) or
            isinstance(action_space, gym.spaces.Box))
    self.prev_dist = distribution_class(prev_logits)

    # TODO(ekl) shouldn't have to save this
    self.observations = observations

    if len(observation_space.shape) > 1:
      self.curr_logits = vision_net(observations, num_classes=logit_dim)
    else:
      assert len(observation_space.shape) == 1
      self.curr_logits = fc_net(observations, num_classes=logit_dim)
    self.curr_dist = distribution_class(self.curr_logits)
    self.sampler = self.curr_dist.sample()
    self.entropy = self.curr_dist.entropy()
    # Make loss functions.
    self.ratio = tf.exp(self.curr_dist.logp(actions) -
                        self.prev_dist.logp(actions))
    self.kl = self.prev_dist.kl(self.curr_dist)
    self.mean_kl = tf.reduce_mean(self.kl)
    self.mean_entropy = tf.reduce_mean(self.entropy)
    self.surr1 = self.ratio * advantages
    self.surr2 = tf.clip_by_value(self.ratio, 1 - config["clip_param"],
                                  1 + config["clip_param"]) * advantages
    self.surr = tf.minimum(self.surr1, self.surr2)
    self.loss = tf.reduce_mean(-self.surr + kl_coeff * self.kl -
                               config["entropy_coeff"] * self.entropy)
    self.sess = sess

    with tf.device("/cpu:0"):
      if report_metrics:
        with tf.name_scope('kl_coeff'):
          tf.summary.scalar('cur_value', kl_coeff)
        with tf.name_scope('kl'):
          tf.summary.scalar('mean', self.mean_kl)
        with tf.name_scope('entropy'):
          tf.summary.scalar('mean', self.mean_entropy)
        with tf.name_scope('surrogate_loss'):
          tf.summary.scalar('mean', tf.reduce_mean(self.surr))

  def compute_actions(self, observations):
    return self.sess.run([self.sampler, self.curr_logits],
                         feed_dict={self.observations: observations})

  def loss(self):
    return self.loss
