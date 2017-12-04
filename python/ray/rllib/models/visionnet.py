from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer


class VisionNetwork(Model):
    """Generic vision network."""

    def _init(self, inputs, num_outputs, options):
        extra = options.get("extra_inputs")
        filters = options.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
            [512, [10, 10], 1],
        ])
        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs, out_size, kernel, stride,
                    scope="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs, out_size, kernel, stride, padding="VALID", scope="fc1")
            if extra is None:
                fc2 = slim.conv2d(fc1, num_outputs, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope="fc2")
                out = tf.squeeze(fc2, [1, 2])
            else:
                fc1_aug = tf.concat([tf.squeeze(fc1, [1, 2]), extra], axis=1)
                fc2 = slim.fully_connected(
                    fc1_aug, num_outputs,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=tf.nn.tanh, scope="fc2_aug")
                fc2_aug = tf.concat([fc2, extra], axis=1)
                out = slim.fully_connected(
                    fc2_aug, num_outputs,
                    weights_initializer=normc_initializer(0.01),
                    activation_fn=None, scope="fc_out_aug")
            return out, tf.squeeze(fc1, [1, 2])
