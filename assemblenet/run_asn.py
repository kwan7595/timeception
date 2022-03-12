# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to run AssembleNets without object input."""
import datetime
import json
import logging
import os
from optparse import OptionParser

import tensorflow.compat.v1 as tf  # tf
from absl import app
from absl import flags
from keras.optimizer_experimental import sgd, adam

import keras_utils
from assemblenet import assemblenet,assemblenet_plus,assemblenet_plus_lite,model_structures
from core import utils, config_utils, config, data_utils

logger = logging.getLogger(__name__)

tf.compat.v1.disable_eager_execution()


flags.DEFINE_string('precision', 'float32',
                    'Precision to use; one of: {bfloat16, float32}.')
flags.DEFINE_integer('num_frames', 32, 'Number of frames to use.')

flags.DEFINE_integer('num_classes', 157,
                     'Number of classes. 157 is for Charades')


flags.DEFINE_string('assemblenet_mode', 'assemblenet',
                    '"assemblenet" or "assemblenet_plus" or "assemblenet_plus_lite"')  # pylint: disable=line-too-long

flags.DEFINE_string('model_structure', '[-1,1]',
                    'AssembleNet model structure in the string format.')
flags.DEFINE_string(
    'model_edge_weights', '[]',
    'AssembleNet model structure connection weights in the string format.')

flags.DEFINE_string('attention_mode', None, '"peer" or "self" or None')

flags.DEFINE_float('dropout_keep_prob', None, 'Keep ratio for dropout.')
flags.DEFINE_bool(
    'max_pool_preditions', True,
    'Use max-pooling on predictions instead of mean pooling on features. It helps if you have more than 32 frames.')  # pylint: disable=line-too-long

flags.DEFINE_bool('use_object_input', False,
                  'Whether to use object input for AssembleNet++ or not')  # pylint: disable=line-too-long
flags.DEFINE_integer('num_object_classes', 151,
                     'Number of object classes, when using object inputs. 151 is for ADE-20k')  # pylint: disable=line-too-long


FLAGS = flags.FLAGS
def __define_data_generator(is_training):
    """
    Define data generator.
    """

    # get some configs for the training
    n_classes = config.cfg.MODEL.N_CLASSES
    dataset_name =config.cfg.DATASET_NAME
    backbone_model_name = config.cfg.MODEL.BACKBONE_CNN
    backbone_feature_name = config.cfg.MODEL.BACKBONE_FEATURE
    n_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS

    batch_size_tr = config.cfg.TRAIN.BATCH_SIZE
    batch_size_te = config.cfg.TEST.BATCH_SIZE
    batch_size = batch_size_tr if is_training else batch_size_te

    # size and name of feature
    feature_name = 'features_%s_%s_%sf' % (backbone_model_name, backbone_feature_name, n_timesteps)
    c, h, w = utils.get_model_feat_maps_info(backbone_model_name, backbone_feature_name)
    feature_dim = (n_timesteps, h, w, c)

    # data generators
    params = {'batch_size': batch_size, 'n_classes': n_classes, 'feature_name': feature_name, 'feature_dim': feature_dim, 'is_shuffle': True, 'is_training': is_training}
    data_generator_class = data_utils.KERAS_DATA_GENERATORS_DICT[dataset_name]
    data_generator = data_generator_class(**params)

    return data_generator

def train_tco(network):
    """
    Train Timeception layers based on the given configurations.
    This train scheme is Timeception-only (TCO).
    """

    # get some configs for the training
    n_workers = config.cfg.TRAIN.N_WORKERS
    n_epochs = config.cfg.TRAIN.N_EPOCHS
    dataset_name = config.cfg.DATASET_NAME
    model_name = '%s_%s' % (config.cfg.MODEL.NAME, utils.timestamp())

    # data generators
    data_generator_tr = __define_data_generator(is_training=True)
    data_generator_te = __define_data_generator(is_training=False)

    logger.info('--- start time')
    logger.info(datetime.datetime.now())
    logger.info('... [tr]: n_samples, n_batch, batch_size: %d, %d, %d' % (data_generator_tr.n_samples, data_generator_tr.n_batches, config.cfg.TRAIN.BATCH_SIZE))
    logger.info('... [te]: n_samples, n_batch, batch_size: %d, %d, %d' % (data_generator_te.n_samples, data_generator_te.n_batches, config.cfg.TEST.BATCH_SIZE))

    # callback to save the model
    save_callback = keras_utils.SaveCallback(dataset_name, model_name)

    # load model
    model = network
    logger.info(model.summary())

    # train the model
    model.fit_generator(epochs=n_epochs, generator=data_generator_tr, validation_data=data_generator_te, use_multiprocessing=True, workers=n_workers, callbacks=[save_callback], verbose=2)

    logger.info('--- finish time')
    logger.info(datetime.datetime.now())

def main(_):
  # Create model.

  batch_size = 8
  image_size = 256

  vid_placeholder = tf.placeholder(tf.float32,
                                   (batch_size, FLAGS.num_frames, image_size, image_size, 3))  # pylint: disable=line-too-long

  if FLAGS.assemblenet_mode == 'assemblenet_plus_lite':
    FLAGS.model_structure = json.dumps(model_structures.asnp_lite_structure)
    FLAGS.model_edge_weights = json.dumps(model_structures.asnp_lite_structure_weights)  # pylint: disable=line-too-long

    network = assemblenet_plus_lite.assemblenet_plus_lite(
        num_layers=[3, 5, 11, 7],
        num_classes=FLAGS.num_classes,
        data_format='channels_last')
  else:
    vid_placeholder = tf.reshape(vid_placeholder,
                                 [batch_size*FLAGS.num_frames, image_size, image_size, 3])  # pylint: disable=line-too-long

    if FLAGS.assemblenet_mode == 'assemblenet_plus':
      # Here, we are using model_structures.asn50_structure for AssembleNet++
      # instead of full_asnp50_structure. By using asn50_structure, it
      # essentially becomes AssembleNet++ without objects, only requiring RGB
      # inputs (and optical flow to be computed inside the model).
      FLAGS.model_structure = json.dumps(model_structures.asn50_structure)
      FLAGS.model_edge_weights = json.dumps(model_structures.asn_structure_weights)  # pylint: disable=line-too-long

      network = assemblenet_plus.assemblenet_plus(
          assemblenet_depth=50,
          num_classes=FLAGS.num_classes,
          data_format='channels_last')
    else:
      FLAGS.model_structure = json.dumps(model_structures.asn50_structure)
      FLAGS.model_edge_weights = json.dumps(model_structures.asn_structure_weights)  # pylint: disable=line-too-long

      network = assemblenet.assemblenet_v1(
          assemblenet_depth=50,
          num_classes=FLAGS.num_classes,
          data_format='channels_last')

  # The model function takes the inputs and is_training.
  outputs = network(vid_placeholder, True)
  # some configurations for the model
  classification_type = config.cfg.MODEL.CLASSIFICATION_TYPE
  solver_name = config.cfg.SOLVER.NAME
  solver_lr = config.cfg.SOLVER.LR
  adam_epsilon = config.cfg.SOLVER.ADAM_EPSILON
  n_tc_timesteps = config.cfg.MODEL.N_TC_TIMESTEPS
  backbone_name = config.cfg.MODEL.BACKBONE_CNN
  feature_name = config.cfg.MODEL.BACKBONE_FEATURE
  n_tc_layers = config.cfg.MODEL.N_TC_LAYERS
  n_classes = config.cfg.MODEL.N_CLASSES
  is_dilated = config.cfg.MODEL.MULTISCALE_TYPE
  n_channels_in, channel_h, channel_w = 3,256,256
  n_groups = int(n_channels_in / 128.0)

  # optimizer and loss for either multi-label "ml" or single-label "sl" classification
  if classification_type == 'ml':
      loss = keras_utils.LOSSES[3]
      output_activation = keras_utils.ACTIVATIONS[2]
      metric_function = keras_utils.map_charades
  else:
      loss = keras_utils.LOSSES[0]
      output_activation = keras_utils.ACTIVATIONS[3]
      metric_function = keras_utils.METRICS[0]

  # define the optimizer
  optimizer = sgd(lr=0.01) if solver_name == 'sgd' else adam(lr=solver_lr,epsilon=adam_epsilon)
  network.compile(loss=loss, optimizer=optimizer, metrics=[metric_function])

  default_config_file = 'charades_asn++_32.yaml'
  parser = OptionParser()
  parser.add_option('-c', '--config_file', dest='config_file', default=default_config_file,
                    help='Yaml config file that contains all training details.')
  (options, args) = parser.parse_args()
  config_file = options.config_file

  # check if exist
  if config_file is None or config_file == '':
      msg = 'Config file not passed, default config is used: %s' % (config_file)
      logging.warning(msg)
      config_file = default_config_file

  # path of config file
  config_path = '../configs/%s' % (config_file)

  # check if file exist
  if not os.path.exists(config_path):
      msg = 'Sorry, could not find config file with the following path: %s' % (config_path)
      logging.error(msg)
  else:
      # read the config from file and copy it to the project configuration "cfg"
      config_utils.cfg_from_file(config_path)

      # choose which training scheme, either 'ete' or 'tco'
      training_scheme = config.cfg.TRAIN.SCHEME

      # start training
      if training_scheme == 'tco':
          train_tco()

  # with tf.Session() as sess:
  #   # Generate a random video to run on.
  #   # This should be replaced by a real video.
  #   arr_path = "C:/Users/oem/anaconda3/envs/Nas_env/charades_data_pipeline/sample3.npy"
  #
  #   arr = np.load(arr_path)
  #   vid = arr
  #   # vid = np.random.rand(*vid_placeholder.shape)
  #   # # print(tf.shape(vid))
  #   sess.run(tf.global_variables_initializer())
  #   logits = sess.run(outputs, feed_dict={vid_placeholder: vid})
  #   print(logits)
  #   print(np.argmax(logits, axis=1))
  #   print(logits.shape)
  #   print(np.max(logits,axis=1))



if __name__ == '__main__':
  app.run(main)
