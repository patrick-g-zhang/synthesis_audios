import sys
sys.path.insert(1, '/home/gyzhang/merlin/src')
# import matplotlib.pyplot as plt
# %matplotlib inline
from frontend.mlpg import MLParameterGenerationFast as MLParameterGeneration

import numpy as np
import configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from io_funcs.binary_io import BinaryIOCollection
from run_tensorflow_with_merlin_io import TensorflowClass
from tensorflow_lib.train import TrainTensorflowModels

from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii import paramgen
from nnmnkwii.io import hts
from nnmnkwii.postfilters import merlin_post_filter

import threading
import pysptk
import pyworld
from scipy.io import wavfile

import re
import tensorflow as tf

cmu_arctic = tf.contrib.training.HParams(
    model_dir='/home/gyzhang/merlin/egs/cmu_arctic/s1/experiments/cmu_arctic_2/acoustic_model/nnets_model/tensorflow/feed_forward_6_tanh',


)
casia = tf.contrib.training.HParams(
    model_dir='/home/gyzhang/merlin/egs/casia/s1/experiments/liuchang/acoustic_model/nnets_model/tensorflow/feed_forward_6_tanh',
    norm_info_file="/home/gyzhang/merlin/egs/casia/s1/experiments/liuchang/acoustic_model/inter_module/norm_info__mgc_lf0_vuv_bap_187_MVN.dat",
    test_norm_path="/home/gyzhang/merlin/egs/casia/s1/experiments/liuchang/acoustic_model/inter_module/nn_no_silence_lab_norm_413/liuchanhg-happy-319.lab",
    hidden_layer_size=[512, 512, 512, 512, 512, 512],
    n_in=413,
    training_num=1614)


class Predict_Syn(object):
    """docstring for Predict_syn"""

    def __init__(self, ):
        # basic parameters of acoustic feature
        self.windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]

        self.sr = 16000
        self.alpha = pysptk.util.mcepalpha(self.sr)
        self.fftlen = 1024
        self.frame_period = 5

        self.mgc_start_idx = 0
        self.lf0_start_idx = 180
        self.vuv_start_idx = 183
        self.bap_start_idx = 184

        # configuration of neural network
        # self.n_in = 421
        self.n_in = casia.n_in

        # self.hidden_layer_size = [1024, 1024, 1024, 1024, 1024, 1024]
        self.hidden_layer_size = casia.hidden_layer_size

        self.n_out = 187
        self.hidden_layer_type = [
            'tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'tanh']
        self.norm_info_file = casia.norm_info_file
        self.test_norm_path = casia.test_norm_path
        self.model_dir = casia.model_dir
        self.training_num = casia.training_num
        # path
        # acoustic mean and var
        # self.norm_info_file = "/home/gyzhang/merlin/egs/cmu_arctic/s1/experiments/cmu_arctic_2/acoustic_model/inter_module/norm_info__mgc_lf0_vuv_bap_187_MVN.dat"
        # linguistic norm features
        # self.test_norm_path = "/home/gyzhang/merlin/egs/cmu_arctic/s1/experiments/cmu_arctic_2/acoustic_model/inter_module/nn_no_silence_lab_norm_421/arctic_b0452.lab"

        self.tensorflow_models = self.load_tensorflow_model()

        self.mlpg_algo = MLParameterGeneration()

    def load_tensorflow_model(self,):
        tensorflow_models = TrainTensorflowModels(
            self.n_in, self.hidden_layer_size, self.n_out, self.hidden_layer_type, self.model_dir)
        tensorflow_models.define_feedforward_model_utt()
        return tensorflow_models

    def load_prev_fea(self,):
        # load acoustic var and mean and linguistic feature
        fid = open(self.norm_info_file, 'rb')
        cmp_min_max = np.fromfile(fid, dtype=np.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_mean_vector = cmp_min_max[0, ]
        cmp_std_vector = cmp_min_max[1, ]
        io_funcs = BinaryIOCollection()
        inp_features, frame_number = io_funcs.load_binary_file_frame(
            self.test_norm_path, self.n_in)
        test_lin_x, test_lab_x = np.hsplit(inp_features, np.array([-1]))
        # set 100 as vary utterance embedding
        test_lab_x = np.tile(np.array(100), (test_lab_x.shape[0], 1))
        return cmp_mean_vector, cmp_std_vector, test_lin_x, test_lab_x

    def inference(self, z1, z2, test_lin_x, test_lab_x, embed_index):
        with self.tensorflow_models.graph.as_default():
            new_saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                latest_ckpt = tf.train.latest_checkpoint(
                    self.tensorflow_models.ckpt_dir, latest_filename=None)
                new_saver.restore(sess, latest_ckpt)
                v1 = sess.graph.get_tensor_by_name('utt-embeddings:0')
                v1_array = sess.run(v1)
                v1_array[embed_index] = [z1, z2]
                sess.run(tf.assign(v1, v1_array))
                y_predict = sess.run(self.tensorflow_models.output_layer, feed_dict={
                                     self.tensorflow_models.input_lin_layer: test_lin_x, self.tensorflow_models.utt_index_t: test_lab_x, self.tensorflow_models.is_training_batch: False})
            return v1_array, y_predict

    def parms_gen(self, z1, z2, embed_index, test_lin_x, test_lab_x, cmp_mean_vector, cmp_std_vector):
        v1_array, y_predict = self.inference(
            z1, z2, test_lin_x, test_lab_x, embed_index)
        norm_features = y_predict * cmp_std_vector + cmp_mean_vector
        T = norm_features.shape[0]
        # Split acoustic features
        mgc = norm_features[:, :self.lf0_start_idx]
        lf0 = norm_features[:, self.lf0_start_idx: self.vuv_start_idx]
        vuv = norm_features[:, self.vuv_start_idx]
        bap = norm_features[:, self.bap_start_idx:]
        cmp_var_vector = cmp_std_vector**2
        mgc_variances = np.tile(cmp_var_vector[:self.lf0_start_idx], (T, 1))
        mgc = self.mlpg_algo.generation(mgc, mgc_variances, 60)
        lf0_variances = np.tile(
            cmp_var_vector[self.lf0_start_idx:self.vuv_start_idx], (T, 1))
        lf0 = self.mlpg_algo.generation(lf0, lf0_variances, 1)
        bap_variances = np.tile(cmp_var_vector[self.bap_start_idx:], (T, 1))
        bap = self.mlpg_algo.generation(bap, bap_variances, 1)
        f0 = lf0.copy()
        f0[vuv < 0.5] = 0
        f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
        return v1_array, y_predict, f0, mgc, bap

    def gen_wav(self, f0, mgc, bap):
        spectrogram = pysptk.mc2sp(mgc, fftlen=self.fftlen, alpha=self.alpha)
        aperiodicity = pyworld.decode_aperiodicity(
            bap.astype(np.float64), self.sr, self.fftlen)
        generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64), spectrogram.astype(
            np.float64), aperiodicity.astype(np.float64), self.sr, self.frame_period)
        x2 = generated_waveform / np.max(generated_waveform) * 32768
        x2 = x2.astype(np.int16)
        wavfile.write("gen.wav", self.sr, x2)
        with open("gen.wav", 'rb') as fd:
            contents = fd.read()
        intensity = 10 * np.log10(np.sum(spectrogram**2, axis=1))
        return contents, intensity

    def load_casia_color(self, path="/home/gyzhang/merlin/egs/casia/s1/experiments/liuchang/acoustic_model/data/metadata.tsv"):
        emotion_dict = {'happy': 0, "sad": 1, "angry": 2,
                        "neutral": 3, "fear": 4, "surprise": 5}
        ['green', 'white', 'black', 'blue','magenta','yellow', 'red']
        colors = np.zeros(casia.training_num)
        with open(path, 'r') as fid:
            file_lines = fid.readlines()
        for num, each_line in enumerate(file_lines[1:]):
            _, _, emotion = re.split("\t", each_line.strip())
            emo_id = emotion_dict[emotion]
            colors[num] = emo_id
        return colors
