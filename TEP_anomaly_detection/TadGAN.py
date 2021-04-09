import logging
import math
import tempfile
from functools import partial

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import TimeDistributed
# from tensorflow.keras.layers import Dense

from scipy import stats

from timeseries_errors import reconstruction_errors

LOGGER = logging.getLogger(__name__)


class TadGAN(tf.keras.Model):
    """
    TadGAN model for time series reconstruction.

    Args:
        encoder_input_shape (tuple):
            Shape of encoder input - (slide window size, number of features)
        generator_input_shape (tuple):
            Shape of generator input - (latent space for slide window, latent space for features)
        critic_x_input_shape (tuple):
            Shape of critic_x input.
        critic_z_input_shape (tuple):
            Shape of critic_z input.
        optimizer (keras.optimizers):
            Keras optimizer, learning rate of the optimizer is recommended set to 0.005
        batch_size (int):
            Integer denoting the batch size. Default 20.
        iterations_critic (int):
            Optional. Integer denoting the number of critic training steps per one Generator/Encoder training step. Default 5.
        gradient_penalty_weight (int):
        reconstruction_penalty_weight (int):
    """

    def __init__(self, encoder_input_shape=(100, 52), generator_input_shape=(20, 36), iterations_critic=5,
                 batch_size=20, gradient_penalty_weight=10, reconstruction_penalty_weight=10, log_all_losses=False):
        super(TadGAN, self).__init__()

        self.iterations_critic = iterations_critic
        self.encoder_input_shape = encoder_input_shape
        self.generator_input_shape = generator_input_shape
        self.critic_x_input_shape = encoder_input_shape
        self.critic_z_input_shape = generator_input_shape

        self.encoder = self._build_encoder()
        self.generator = self._build_generator()
        self.critic_x = self._build_critic_x()
        self.critic_z = self._build_critic_z()
        self.log_all_losses = log_all_losses
        self.batch_size = batch_size
        self.gradient_penalty_weight = gradient_penalty_weight
        self.reconstruction_penalty_weight = reconstruction_penalty_weight

    def compile(self, critic_x_optimizer, critic_z_optimizer, encoder_generator_optimizer):
        super(TadGAN, self).compile()
        self.critic_x_optimizer = critic_x_optimizer
        self.critic_z_optimizer = critic_z_optimizer
        self.encoder_generator_optimizer = encoder_generator_optimizer

        self._critic_x_loss_fn = self._wasserstein_loss
        self._critic_z_loss_fn = self._wasserstein_loss
        self._encoder_generator_loss_fn = self._wasserstein_loss

    @tf.function # speed up training (https://keras.io/guides/writing_a_training_loop_from_scratch/)
    def _wasserstein_loss(self, y_true, y_pred):
        """
        for real data, label = -1
        for fake data, label = 1
        encourage small scores for fake data and large scores for real data
        Reference: (https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/)
        :param y_true: label for data
        :param y_pred: critic score for data
        """
        return K.mean(y_true * y_pred)

    @tf.function
    def _critic_x_gradient_penalty(self, batch_size, x_true, x_pred):
        """
        Calculate gradient penalty of critic_x
        Reference: Improved Training of Wasserstein GANs (https://arxiv.org/pdf/1704.00028.pdf)
        Reference: WGAN-GP overriding Model.train_step (https://keras.io/examples/generative/wgan_gp/)
        Reference: tf.GradientTape (https://www.tensorflow.org/api_docs/python/tf/GradientTape)

        :param batch_size:
        :param x_true: sampled from real dataset
        :param x_pred: generated by generator with input from random latent space
        :return:
        """

        # Get the interpolated value
        alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
        interpolated = (alpha * x_true) + ((1 - alpha) * x_pred)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)

            # 1. Get the critic_x output for this interpolated value.
            pred = self.critic_x(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 2. Calculate the gradients w.r.t to this interpolated value.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def _critic_z_gradient_penalty(self, batch_size, z_true, z_pred):
        """
        Calculate gradient penalty of critic_z
        check comment of function: _critic_x_gradient_penalty(self, batch_size, x_true, x_pred)

        :param batch_size:
        :param z_true:
        :param z_pred:
        :return:
        """
        alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
        interpolated = (alpha * z_true) + ((1 - alpha) * z_pred)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic_z(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((1.0 - norm) ** 2)
        return gp

    @tf.function
    def _critic_x_loss(self, x, z, valid, fake, batch_size, training=True):
        """

        :param x: sampled from real dataset (real x)
        :param z: sampled from random latent space
        :param valid: label for real x, default to -1
        :param fake: label for fake x, default to 1
        """

        # fake x generated with input from random latent space
        x_ = self.generator(z, training=training)

        # label for fake x predicted by critic_x
        fake_x = self.critic_x(x_, training=training)

        # label for real x predicted by critic_x
        valid_x = self.critic_x(x, training=training)

        critic_x_valid_cost = self._critic_x_loss_fn(y_true=valid, y_pred=valid_x)
        critic_x_fake_cost = self._critic_x_loss_fn(y_true=fake, y_pred=fake_x)
        critic_x_gradient_penalty = self._critic_x_gradient_penalty(batch_size, x, x_)
        critic_x_total_loss = critic_x_valid_cost + critic_x_fake_cost + (
                self.gradient_penalty_weight * critic_x_gradient_penalty)

        return critic_x_total_loss, critic_x_valid_cost, critic_x_fake_cost, critic_x_gradient_penalty

    @tf.function
    def _critic_z_loss(self, x, z, valid, fake, batch_size, training=True):
        """

        :param x: sampled from real dataset
        :param z: sampled from random latent space (real z)
        :param valid: label for real z, default to -1
        :param fake: label for fake z, default to 1
        """

        # fake z encoded with real x
        z_ = self.encoder(x, training=training)

        # label for fake z predicted by critic_z
        fake_z = self.critic_z(z_, training=training)

        # label for real z predicted by critic_z
        valid_z = self.critic_z(z, training=training)

        critic_z_valid_cost = self._critic_z_loss_fn(y_true=valid, y_pred=valid_z)
        critic_z_fake_cost = self._critic_z_loss_fn(y_true=fake, y_pred=fake_z)
        critic_z_gradient_penalty = self._critic_z_gradient_penalty(batch_size, z, z_)
        critic_z_total_loss = critic_z_valid_cost + critic_z_fake_cost + (self.gradient_penalty_weight *
                                                                          critic_z_gradient_penalty)

        return critic_z_total_loss, critic_z_valid_cost, critic_z_fake_cost, critic_z_gradient_penalty

    @tf.function
    def _encoder_generator_loss(self, x, z, valid, training=True):
        """

        :param x: sampled from real dataset
        :param z: sampled from random latent space (real z)
        :param valid:
        :param training:
        :return:
        """
        x_gen_ = self.generator(z, training=training)
        fake_gen_x = self.critic_x(x_gen_, training=training)

        z_gen_ = self.encoder(x, training=training)
        x_gen_rec = self.generator(z_gen_, training=training)

        fake_gen_z = self.critic_z(z_gen_, training=training)

        encoder_generator_fake_gen_x_cost = self._encoder_generator_loss_fn(y_true=valid, y_pred=fake_gen_x)
        encoder_generator_fake_gen_z_cost = self._encoder_generator_loss_fn(y_true=valid, y_pred=fake_gen_z)

        # two terms of loss function not implemented as in reference
        # Reference: TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks (https://arxiv.org/pdf/2009.07769.pdf)
        general_reconstruction_cost = tf.reduce_mean(tf.square((x - x_gen_rec)))
        encoder_generator_total_loss = encoder_generator_fake_gen_x_cost + encoder_generator_fake_gen_z_cost + (
                self.reconstruction_penalty_weight * general_reconstruction_cost)

        return encoder_generator_total_loss, encoder_generator_fake_gen_x_cost, encoder_generator_fake_gen_z_cost, general_reconstruction_cost

    def _build_encoder(self, lstm_units: int = 100, latent_dim_1=40):
        """
        Build the Encoder subnetwork for the GAN. This model learns the compressed representation of the input time series.
        The encoder uses a single layer BI-LSTM network to learn the compressed representation.
        The number of LSTM units can be adjusted.

        :param lstm_units: Number of LSTM units that could be used for the time series encoding
        :return: Encoder model
        """
        x = keras.layers.Input(shape=self.encoder_input_shape, name="encoder_input")

        # Encode the sequence and extend its dimensions
        encoded = keras.layers.Bidirectional(keras.layers.LSTM(units=lstm_units, return_sequences=True))(x)
        encoded = keras.layers.Flatten()(encoded)

        # adding one dense layer to reduce the number of parameters
        # encoded = keras.layers.Dense(units=latent_dim_1, name="latent_encoding_1")(encoded)

        latent_dim_2 = self.generator_input_shape[0] * self.generator_input_shape[1]
        encoded = keras.layers.Dense(units=latent_dim_2, name="latent_encoding_2")(encoded)

        encoded = keras.layers.Reshape(target_shape=self.generator_input_shape, name='output_encoder')(encoded)

        model = keras.Model(inputs=x, outputs=encoded, name="encoder_model")

        return model

    def _build_generator(self, generator_lstm_units: int = 100, output_activation: str = "tanh") -> keras.Model:
        '''
        Build the Generator model for the GAN. This model uses the compressed representation of the encoder and
        tries to reconstruct the original time series from it.
        At the moment a two-layer Bi-LSTM network is used for the reconstruction.

        :param generator_lstm_units:
        :param output_activation:
        :return: Generator model
        '''
        x = keras.layers.Input(shape=self.generator_input_shape, name="generator_input")

        # Remove additional dimensions from the latent embedding
        decoded = keras.layers.Flatten()(x)

        # Check if the sequence length is a even number (this is required for this model architecture)
        if self.encoder_input_shape[0] % 2 == 1:
            raise ValueError(f"The encoder_input_shape[0] needs to be even.")

        # Build the first layer of the generator that should be half the size of the sequence length
        half_seq_length = self.encoder_input_shape[0] // 2
        decoded = keras.layers.Dense(units=half_seq_length)(decoded)
        decoded = keras.layers.Reshape(target_shape=(half_seq_length, 1))(decoded)

        # Generation of a new time series using LSTM in combination with up sampling
        decoded = keras.layers.Bidirectional(
            keras.layers.LSTM(units=generator_lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat')(decoded)

        decoded = keras.layers.UpSampling1D(2)(decoded)

        decoded = keras.layers.Bidirectional(
            keras.layers.LSTM(units=generator_lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            merge_mode='concat')(decoded)

        # Rebuild the original time series signal for all channels
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(self.encoder_input_shape[1]))(decoded)
        decoded = keras.layers.Activation(activation=output_activation)(decoded)

        model = keras.Model(inputs=x, outputs=decoded, name="generator_model")

        return model

    def _build_critic_x(self, n_cnn_filters: int = 64, n_cnn_blocks: int = 4) -> keras.Model:
        """
        Build the critic x network that learns to differ between a fake and real input sequence.
        The classifier uses a stack of 1D CNN block (CNN + leaky relu + dropout) and a final fully connected classifier network.
        :param n_cnn_filters:
        :param n_cnn_blocks:
        :return: Critic x model
        """
        x = keras.layers.Input(shape=self.encoder_input_shape, name="critic_x_input")

        if n_cnn_blocks < 1:
            raise ValueError(f"The number of CNN blocks needs to be greater than 1 (current value: {n_cnn_blocks})")

        y = keras.layers.Conv1D(filters=n_cnn_filters, kernel_size=5)(x)
        y = keras.layers.LeakyReLU(alpha=0.2)(y)
        y = keras.layers.Dropout(rate=0.25)(y)

        if n_cnn_blocks > 1:
            for i in range(n_cnn_blocks - 1):
                y = keras.layers.Conv1D(filters=n_cnn_filters, kernel_size=5)(y)
                y = keras.layers.LeakyReLU(alpha=0.2)(y)
                y = keras.layers.Dropout(rate=0.25)(y)

        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(1)(y)

        model = keras.Model(inputs=x, outputs=y, name="critic_x_model")

        return model

    def _build_critic_z(self, critic_z_dense_units: int = 100) -> keras.Model:
        """
        Build the critic z model that learns to differ between a real and a fake encoding coming from the encoder.
        The network works with a two-layer fully connected network in combination with a leaky RELU activation and dropout regularization.

        :param critic_z_dense_units:
        :return: Critic z model
        """
        x = keras.layers.Input(shape=self.generator_input_shape, name="critic_z_input")
        y = keras.layers.Flatten()(x)

        y = keras.layers.Dense(units=critic_z_dense_units)(y)
        y = keras.layers.LeakyReLU(alpha=0.2)(y)
        y = keras.layers.Dropout(rate=0.2)(y)

        y = keras.layers.Dense(units=critic_z_dense_units)(y)
        y = keras.layers.LeakyReLU(alpha=0.2)(y)
        y = keras.layers.Dropout(rate=0.2)(y)

        y = keras.layers.Dense(1)(y)

        model = keras.Model(inputs=x, outputs=y, name="critic_z_model")

        return model

    @tf.function  # speed up training (https://keras.io/guides/writing_a_training_loop_from_scratch/)
    def train_step(self, X_mini_batch):
        """
        Custom training step for this Subclassing API Keras model.

        mini_batch = batch_size * iterations_critic

        :param X_mini_batch:
        :return:
        """

        print("X_mini_batch: ", X_mini_batch)
        # print("self.train_step_counter: ", self.train_step_counter)
        # self.train_step_counter = self.train_step_counter + 1

        # print(X_mini_batch[0])

        mini_batch_size_shape = tf.shape(X_mini_batch)
        # print("mini_batch_size_shape", mini_batch_size_shape)
        # mini_batch_size_shape = X_mini_batch.shape
        # print("mini_batch_size_shape", mini_batch_size_shape)
        # mini_batch_size_shape = X_mini_batch.get_shape()
        # print("mini_batch_size_shape", mini_batch_size_shape)

        mini_batch_size = mini_batch_size_shape[0]

        # print("mini_batch_size", mini_batch_size)

        # batch_size = self.batch_size
        batch_size = self.batch_size

        # print("batch_size", batch_size)

        # Prepare the ground truth data
        fake = tf.ones((batch_size, 1))
        valid = -tf.ones((batch_size, 1))

        critic_x_loss_steps = []
        critic_z_loss_steps = []
        encoder_generator_loss_steps = []

        # train critic_x and critic_z for iterations_critic steps
        for j in range(self.iterations_critic):
            x = X_mini_batch[j * batch_size: (j + 1) * batch_size]

            print("j * batch_size: ", j * batch_size)

            # cast x to tf.float32
            x = tf.cast(x, tf.float32)

            z = tf.random.normal(shape=(batch_size, self.generator_input_shape[0], self.generator_input_shape[1]))

            # Optimize step on critic x
            with tf.GradientTape() as tape:
                critic_x_losses = self._critic_x_loss(x, z, valid, fake, batch_size, training=True)

            critic_x_gradient = tape.gradient(critic_x_losses[0], self.critic_x.trainable_variables)

            # ??? this code makes same i run two times when @tf.function
            self.critic_x_optimizer.apply_gradients(zip(critic_x_gradient, self.critic_x.trainable_variables))

            critic_x_loss_steps.append(np.array(critic_x_losses))

            # Optimize step on critic z
            with tf.GradientTape() as tape:
                critic_z_losses = self._critic_z_loss(x, z, valid, fake, batch_size, training=True)
            critic_z_gradient = tape.gradient(critic_z_losses[0], self.critic_z.trainable_variables)

            self.critic_z_optimizer.apply_gradients(zip(critic_z_gradient, self.critic_z.trainable_variables))

            critic_z_loss_steps.append(np.array(critic_z_losses))

        print("begin encoder generator gradient cal")

        # Optimize step on encoder & generator and collect gradients
        with tf.GradientTape() as tape:
            # Do a step forward on the encoder generator model
            encoder_generator_losses = self._encoder_generator_loss(x, z, valid)

        # apply gradient option 1:
        encoder_generator_gradient = tape.gradient(encoder_generator_losses[0],
                                                   self.encoder.trainable_variables + self.generator.trainable_variables)
        self.encoder_generator_optimizer.apply_gradients(
            zip(encoder_generator_gradient, self.encoder.trainable_variables + self.generator.trainable_variables))

        print("finish encoder generator apply gradient")

        # apply gradient option 2:
        # encoder_generator_gradient = tape.gradient(encoder_generator_losses[0], [self.encoder.trainable_variables,
        #                                                                          self.generator.trainable_variables])
        # self.encoder_generator_optimizer.apply_gradients(
        #     zip(encoder_generator_gradient[0], self.encoder.trainable_variables))
        # self.encoder_generator_optimizer.apply_gradients(
        #     zip(encoder_generator_gradient[1], self.generator.trainable_variables))

        # test for gradient shape
        # print("encoder_generator_gradient: ", encoder_generator_gradient)
        # print("self.encoder.trainable_variables: ", self.encoder.trainable_variables)
        # print("self.generator.trainable_variables: ", self.generator.trainable_variables)
        # print("self.encoder.trainable_variables + self.generator.trainable_variables: ",
        #       self.encoder.trainable_variables + self.generator.trainable_variables)
        # print("[self.encoder.trainable_variables, self.generator.trainable_variables]: ",
        #       [self.encoder.trainable_variables, self.generator.trainable_variables])

        print("generate dict")

        encoder_generator_loss_steps.append(np.array(encoder_generator_losses))

        critic_x_loss = np.mean(np.array(critic_x_loss_steps), axis=0)
        critic_z_loss = np.mean(np.array(critic_z_loss_steps), axis=0)
        encoder_generator_loss = np.mean(np.array(encoder_generator_loss_steps), axis=0)

        print("finish dict")

        if self.log_all_losses:
            loss_dict = {
                "Cx_total": critic_x_loss[0],
                "Cx_valid": critic_x_loss[1],
                "Cx_fake": critic_x_loss[2],
                "Cx_gp_penalty": critic_x_loss[3],

                "Cz_total": critic_z_loss[0],
                "Cz_valid": critic_z_loss[1],
                "Cz_fake": critic_z_loss[2],
                "Cz_gp_penalty": critic_z_loss[3],

                "EG_total": encoder_generator_loss[0],
                "EG_fake_gen_x": encoder_generator_loss[1],
                "EG_fake_gen_z": encoder_generator_loss[2],
                "G_rec": encoder_generator_loss[3],
            }
        else:
            loss_dict = {
                "Cx_total": critic_x_loss[0],
                "Cz_total": critic_z_loss[0],
                "EG_total": encoder_generator_loss[0]
            }
        return loss_dict
        # return {"critic_x_loss": critic_x_loss, "critic_z_loss": critic_z_loss,
        #         "encoder_generator_loss": encoder_generator_loss}

    # to be implemented
    # @tf.function
    # def test_step(x, y):

    def predict(self, X):
        """
        Predict values using the initialized object.

        :return: y_hat: reconstructed X, critic: critic_x score of X
        """

        # X = X.reshape((-1, self.shape[0], 1))
        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(X)

        return y_hat, critic


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
