from .base import Layers
import tensorflow as tf


class BayesLadder(Layers):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.enc_ladder = dict()
        self.dec_ladder = dict()
        self.stoch_count = 0
        self.stoch_count_dec = 0

    def _latent(self, x):
        if x is None:
            mean = None
            stddev = None
            logits = None
            class_predictions = None
            z = self.epsilon
        else:
            enc_output = tf.reshape(x, [-1, self.flags['hidden_size'] * 2])
            mean, stddev = tf.split(1, 2, enc_output)  # Compute latent variables (z) by calculating mean, stddev
            stddev = tf.nn.softplus(stddev)
            with tf.variable_scope("y_network"):
                mlp = Layers(mean)
                mlp.fc(self.flags['num_classes'])
                logits = mlp.get_output()
                class_predictions = tf.nn.softmax(logits)
            z = (mean + self.epsilon * stddev) #* tf.cast(y_hat, tf.float32)
        return mean, stddev, class_predictions, logits, z

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', stoch=None, bn=True, test=False, activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0):
        """
        2D Convolutional Layer.
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        self.count['conv'] += 1
        scope = 'conv_' + str(self.count['conv'])
        with tf.variable_scope(scope):

            # Conv function
            input_channels = self.input.get_shape()[3]
            if filter_size == 0:  # outputs a 1x1 feature map; used for FCN
                filter_size = self.input.get_shape()[2]
                padding = 'VALID'
            output_shape = [filter_size, filter_size, input_channels, output_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding=padding)

            # Additional functions
            if stoch is not None:  # Draw feature map values from a normal distribution
                if stoch == 'N':  # Normal
                    output_shape = [3, 3, output_channels, 1]
                    w2 = self.weight_variable(name='weights_mean', shape=output_shape)
                    mean = tf.nn.conv2d(self.input, w2, strides=[1, 1, 1, 1], padding=padding)
                    w3 = self.weight_variable(name='weights_std', shape=output_shape)
                    std = tf.nn.conv2d(self.input, w3, strides=[1, 1, 1, 1], padding=padding)
                    map_size = tf.pack([mean.get_shape()[1], mean.get_shape()[2]])
                    z = mean + tf.random_normal(map_size) * std
                if stoch == 'B':  # Bernoulli
                    mean = 0
                    map_size = tf.pack([mean.get_shape()[1], mean.get_shape()[2]])
                with tf.variable_scope("stoch"):
                    output_shape = tf.pack([self.input.get_shape()[1], self.input.get_shape()[2], 1, 1])
                    w3 = self.weight_variable(name='weights_refinement', shape=output_shape)
                    self.input = self.input + z * w3
            if bn is True:  # batch normalization
                self.input = self.batch_norm(self.input)
            if b_value is not None:  # bias value
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:  # scale value
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = tf.mul(self.input, s)
            if activation_fn is not None:  # activation function
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def deconv2d(self, filter_size, output_channels, stride=1, padding='SAME', stoch=False, ladder=None,
                 activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0, bn=True):
        """
        2D Deconvolutional Layer
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        self.count['deconv'] += 1
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):

            # Calculate the dimensions for deconv function
            batch_size = tf.shape(self.input)[0]
            input_height = tf.shape(self.input)[1]
            input_width = tf.shape(self.input)[2]

            if padding == "VALID":
                out_rows = (input_height - 1) * stride + filter_size
                out_cols = (input_width - 1) * stride + filter_size
            else:  # padding == "SAME":
                out_rows = input_height * stride
                out_cols = input_width * stride

            # Deconv function
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, output_channels, input_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            deconv_out_shape = tf.pack([batch_size, out_rows, out_cols, output_channels])
            self.input = tf.nn.conv2d_transpose(self.input, w, deconv_out_shape, [1, stride, stride, 1], padding)

            # Additional functions
            if ladder is not None:
                stoch = False
                enc_mean = self.dec_ladder[self.stoch_count_dec][0]
                enc_std = self.dec_ladder[self.stoch_count_dec][1]
                self.stoch_count_dec += 1
                with tf.variable_scope("ladder"):
                    input_shape = [enc_mean.get_shape()[1], enc_mean.get_shape()[2], enc_mean.get_shape()[3]]
                    w_std = self.weight_variable(name='weights_mean', shape=input_shape)
                    w_mean = self.weight_variable(name='weights_std', shape=input_shape)
                    mean = self.input * w_mean
                    std = tf.nn.softplus(self.input * w_std)
                    if ladder == 1:  # LVAE Implementation
                        eps = 1e-10
                        new_std = 1 / ((enc_std + eps) ** 2 + (std + eps) ** 2)
                        new_mean = new_std * (enc_mean * (1 / (enc_std + eps) ** 2) + mean * (1 / (std + eps) ** 2))
                        self.input = new_mean + tf.random_normal(tf.shape(self.input)) * new_std
                    elif ladder == 2:  # BLN Implementation
                        raise NotImplementedError
                    else:
                        self.input = mean + tf.random_normal(tf.shape(self.input)) * std
            if stoch is True:  # Draw sample from Normal Layer
                mean, std = tf.split(3, 2, self.input)
                self.input = mean + tf.random_normal(tf.shape(mean)) * std
                output_channels = int(output_channels/2)
            if bn is True:  # batch normalization
                self.input = self.batch_norm(self.input)
            if b_value is not None:  # bias value
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:  # scale value
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = tf.mul(self.input, s)
            if activation_fn is not None:  # non-linear activation function
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))  # print shape of output




class Ladder(Layers):
    def __init__(self, x, layer_num=0, z_noisy_dict=dict(), clean_batch=dict()):
        super().__init__(x)
        self._noisy_z_dict = z_noisy_dict  # for ladder network
        self.clean_batch_dict = clean_batch  # for ladder network
        self._clean_z = dict()  # for ladder network
        self._z_hat_bn = dict()
        self.layer_num = layer_num  # only used for decoder
        self._layer_count = 0  # only used for encoder

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0, bn=True, stoch=False):
        """
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        self.count['conv'] += 1
        self._layer_count += 1
        scope = 'conv_' + str(self.count['conv'])
        if stoch is True:
            clean = False
        else:
            clean = True
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, input_channels, output_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding=padding)
            if bn is True:
                self.input = self.conv_batch_norm(self.input, clean=clean, count=self._layer_count)
            if stoch is True:
                self.input = tf.random_normal(tf.shape(self.input)) + self.input
                self._noisy_z_dict[self._layer_count] = self.input
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = tf.multiply(self.input, s)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def deconv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0, bn=True, ladder=False):
        self.count['deconv'] += 1
        self._layer_count += 1
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, output_channels, input_channels]
            w = self.weight_variable(name='weights', shape=output_shape)

            batch_size = tf.shape(self.input)[0]
            input_height = tf.shape(self.input)[1]
            input_width = tf.shape(self.input)[2]
            filter_height = tf.shape(w)[0]
            filter_width = tf.shape(w)[1]
            out_channels = tf.shape(w)[2]
            row_stride = stride
            col_stride = stride

            if padding == "VALID":
                out_rows = (input_height - 1) * row_stride + filter_height
                out_cols = (input_width - 1) * col_stride + filter_width
            else:  # padding == "SAME":
                out_rows = input_height * row_stride
                out_cols = input_width * col_stride

            out_shape = tf.stack([batch_size, out_rows, out_cols, out_channels])

            self.input = tf.nn.conv2d_transpose(self.input, w, out_shape, [1, stride, stride, 1], padding)
            if bn is True:
                self.input = self.conv_batch_norm(self.input)
                if ladder is True:
                    s_value = None
                    noisy_z_ind = self.layer_num - self.count['deconv'] - self.count['fc']
                    noisy_z = self._noisy_z_dict[noisy_z_ind]
                    z_hat = self.ladder_g_function(noisy_z, self.input)
                    self._z_hat_bn[noisy_z_ind] = (z_hat - self.clean_batch_dict[noisy_z_ind][0]) / self.clean_batch_dict[noisy_z_ind][1]
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = tf.multiply(self.input, s)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def fc(self, output_nodes, keep_prob=1, activation_fn=tf.nn.relu, b_value=0.0, s_value=None, bn=False, stoch=False, ladder=False, clean=False):
        self.count['fc'] += 1
        self._layer_count += 1
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):
            input_nodes = self.input.get_shape()[1]
            output_shape = [input_nodes, output_nodes]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.matmul(self.input, w)
            if bn is True:
                self.input = self.batch_norm(self.input, clean=clean, count=self._layer_count)
                if ladder is True:
                    b_value = s_value = None
                    noisy_z_ind = self.layer_num - self.count['deconv'] - self.count['fc']
                    noisy_z = self._noisy_z_dict[noisy_z_ind]
                    z_hat = self.ladder_g_function(noisy_z, self.input)
                    self._z_hat_bn[noisy_z_ind] = (z_hat - self.clean_batch_dict[noisy_z_ind][0]) / self.clean_batch_dict[noisy_z_ind][1]
            if stoch is True:
                self.input = tf.random_normal(tf.shape(self.input)) + self.input
                self._noisy_z_dict[self._layer_count] = self.input
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_nodes], value=b_value)
                self.input = tf.add(self.input, b)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_nodes], value=s_value)
                self.input = tf.multiply(self.input, s)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
            if keep_prob != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def ladder_g_function(self, noisy_z, u):
        shape = [noisy_z.get_shape()[3]]
        with tf.variable_scope('ladder'):
            a_1 = self.const_variable(name='a_1', shape=shape, value=1.0)
            a_2 = self.const_variable(name='a_2', shape=shape, value=1.0)
            a_3 = self.const_variable(name='a_3', shape=shape, value=0.0)
            a_4 = self.const_variable(name='a_4', shape=shape, value=1.0)
            a_5 = self.const_variable(name='a_5', shape=shape, value=1.0)
            mu = a_1 * tf.nn.sigmoid(a_2 * u + a_3) + a_4 * u + a_5

            a_6 = self.const_variable(name='a_6', shape=shape, value=1.0)
            a_7 = self.const_variable(name='a_7', shape=shape, value=1.0)
            a_8 = self.const_variable(name='a_8', shape=shape, value=0.0)
            a_9 = self.const_variable(name='a_9', shape=shape, value=1.0)
            a_10 = self.const_variable(name='a_10', shape=shape, value=1.0)
            nu = a_6 * tf.nn.sigmoid(a_7 * u + a_8) + a_9 * u + a_10
        return (noisy_z - mu) * nu + mu

    def batch_norm(self, x, epsilon=1e-3, clean=False, count=1):
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(x, [0], keep_dims=True)

        # Apply the initial batch normalizing transform
        z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        if clean is True:
            self.clean_batch_dict[count] = (tf.squeeze(batch_mean1), tf.squeeze(batch_var1))
            self._clean_z[count] = z1_hat
        return z1_hat

    def conv_batch_norm(self, x, epsilon=1e-3, clean=False, count=1):
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(x, [0, 1, 2], keep_dims=True)

        # Apply the initial batch normalizing transform
        z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        if clean is True:
            self.clean_batch_dict[count] = (tf.squeeze(batch_mean1), tf.squeeze(batch_var1))
            self._clean_z[count] = z1_hat
        return z1_hat

    @property
    def z_hat_bn(self):
        return self._z_hat_bn

    @property
    def clean_z(self):
        return self._clean_z

    @property
    def clean_batch(self):
        return self.clean_batch_dict

    @property
    def layer_count(self):
        return self._layer_count

    @property
    def noisy_z(self):
        return self._noisy_z_dict

