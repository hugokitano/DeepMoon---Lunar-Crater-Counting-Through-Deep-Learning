
from .base import Layers
from .stoch import GaussianLayer, MultinomialLayer, BernoulliLayer
import tensorflow as tf


class AuxDeepGenMod:
    def __init__(self, x_l, t_l, x_u, num_classes=10):
        self.x_l, self.t_l = x_l, t_l
        self.beta = 0.5
        self.batch_size = int(x_l.get_shape()[0]/2)
        self.num_classes = num_classes
        self.x_u = x_u
        self.x_u_rep, self.t_u_rep = self.unlabeled_data(x_u)
        self.samples = {'qa_l': None, 'qz_l': None, 'qy_l': None, 'qa_u': None, 'qz_u': None, 'qy_u': None}
        self.qa_l, self.qa_u = self.q_a_x()
        self.qy_l, self.qy_u = self.q_y_ax()
        self.qz_l, self.qz_u = self.q_z_xay()
        self.pa_l, self.pa_u = self.p_a_zy()
        self.px_l, self.px_u = self.p_x_azy()

    def unlabeled_data(self, x_u):
        # repeat data
        x_u = tf.tile(x_u, [self.num_classes, 1])
        nums = tf.range(0, self.num_classes, 1)
        _, t_u = tf.meshgrid(tf.zeros(self.batch_size, dtype=tf.int32), nums)
        return x_u, t_u

    def q_a_x(self):
        with tf.variable_scope('q_a_x'):
            q_a_x_l = Layers(self.x_l)
            q_a_x_l.fc(500).fc(500)
            qa_l = GaussianLayer(q_a_x_l.get_output(), num_latent=100, eq_samples=10)
            self.samples['qa_l'] = qa_l.compute_samples()
        with tf.variable_scope('q_a_x', reuse=True):
            q_a_x_u = Layers(self.x_u)
            q_a_x_u.fc(500).fc(500)
            qa_u = GaussianLayer(q_a_x_u.get_output(), num_latent=100, eq_samples=10)
            q_a_samples = qa_u.compute_samples()
            self.samples['qa_u'] = tf.tile(q_a_samples, [self.num_classes, 1])
        return qa_l, qa_u

    def q_y_ax(self):
        with tf.variable_scope('q_y_ax'):
            x_to_qy_l = Layers(self.x_l)
            x_to_qy_l.fc(500, activation_fn=None)
            q_y_ax_l = Layers(x_to_qy_l.get_output() + self.samples['qa_l'])
            q_y_ax_l.fc(500).fc(500)
            qy_l = MultinomialLayer(q_y_ax_l.get_output(), num_latent=100, eq_samples=10)
            self.samples['qy_l'] = qy_l.compute_samples()
        with tf.variable_scope('q_y_ax', reuse=True):
            x_to_qy_u = Layers(self.x_u_rep)
            x_to_qy_u.fc(500, activation_fn=None)
            q_y_ax_u = Layers(x_to_qy_u.get_output() + self.samples['qa_u'])
            q_y_ax_u.fc(500).fc(500)
            qy_u = MultinomialLayer(q_y_ax_u.get_output(), num_latent=100, eq_samples=10)
            self.samples['qy_u'] = qy_u.compute_samples()
        return qy_l, qy_u

    def q_z_xay(self):
        with tf.variable_scope('q_z_xay'):
            x_to_qz_l = Layers(self.x_l)
            x_to_qz_l.fc(500, activation_fn=None)
            y_to_qz_l = Layers(self.t_l)
            y_to_qz_l.fc(500, activation_fn=None)
            qa_to_qz_l = Layers(self.samples['qa_l'])
            qa_to_qz_l.fc(500, activation_fn=None)
            q_z_xay_l = Layers(x_to_qz_l.get_output() + y_to_qz_l.get_output() + qa_to_qz_l.get_output())
            q_z_xay_l.fc(500).fc(500)
            qz_l = GaussianLayer(q_z_xay_l.get_output(), num_latent=100, eq_samples=10)
            self.samples['qz_l'] = qz_l.compute_samples()
        with tf.variable_scope('q_z_xay', reuse=True):
            x_to_qz_u = Layers(self.x_u_rep)
            x_to_qz_u.fc(500, activation_fn=None)
            y_to_qz_u = Layers(self.t_u_rep)
            y_to_qz_u.fc(500, activation_fn=None)
            qa_to_qz_u = Layers(self.samples['qa_u'])
            qa_to_qz_u.fc(500, activation_fn=None)
            q_z_xay_u = Layers(x_to_qz_u.get_output() + y_to_qz_u.get_output() + qa_to_qz_u.get_output())
            q_z_xay_u.fc(500).fc(500)
            qz_u = GaussianLayer(q_z_xay_u.get_output(), num_latent=100, eq_samples=10)
            self.samples['qz_u'] = qz_u.compute_samples()
        return qz_l, qz_u

    def p_a_zy(self):
        with tf.variable_scope('p_a_zy'):
            y_to_pa_l = Layers(self.t_l)
            y_to_pa_l.fc(500, activation_fn=None)
            qz_to_pa_l = Layers(self.samples['qz_l'])
            qz_to_pa_l.fc(500, activation_fn=None)
            p_a_zy = Layers(y_to_pa_l.get_output() + qz_to_pa_l.get_output())
            pa_l = GaussianLayer(p_a_zy.get_output(), num_latent=100, eq_samples=10)
            self.samples['pa_l'] = pa_l.compute_samples()
        with tf.variable_scope('p_a_zy', reuse=True):
            y_to_pa_u = Layers(self.t_u)
            y_to_pa_u.fc(500, activation_fn=None)
            qz_to_pa_u = Layers(self.samples['qz_u'])
            qz_to_pa_u.fc(500, activation_fn=None)
            p_a_zy = Layers(y_to_pa_u.get_output() + qz_to_pa_u.get_output())
            pa_u = GaussianLayer(p_a_zy.get_output(), num_latent=100, eq_samples=10)
            self.samples['pa_u'] = pa_u.compute_samples()
            return pa_l, pa_u

    def p_x_azy(self):
        with tf.variable_scope('p_x_azy'):
            a_to_px_l = Layers(self.samples['qa_l'])
            a_to_px_l.fc(500, activation_fn=None)
            z_to_px_l = Layers(self.samples['qz_l'])
            z_to_px_l.fc(500, activation_fn=None)
            y_to_px_l = Layers(self.t_l)
            y_to_px_l.fc(500, activation_fn=None)
            p_x_azy_l = Layers(a_to_px_l.get_output() + z_to_px_l.get_output() + y_to_px_l.get_output())
            p_x_azy_l.fc(500).fc(500)
            px_l = BernoulliLayer(p_x_azy_l.get_output(), num_latent=784, eq_samples=10)
            self.samples['px_l'] = px_l.compute_samples()
        with tf.variable_scope('p_x_azy'):
            a_to_px_u = Layers(self.samples['qa_u'])
            a_to_px_u.fc(500, activation_fn=None)
            z_to_px_u = Layers(self.samples['qz_u'])
            z_to_px_u.fc(500, activation_fn=None)
            y_to_px_u = Layers(self.t_u_rep)
            y_to_px_u.fc(500, activation_fn=None)
            p_x_azy_u = Layers(a_to_px_u.get_output() + z_to_px_u.get_output() + y_to_px_u.get_output())
            p_x_azy_u.fc(500).fc(500)
            px_u = BernoulliLayer(p_x_azy_u.get_output(), num_latent=784, eq_samples=10)
            self.samples['px_u'] = px_u.compute_samples()
        return px_u, px_u

    def bound(self):

        # labeled data
        log_qa = self.qa_l.neg_log_likelihood(self.samples['qa_l'])
        log_qz = self.qz_l.neg_log_likelihood(self.samples['qz_l'])
        log_qy = self.qy_l.neg_log_likelihood(self.samples['qy_l'])
        log_pz = self.qz_l.neg_log_likelihood(self.samples['qz_l'], standard=True)
        log_pa = self.pa_l.neg_log_likelihood(self.samples['qa_l'])
        log_px = self.px_l.neg_log_likelihood(self.x_l)

        logits = tf.zeros([self.batch_size, self.num_classes])
        log_py_l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.t_l)
        log_py_l = tf.expand_dims(tf.expand_dims(log_py_l, 1), 1)
        lb_l = log_px + log_py_l + log_pz + log_pa - log_qa - log_qz
        lb_l = tf.reduce_mean(lb_l, axis=(1,2))
        lb_l -= tf.reduce_mean(log_qy * self.beta, axis=(1, 2))

        # q_a_x



        # unlabeled data
        log_qa = self.qa_u.neg_log_likelihood(self.samples['qa_u'])
        log_qz = self.qz_u.neg_log_likelihood(self.samples['qz_u'])
        log_qy = self.qy_u.neg_log_likelihood(self.samples['qy_u'])
        log_pz = self.qz_u.neg_log_likelihood(self.samples['qz_u'], standard=True)
        log_pa = self.pa_u.neg_log_likelihood(self.samples['qa_u'])
        log_px = self.px_u.neg_log_likelihood(self.x_u)

        logits = tf.zeros([self.batch_size, self.num_classes])
        log_py_l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.t_u)
        lb_u = log_px + log_py_l + log_pz + log_pa - log_qa - log_qz
        lb_u = tf.reduce_mean(lb_u, axis=(1,2))
        y_u = tf.reduce_mean(self.samples['qy_u'], axis=(1, 2))
        y_u += 1e-8
        y_u /= tf.reduce_sum(y_u, axis=1
        lb_u = tf.reduce_sum(y_u (lb_u - tf.log(y_u)), axis=1)
        lb_l = log_px + log_py_l + log_pz + log_pa - log_qa - log_qz
        # elbo
        elbo = tf.reduce_mean(lb_l) + tf.reduce_mean(lb_u)