import numpy as np
import tensorflow as tf
from tools.config import FLAGS

class DAE_Layer(object):
    
    def __init__(self, in_data=None, prev_layer_size=None, next_layer_size=None, nth_layer=None, sess=None, last_layer=True):
        self._is_last = last_layer
        self._layer = nth_layer
        
        self._prev_layer_size = prev_layer_size
        self._next_layer_size = next_layer_size
        self._shape = [self._prev_layer_size, self._next_layer_size]

        self._x = in_data
        
        self._l_rate = self._get_l_rate
        self._noise = self._get_noise

        self.vars_to_init = self._setup_variables()

    
    def _setup_variables(self):
        with tf.name_scope("Initialize_Variables"):
            self._w = self._init_w_or_b(shape=self._shape, trainable=True, name='weights')#_{0}'.format(self._layer))
#             lmt = tf.mul(4.0, tf.sqrt(6.0 / (self._shape[0] + self._shape[1])))
#             self._w = tf.Variable(tf.random_uniform(self._shape, -1*lmt, lmt), trainable=True, name='weights')
            self._b_y = self._init_w_or_b(shape=[self._next_layer_size], trainable=True, is_bias=True, name='prev_biases')
            
            vars_to_init = [self._w, self._b_y]
            if not self._is_last:
                self._fixed_w = tf.Variable(tf.identity(self._w.initialized_value()), trainable=False, name="weights_fixed")
                self._fixed_b = tf.Variable(tf.identity(self._b_y.initialized_value()), trainable=False, name="biases_fixed")
                self._b_z = self._init_w_or_b(shape=[self._prev_layer_size], trainable=True, is_bias=True, name='next_biases')
                vars_to_init.append(self._fixed_w)
                vars_to_init.append(self._fixed_b)
                vars_to_init.append(self._b_z)

        return vars_to_init
    
        
        """ TODO: TRY initialization for different functions (e.g. tanh) """
    def _init_w_or_b(self, shape, trainable=True, name=None, is_bias=False, method='sigmoid'):
#         with tf.name_scope("dae_{0}_{1}".format(self._layer, name)):
        if is_bias:
            return tf.Variable(tf.zeros(shape), trainable=trainable, name=name)
        
        if method=='sigmoid':
            # Upper and Lower limit for the weights
            lmt = tf.mul(4.0, tf.sqrt(6.0 / (shape[0] + shape[1])))
            return tf.Variable(tf.random_uniform(shape, -1*lmt, lmt), trainable=trainable, name=name)


    def clean_activation(self, x_in=None, use_fixed=True):
        if x_in is None:
            x = self._x
        else:
            x = x_in
        if use_fixed:
            return self._activate(x, self._fixed_w, self._fixed_b, name='Latent_layer_next')
        else:
            return self._activate(x, self._w, self._b_y, name='Latent_layer_next')
    

    def encode(self, x_in=None, noise=None):
        if x_in is None:
            x = self._x
        else:
            x = x_in
        
        if noise is None:
            ratio = self._noise[0]
            ntype = self._noise[1]
        else:
            ratio = noise[0]
            ntype = noise[1]

        self._x_tilde, self._noise_map = self._corrupt(x, ratio=ratio, n_type=ntype)
        with tf.name_scope("Encoder"):
            self._y = self._activate(self._x_tilde, self._w, self._b_y, name='Latent_layer_next')
        return self._y
    
    def decode(self):
#         self._y = self.encode()
        with tf.name_scope("Decoder"):
            y = self.encode()
            if self._is_last:
                exit("This is the last layer. Currently the reconstruction of this layer cannot be done.")
            self._z = self._activate(y, self._w, self._b_z, transpose_w=True, name='Reconstr_layer_{0}'.format(self._layer))
        return self._z

    @property
    def get_loss(self):
        z = self.decode()
        noise_map = None
        
        if FLAGS.emphasis:
            noise_map = self._noise_map

        loss = self._loss_x_entropy(x=self._x, z=z, noise=noise_map)
        
        return loss
        
#     def __call__(self):
#         cost = self.get_cost
#         return cost, self.train(cost=cost)
        
    @property
    def get_w_all_b(self):
        return [self._w, self._b_y, self._b_z]
    
    @property
    def get_w_b(self):
        return [self._w, self._b_y]
    
    @property
    def get_w(self):
        return self._w

    @property
    def get_fixed_w(self):
        return self._fixed_w
    
#     @get_w.setter
#     def set_w(self, w):
#         self._w = tf.Variable(tf.zeros(self._shape), trainable=True, name='given_weights')
#         update = tf.assign(self._w, w, name='external_w')
# #         self._sess.run(tf.initialize_variables([self._w]))
#         self._sess.run(update)

    @property
    def get_b(self):
        return self._b_y
    
    @property
    def get_fixed_b(self):
        return self._fixed_b

#     @get_b_repr.setter
#     def set_b(self, b):
#         self._b_y = tf.Variable(tf.zeros([self._next_layer_size]), trainable=True, name='given_biases')
#         update = tf.assign(self._b_y, b, name='external_b')
# #         self._sess.run(tf.initialize_variables([self._b_y]))
#         self._sess.run(update)

    @property
    def get_b_recon(self):
        return self._b_z
    
    @property
    def get_representation_y(self):
        return self._y
    
    @property
    def get_reconstruction_z(self):
        return self._z
    
    @property
    def which(self):
        return self._layer - 1
    
    @staticmethod
    def _activate(x, w, b, transpose_w=False, name=None):
        """ TODO: TRY different activation functions (e.g. tanh, sigmoid...) """
        return tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b), name=name)

    @property
    def _get_noise(self):
        assert self._layer >= 0

        try:
            return getattr(FLAGS, "noise_{0}".format(self._layer))
        except AttributeError:
            print "Noise out of bounds. Using default noise for this Layer (Layer {0})".format(self._layer)
            return FLAGS.default_noise
    
    @property
    def _get_l_rate(self):
        return getattr(FLAGS, "unsupervised_learning_rate")

    @property
    def _get_emph_params(self):
        if FLAGS.emphasis_type == 'Full':
            return 1, 0
        elif FLAGS.emphasis_type == 'Double':
            return 1, 0.5
        else:
            print("Unspecified/Wrong Emphasis type. Default Full [0-1] is used.")
            return 1, 0

    def _loss_x_entropy(self, x, z, noise=None):
        with tf.name_scope("xentropy_loss"):
            z_clipped = tf.clip_by_value(z, FLAGS.zero_bound, FLAGS.one_bound)
            z_minus_1_clipped = tf.clip_by_value((1.0 - z), FLAGS.zero_bound, FLAGS.one_bound)
            x_clipped = tf.clip_by_value(x, FLAGS.zero_bound, FLAGS.one_bound)
            x_minus_1_clipped = tf.clip_by_value((1.0 - x), FLAGS.zero_bound, FLAGS.one_bound)
            
            # cross_entropy = x * log(z) + (1 - x) * log(1 - z)
            
            cross_entropy = tf.add(tf.mul(tf.log(z_clipped), x_clipped),
                                   tf.mul(tf.log(z_minus_1_clipped), x_minus_1_clipped), name='X-Entr')

            if noise:
                with tf.name_scope("Given_Emphasis"):
                    a, b = self._get_emph_params
                    corrupted = tf.select(noise, cross_entropy, tf.zeros_like(cross_entropy), name='Corrupted_Emphasis')
                    
                    # OR -- tf.select(tf.logical_not(noisy_points), cross_entropy, tf.zeros_like(cross_entropy), name='Uncorrupted_Emphasis')
                    uncorrupted = tf.select(noise, tf.zeros_like(cross_entropy), cross_entropy, name='Uncorrupted_Emphasis')
                    
                    loss = a * (-1 * tf.reduce_sum(corrupted, 1)) + b * (-1 * tf.reduce_sum(uncorrupted, 1))
            else:
                # Sum the cost for each example
                loss = -1 * tf.reduce_sum(cross_entropy, 1)
        
            # Reduce mean to find the overall cost of the loss
            cross_entropy_mean = tf.reduce_mean(loss, name='xentropy_mean')
    
            return cross_entropy_mean


#     @property
#     def get_cost(self):
#         z = self.get_reconstruction_z
#         noise_map = None
#         
#         if FLAGS.emphasis:
#             noise_map = self._noise_map
# 
#         cost = self._loss_x_entropy(x=self._x, z=z, noise=noise_map)
#         
#         return cost


    def _corrupt(self, x, ratio, n_type='MN'):
        with tf.name_scope("Corruption"):
            """ Noise adding (or input corruption)
            This function adds noise to the given data.
            
            Args:
                x    : The input data for the noise to be applied
                ratio: The percentage of the data affected by the noise addition
                n_type: The type of noise to be applied.
                        Choices: MN (masking noise), SP (salt-and-pepper noise)
            """
            
            # Safety check. If unspecified noise type given, use Masking noise instead.
            if n_type != 'MN' and n_type != 'SP':
                n_type = 'MN'
                print("Unknown noise type. Masking noise will be used instead.")
            
            # It makes a copy of the data, otherwise 'target_feed' will also be affected
    #         x_cp = np.copy(x)
            x_tilde = tf.identity(x, name='X_tilde')
            shape = tf.Tensor.get_shape(x_tilde)
            # Creating and applying random noise to the data. (Masking noise)
            points_to_alter = tf.random_uniform(shape=shape, dtype=tf.float32) < ratio
            
            if n_type == 'MN':
                x_tilde = tf.select(points_to_alter, tf.add(tf.zeros_like(x_tilde, dtype=tf.float32),
                                                            FLAGS.zero_bound), x_tilde, name='X_tilde')
                
            elif n_type == 'SP':
                coin_flip = np.asarray([np.random.choice([FLAGS.zero_bound, FLAGS.one_bound]) for _ in range(shape[0]) for _ in range(shape[1])]).reshape(shape)
                x_tilde = tf.select(points_to_alter, tf.to_float(coin_flip), x_tilde, name='X_tilde')
    
            # Also returns the 'points_to_alter' in case of applied Emphasis
            if not FLAGS.emphasis:
                points_to_alter = None
    
            return x_tilde, points_to_alter

        