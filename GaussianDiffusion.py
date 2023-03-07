import numpy as np
import tensorflow as tf

class GaussianDiffusion:
    def __init__(self, beta_start, beta_end, time_steps, clip_min=-1.0, clip_max=1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_steps = int(time_steps)
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Define the betas schedule
        self.betas = betas = self.linear_beta_scheduler(beta_start, beta_end, time_steps)      ########## Beta schedule
        # Calcuate alphas
        alphas = 1- self.betas
        # Set up the basic variables
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Set up the next level of variables
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)

        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32)
        self.sqrt_recip_alphas = tf.constant(np.sqrt(1 / alphas), dtype=tf.float32)

        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1. - alphas_cumprod)
        pass

    def linear_beta_scheduler(self, beta_start, beta_end, time_steps):
        return np.linspace(beta_start, beta_end, time_steps, np.float32)

    def extract(self, a, t, x_shape):
        # Get the relevant tensor at a certain time step
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    # Forward diffusion
    def q_sample(self, x_start, t, noise):
        x_start_shape = tf.shape(x_start)
        first_term = self.extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        second_term = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        noised_image = first_term + second_term
        return noised_image

    # Backward diffusion
    def p_sample(self, pred_noise, x, t):
        sqrt_recipt_alphas_t = self.extract(self.sqrt_recip_alphas, t, tf.shape(x))
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, tf.shape(x))
        betas_t = self.extract(self.betas, t, tf.shape(x))
        posterior_variance_t = self.extract(self.posterior_variance, t, tf.shape(x))
        model_mean = sqrt_recipt_alphas_t * (x - (betas_t * pred_noise) / sqrt_one_minus_alphas_cumprod_t)
        if t[0] == 0:
            return model_mean
        standard_deviation = tf.sqrt(posterior_variance_t)
        standard_deviation = tf.cast(standard_deviation, tf.float32)
        second_term = standard_deviation * tf.random.normal(shape=tf.shape(x))
        sampled_image = model_mean + second_term
        return sampled_image