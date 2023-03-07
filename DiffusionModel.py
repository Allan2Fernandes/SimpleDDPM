import keras
import tensorflow as tf
import time
import matplotlib.pyplot as plt

class DiffusionModel():
    def __init__(self, network, time_steps, gdf_util, learning_rate, img_size, img_channels, clip_min, clip_max):
        self.network = network
        self.time_steps = time_steps
        self.gdf_util = gdf_util
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.img_channels = img_channels
        self.clip_min = clip_min
        self.clip_max = clip_max
        pass

    def generate_images(self, num_images=2):
        # Generate the starting point - noisy images
        samples = tf.random.normal(shape=(num_images, self.img_size, self.img_size, self.img_channels), dtype=tf.float32)
        # Incremental sampling process
        for t in reversed(range(0, self.time_steps)):
            # Get time steps of correct shape
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int32)
            # Using the time step and samples, predict the noise in the samples
            pred_noise = self.network([samples, tt])
            # Get the images from the samples + predicted noise
            samples = self.gdf_util.p_sample(pred_noise, samples, tt)
            if t % 50 == 0:
                print(f"Time step: {self.time_steps - t}")
            pass
        # Clip the image
        samples = tf.clip_by_value(samples, clip_value_min=self.clip_min, clip_value_max=self.clip_max)
        return samples

    def train_step(self, images, step, epoch, num_batches):
        step_start_time = time.time()
        # Get the batch size
        batch_size = images.shape[0]
        # Get time steps for the whole batch
        t = tf.random.uniform(minval=0, maxval=self.time_steps, shape=(batch_size,), dtype=tf.int32)

        with tf.GradientTape() as tape:
            # Create noise
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            # Create the input noisy images by forward diffusion
            images_t = self.gdf_util.q_sample(images, t, noise)
            # Pass them through the model to predict the noise in the images
            pred_noise = self.network([images_t, t], training=True)
            # Calculate loss by comparing pred noise to noise
            loss = self.loss_fn(y_true=noise, y_pred=pred_noise)
            pass
        # Calculate gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)
        # Gradient descent
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        step_training_time = time.time()-step_start_time
        est_epoch_training_time = step_training_time*num_batches
        print("Epoch = {2} || Step = {0} || loss = {1} || Step training time = {3} || Estimated epoch training time = {4}".format(step, loss, epoch, step_training_time, est_epoch_training_time))
        return loss

    def plot_images(self, images, title):

        # Get one of the images
        image = images[1]
        image = (image + 1) / 2
        image = image * 255
        image = tf.cast(image, tf.uint16)
        plt.imshow(image)
        plt.title(title)
        plt.show()

    def train_model(self, dataset, epochs):
        num_batches = len(dataset)
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            # self.learning_rate = tf.math.pow(0.99, (epoch-1))*self.learning_rate
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            epoch_start_time = time.time()
            for step, images in enumerate(dataset):
                step_loss = self.train_step(images=images, step=step, epoch=epoch, num_batches=num_batches)
                epoch_losses.append(step_loss)
                print("Mean loss = {0}".format(tf.reduce_mean(epoch_losses)))
                pass
            epoch_end_time = time.time()
            epoch_loop_time = epoch_end_time - epoch_start_time
            print("Time per epoch = {0}".format(epoch_loop_time))
            if epoch % 5 == 0:
                # Generate images from time steps
                images = self.generate_images(2)
                self.plot_images(images, f"Epoch: {epoch}")
                self.network.save(f"64pxModelsRandomUniformMyAttention/Epoch{epoch}")