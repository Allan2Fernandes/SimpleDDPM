import tensorflow as tf

class DatasetBuilder:
    def __init__(self, directory_path, img_size, batch_size):
        self.directory_path = directory_path
        self.img_size = img_size
        self.batch_size = batch_size
        pass

    def preprocess_dataset(self, image):
        image = image / 127.5 - 1.0
        return image

    def build_dataset(self):
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.directory_path,
            labels=None,
            label_mode=None,
            class_names=None,
            color_mode='rgb',
            batch_size=None,
            image_size=(self.img_size, self.img_size),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False)

        self.dataset = dataset.map(self.preprocess_dataset).batch(batch_size=self.batch_size, drop_remainder=True).prefetch(buffer_size=1)
        pass

    def get_dataset(self):
        return self.dataset

