import os
import argparse
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau


class Trainer:
    def __init__(self, options):
        self.options = options
        self.callbacks = None
        self.train_gen = None
        self.valid_gen = None
        self.num_train_samples = Trainer.get_file_count(self.options.train_dir)
        self.num_valid_samples = Trainer.get_file_count(self.options.valid_dir)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def get_file_count(dir):
        count = 0
        for current_dir, dirs, files in os.walk(dir):
            count += len(files)

        return count

    def setup_callback(self):
        tb = keras.callbacks.TensorBoard(log_dir='./logs',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)
        log = CSVLogger('logs/training.log', append=False)
        es = EarlyStopping('val_acc', patience=self.options.patience)
        lr = ReduceLROnPlateau('val_acc', factor=0.1,
                               patience=int(self.options.patience / 4),
                               VERBOSE=1)
        model_names = 'models/model.hdf5'
        mc = ModelCheckpoint(model_names, monitor='val_acc',
                             verbose=1, save_best_only=True)

        self.callbacks = [
            tb,
            mc,
            log,
            es,
            lr
        ]

    def setup_generator(self):
        # prepare data augmentation configuration
        train_idg = ImageDataGenerator(rotation_range=self.options.rotate_range,
                                       width_shift_range=self.options.shift_range,
                                       height_shift_range=self.options.shift_range,
                                       zoom_range=self.options.zoom_range,
                                       horizontal_flip=self.options.horizontal_flip)
        valid_idg = ImageDataGenerator()

        self.train_gen = train_idg.flow_from_directory(self.options.train_dir,
                                                       (self.options.image_width, self.options.image_height),
                                                       batch_size=self.options.batch_size,
                                                       class_mode='categorical'
                                                       )
        self.valid_gen = valid_idg.flow_from_directory(self.options.valid_dir,
                                                       (self.options.image_width, self.options.image_height),
                                                       batch_size=self.options.batch_size,
                                                       class_mode='categorical'
                                                       )

    def train(self):
        self.setup_callback()
        self.setup_generator()

        # create resnet model with random initial weights
        model = ResNet50(weights=None, classes=3)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit_generator(
            self.train_gen,
            steps_per_epoch=self.num_train_samples / self.options.batch_size,
            validation_data=self.valid_gen,
            validation_steps=self.num_valid_samples / self.options.batch_size,
            epochs=self.options.epoch,
            callbacks=self.callbacks,
            verbose=self.options.verbose)


DEF_IMG_SIZE = 224

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a resnet 50 model "\
                                     "with your image dataset.')
    parser.add_argument('--image_width', default=DEF_IMG_SIZE)
    parser.add_argument('--image_height', default=DEF_IMG_SIZE)
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--shift_range', default=0.1, type=float)
    parser.add_argument('--rotate_range', default=10, type=float)
    parser.add_argument('--zoom_range', default=0.1, type=float)
    parser.add_argument('--horizontal_flip', action='store_true')
    parser.add_argument('--train_dir', default='image/train')
    parser.add_argument('--valid_dir', default='image/valid')
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--patience', default=5, type=int)
    options = parser.parse_args()

    t = Trainer(options)
    t.train()
