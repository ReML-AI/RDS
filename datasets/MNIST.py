import os
import numpy as np
import tensorflow as tf
import tqdm
from skimage import feature


class MNIST:

    def load(self, loc="policy"):
        paths = {"data_x": "./datasets/MNIST_X.npy", "data_y": "./datasets/MNIST_Y.npy", "policy_x": "./datasets/MNIST_X_pca.npy"}
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print("#INFO: default split: {}, {}, {}, {}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        X_train_all = np.concatenate((x_train, x_test), axis=0)
        X_train_all = X_train_all.astype('float32')

        y_train_all = np.concatenate((y_train, y_test), axis=0)
        del x_train, y_train, x_test, y_test

        # convert y to one-hot
        data_y_one_hot = np.zeros((y_train_all.size, y_train_all.max() + 1))
        data_y_one_hot[np.arange(y_train_all.size), y_train_all] = 1
        data_y_one_hot = data_y_one_hot.astype('uint8')
        print("#INFO: check data_y_one_hot ...")

        X_train_all = X_train_all.reshape(X_train_all.shape[0], 28, 28, 1)

        # Hog features for policy
        if os.path.exists(paths["policy_x"]):
            policy_X_features_np = np.load(paths["policy_x"])
        else:
            hog = HOG(orientations=3, pixelsPerCell=(2, 2), cellsPerBlock=(4, 4), block_norm='L2-Hys')
            policy_X_features_np = np.zeros(shape=(X_train_all.shape[0], 5808))
            for idx in tqdm.tqdm(range(X_train_all.shape[0])):
                image = X_train_all[idx]
                hog_hist = hog.describe(image)
                policy_X_features_np[idx] = hog_hist
            np.save(paths["policy_x"], policy_X_features_np)

        input_shape = (28, 28, 1)

        print('CNN X_features shape:', X_train_all.shape)
        print('y_all shape:', data_y_one_hot.shape)

        if not os.path.exists(paths["data_x"]):
            np.save(paths["data_x"], X_train_all)

        if not os.path.exists(paths["data_y"]):
            np.save(paths["data_y"], data_y_one_hot)

        if loc == "policy":
            return policy_X_features_np, data_y_one_hot
        else:
            return X_train_all, data_y_one_hot


# Histogram of Oriented Gradients
class HOG:
	def __init__(self, orientations=9, pixelsPerCell=(9, 9),
		cellsPerBlock=(3, 3), block_norm='L2-Hys'):
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		# changing from default to L2-Hys, improved a lot
		self.block_norm = block_norm

	def describe(self, image):
		# compute HOG for the image
		hist = feature.hog(image, orientations = self.orientations,
			pixels_per_cell = self.pixelsPerCell,
			cells_per_block = self.cellsPerBlock,
			block_norm = self.block_norm)

		# return the HOG features
		return hist