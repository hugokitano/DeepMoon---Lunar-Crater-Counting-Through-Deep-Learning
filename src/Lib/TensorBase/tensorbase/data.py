#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/5/2016
Purpose: Class for MNIST
Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
"""

from .base import Data
from shutil import copyfile
import numpy as np
import os
import gzip
import urllib.request


class Mnist(Data):
    def __init__(self, flags):
        super().__init__(flags)

    def load_data(self, test_percent=0.15):
        one_hot = True
        validation_size = 5000

        SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
        TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
        TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
        TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

        local_file = self.maybe_download(TRAIN_IMAGES, self.flags['data_directory'], SOURCE_URL + TRAIN_IMAGES)
        with open(local_file, 'rb') as f:
            train_images = self.extract_images(f)

        local_file = self.maybe_download(TRAIN_LABELS, self.flags['data_directory'], SOURCE_URL + TRAIN_LABELS)
        with open(local_file, 'rb') as f:
            train_labels = self.extract_labels(f, one_hot=one_hot)

        local_file = self.maybe_download(TEST_IMAGES, self.flags['data_directory'], SOURCE_URL + TEST_IMAGES)
        with open(local_file, 'rb') as f:
            test_images = self.extract_images(f)

        local_file = self.maybe_download(TEST_LABELS, self.flags['data_directory'], SOURCE_URL + TEST_LABELS)
        with open(local_file, 'rb') as f:
            test_labels = self.extract_labels(f, one_hot=one_hot)

        if not 0 <= validation_size <= len(train_labels):
            raise ValueError(
                'Validation size should be between 0 and {}. Received: {}.'.format(len(train_labels), validation_size))
        return train_images, train_labels, test_images, test_labels

    def maybe_download(self, filename, work_directory, source_url):
        """Download the data from source url, unless it's already here.
        Args:
            filename: string, name of the file in the directory.
            work_directory: string, path to working directory.
            source_url: url to download from if file doesn't exist.
        Returns:
            Path to resulting file.
        """
        if not os.path.exists(work_directory):
            os.makedirs(work_directory)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            temp_file_name, _ = urllib.request.urlretrieve(source_url)
            copyfile(temp_file_name, filepath)
            print('Successfully downloaded', filename)
        return filepath

    def extract_images(self, f):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
        Args:
          f: A file object that can be passed into a gzip reader.
        Returns:
          data: A 4D unit8 numpy array [index, y, x, depth].
        Raises:
          ValueError: If the bytestream does not start with 2051.
        """
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                                 (magic, f.name))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def extract_labels(self, f, one_hot=False, num_classes=10):
        """Extract the labels into a 1D uint8 numpy array [index].
        Args:
          f: A file object that can be passed into a gzip reader.
          one_hot: Does one hot encoding for the result.
          num_classes: Number of classes for the one hot encoding.
        Returns:
          labels: a 1D unit8 numpy array.
        Raises:
          ValueError: If the bystream doesn't start with 2049.
        """
        print('Extracting', f.name)
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                                 (magic, f.name))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            if one_hot:
                return self.dense_to_one_hot(labels, num_classes)
            return labels

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    @staticmethod
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]
