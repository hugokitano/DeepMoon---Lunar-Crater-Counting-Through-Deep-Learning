## TensorBase: Minimalistic TensorFlow Framework

TensorBase provides a set of Python classes that abstract the typical functions involved
in a machine learning application and facilitate the creation of new applications
from data management and to model implementation. TensorBase differs from other
TensorFlow-compatible APIs such as Keras and PrettyTensor in several ways:

  1. Has a simpler structure but demands more knowledge of TensorFlow syntax
  2. Facilitates the creation of new layers and functions in networks and training
  3. Addresses data and metrics (e.g. loss, accuracy) management

## Contents
The TensorBase packages contains 3 Classes in ```base.py``` and 1 Class in ```data.py```.

### Base:
* **Model**: a parent class that defines the general structure of TensorFlow models and manages metrics.
* **Layers**: a parent class that iteratively creates connected and convolutional networks.
* **Data**: a parent class for batch generation.

### Data:
* **MNIST**: a child class that generates batchs for the MNIST dataset.