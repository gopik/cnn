# CNN
OCR using MNIST based CNN.

## Training
python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100

tensorboard --logdir=/tmp/cnn --port:6006

To save a model for later serving:


python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100 --save_model_dir=\<save model dir>

inv_* models are trained using inverted images (white font on black background)

## Prediction
python3 mnist_cnn_predict.py --save_model_dir=\<save model dir>

## Notes
libfreetype6-dev needed for PIL imagefont true type fonts

Converting images to tfrecord format - https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html

## Things to try (to reduce overfitting):
 -[X] Use dropout for conv layers
 -[X] Use inverted images for training (and skip mean image subtraction)
 -[X] Use thresholded images for training (not required, since we do blur during training).
 -[X] Add zoomed images to training [Using keras image augmentation]
 -[ ] Crop images at font boundary for training and inference (using histograms).
