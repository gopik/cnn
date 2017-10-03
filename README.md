# CNN
OCR using MNIST based CNN.

## Training
python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100

tensorboard --logdir=/tmp/cnn --port:6006

To save a model for later serving:


python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100 --save_model_dir=\<save model dir>

## Prediction
python3 mnist_cnn_predict.py --save_model_dir=\<save model dir>

## Notes
libfreetype6-dev needed for PIL imagefont true type fonts

Converting images to tfrecord format - https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html

## Things to try (to reduce overfitting):
2. Use dropout for conv layers
3. Use inverted images for training (and skip mean image subtraction) [Didn't work]
4. Use thresholded images for training.
5. Add zoomed images to training.
6. Crop images at font boundary for training and inference (using histograms).
