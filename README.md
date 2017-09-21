# CNN
## Training
python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100

tensorboard --logdir=/tmp/cnn --port:6006

To save a model for later serving:


python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100 --save_model_dir=\<save model dir>

## Prediction
python3 mnist_cnn_predict.py --save_model_dir=\<save model dir>

## Notes
libfreetype6-dev needed for PIL imagefont true type fonts