# CNN
## Training
python3 mnist_cnn_scratch.py --num_training_steps=1000 --checkpoint_every=100 --checkpoint_dir=/tmp/cnn/checkpoints --batch_size=100

tensorboard --logdir=/tmp/cnn --port:6006

