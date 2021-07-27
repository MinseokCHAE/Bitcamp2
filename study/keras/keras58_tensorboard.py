from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_images=True)

'''
cmd
d:
cd bitcamp2
cd study
cd _save
cd _graph
dir/w
tensorboard --logdir=.
http://localhost:6006/ -> copy & paste into Chrome
or
127.0.0.1:6006

'''