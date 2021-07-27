from tensorflow.keras.callbacks import TensorBoard

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0, write_graph=True, write_image=True)
