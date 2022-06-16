import tensorflow as tf

class 모델:
    def __init__(self, height=480, width=240, batch_size=3, max_disp=192):
        
        self.reg = 1e-4
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_disp = max_disp #multiple of 32
        self.lr = 0.001
        self.이니셔 = tf.random_normal_initializer(0., 0.02)

    def 다운샘플링(self, filter, kernel):
        네트워크 = tf.keras.Sequential()
        네트워크.add(tf.keras.layers.Conv2D(filter, kernel, strides=2, padding='same',
                                            kernel_initializer=self.이니셔, use_bias=False))
        네트워크.add(tf.keras.layers.BatchNormalization())
        네트워크.add(tf.keras.layers.ReLU())

        return 네트워크

    def 업샘플링(self, filter, kernel, 드랍레이어 = False):
        네트워크 = tf.keras.Sequential()
        네트워크.add(tf.keras.layers.Conv2DTranspose(filter, kernel, strides=2, padding='same',
                                            kernel_initializer=self.이니셔, use_bias=False))
        네트워크.add(tf.keras.layers.BatchNormalization())
        if 드랍레이어:
            네트워크.add(tf.keras.layers.Dropout(0.5))
        네트워크.add(tf.keras.layers.ReLU())

        return 네트워크
