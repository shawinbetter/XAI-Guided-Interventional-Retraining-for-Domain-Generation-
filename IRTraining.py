import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from transform import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger

if __name__ == '__main__':
    log_name = 'IRT.txt'

    def process(img_array): 
        # print("IMG_ARRAY SHAPE:",img_array.shape)
        x = transform(img_array,delta1=config.delta1,delta2=config.delta2)
        return x

    # Define generator
    IR_IDG = ImageDataGenerator()

    train_generator = IR_IDG.flow_from_directory(config.train,
                                            target_size=config.input_shape,
                                            batch_size=config.batch_size)


    #define new model
    class TransformLayer(tf.keras.layers.Layer):
        def __init__(self):
            super(TransformLayer, self).__init__()
            self.function = process

        def call(self, inputs):
            return tf.map_fn(fn=self.function, elems=inputs)

    def weight_initializer(shape, dtype=None):
        return config.weight
    
    def bias_initializer(shape, dtype=None):
        return config.bias

    IR_model = keras.Sequential(
        [   
            TransformLayer(),
            layers.Dense(1000,kernel_initializer = weight_initializer,bias_initializer=bias_initializer,activation='softmax')
        ]
    )
    IR_model.compile(loss='categorical_crossentropy',metrics=['accuracy','top_k_categorical_accuracy'],optimizer=Adam(1e-4))
    
    ### Training Setup
    csv_logger = CSVLogger('logs/ir.log')

    history = IR_model.fit(x = train_generator, 
            validation_data = None,
            epochs=config.epochs, verbose=1,max_queue_size=config.max_queue_size,
            workers=config.workers,
            callbacks=[csv_logger])
    
    ########Save the final Model#############
    IR_model.save('model/IrModel.h5')



