import tensorflow as tf

from tensorpack import *

'''
This model architecture is as defined in the CW attack paper: "Towards Evaluating the Robustness of Neural Networks"
https://arxiv.org/abs/1608.04644

The model and code is availabe at: https://github.com/carlini/nn_robust_attacks/blob/master/train_models.py

'''

class Model(ModelDesc):

    '''
    Define model using the ModelDesc parent. To train or to inference a network, you would require 
    1. The dataset (to train or infer from)
    2. The model definition
    3. The input placeholder which is used to later feed data
    4. Optimizer function (required by trainer only)

    Three of these four (2,3,4) are supplied by the ModelDesc. Thus this class the core of tensorpack understanding and a
    model should be restored (especially quantized) using the ModelDesc description in order to make sure the graph is made correctly  
    
    ref: 
        1. https://tensorpack.readthedocs.io/en/latest/tutorial/inference.html#step-1-build-the-model-graph
        2. https://tensorpack.readthedocs.io/tutorial/training-interface.html#with-modeldesc-and-trainconfig
    '''


    # Provide the input signature
    # By default the image size is 28x28 for MNIST, so need to resize images before feeding to the network
    def inputs(self):
                #This is TF 1.13 code so we need to specify a placeholder for the graph and feed input to it later
        return [tf.TensorSpec((None, 28, 28), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    #define the model
    def build_graph(self, image, label):
        """
            The default dataset for MNIST only has 3 dim (Batch, image_height, Image_width). In tf, one addition dimension
            for channel is required so add one additional channel at axis =3

            Here also notics that input has same dimension as in inputs() TensorSpec meaning that data or input fed is in batches
            i.e input has dim: (128, 28, 28), 128 being the batch size and thus the accuracy computed laster is of 128 data points.
            tensorflow accepts input of (BHWC); Batch, height, width and channel 
        """
        image = tf.expand_dims(image, 3)

        #normalize image
        #image = image/255.0

        # Define the architecture.
        print("is training ", self.training) 
        # conv2d: The default stride is (1,1) for tf.layers so not changing those
        # conv2d: The default padding is "same", default activation is "none" 
        # pooling: The default padding is "same", default stride is equal to the pool size
        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu):
            # LinearWrap is just a syntax sugar.
            # See tutorial at https://tensorpack.readthedocs.io/tutorial/symbolic.html
            logits = (LinearWrap(image)
                      .Conv2D('conv0', filters=32)
                      .Conv2D('conv1', filters=32)
                      .MaxPooling('pool0', 2)
                      .Conv2D('conv2', filters=64)
                      .Conv2D('conv3', filters=64)
                      .MaxPooling('pool1', 2)
                      .FullyConnected('fc0', 200, activation=tf.nn.relu)
                      .FullyConnected('fc1', 200, activation=tf.nn.relu)
                      .Dropout(rate=0.5 if self.training else 0.0)
                      .FullyConnected('linear', 10)())
        tf.nn.softmax(logits, name="output")
        tf.add_to_collection("logits", logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='cross_entropy_loss')  # the average cross-entropy loss

        wd_cost = tf.multiply(1e-6,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, loss], name='total_cost')

        summary.add_moving_summary(loss)
        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')
        print(correct)

        train_error = tf.reduce_mean(1 - correct, name='train_error')

        summary.add_moving_summary(train_error, accuracy)
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))

        # the function should return the total cost to be optimized
        return total_cost

    #define the optimizer
    def optimizer(self):
        # original: sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)   
        #sgd_optimizer = optimizer.apply_grad_processors(opt, [optimizer.VariableUpdateProcessor(decay=1e-6)])
        return opt   
