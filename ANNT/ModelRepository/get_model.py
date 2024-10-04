from ..ModelRepository import * 

"""
    This module should be updated when a new model is added.
"""


def get_model(model_name, precision):
    """
    Returns the model definition of the selected model.

    Args:
    -------

    model_name: SupportedModels enum signifying the name of the model, as specified in the SupportedModels enum

    precision: Precision of the model. This is required because to get the quantized version of the model definition,
                we need to set the bit width in the graph

    Returns:
    ---------

    A modelDesc class implementation that contains 
    
        a. input/output placeholders

        b. Optimizer to be used during training

        c. Model description/ graph

    """
    #FP LeNet models
    if(model_name == SupportedModels.lenet5):
        print("Full precision LeNet5 base Model.")
        return lenet_fp.Model()
    elif(model_name == SupportedModels.lenet5b):
        print("Full precision LeNet5 variant b Model.")
        return lenet_fp_b.Model()
    elif(model_name == SupportedModels.lenet5c):
        print("Full precision LeNet5 variant c Model.")
        return lenet_fp_c.Model()
    elif(model_name == SupportedModels.lenet5tf):
        print("Full precision LeNet5 Model with tf layers.")
        return lenet_tf.Model()
    #FP Models A, B and C
    elif(model_name == SupportedModels.model_a):
        print("Full precision Model A selected.")
        model_a.model_capacity = 16 #32
        return model_a.Model()
    elif(model_name == SupportedModels.model_b):
        print("Full precision Model B selected.")
        model_b.model_capacity = 32 #64
        return model_b.Model()
    elif(model_name == SupportedModels.model_c):
        print("Full precision Model C selected.")
        model_c.model_capacity = 64 #48 #96
        return model_c.Model()
    elif(model_name == SupportedModels.cw_mnist):
        print("Full precision custom model for cw attack selected. Model is defined in the attack paper")
        return custom_cw.Model()

    #Quantized LeNet
    elif(model_name == SupportedModels.lenet_q):
        lenet_q.BITW, lenet_q.BITA, lenet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for LeNet5".format(lenet_q.BITW, lenet_q.BITA, lenet_q.BITG))
        return lenet_q.Model()
    elif(model_name == SupportedModels.lenet_qb):
        lenet_q_b.BITW, lenet_q_b.BITA, lenet_q_b.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for LeNet5 one additional FC".format(lenet_q_b.BITW, lenet_q_b.BITA, lenet_q_b.BITG))
        return lenet_q_b.Model()
    elif(model_name == SupportedModels.lenet_qc):
        lenet_q_c.BITW, lenet_q_c.BITA, lenet_q_c.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for LeNet5 two additional FC".format(lenet_q_c.BITW, lenet_q_c.BITA, lenet_q_c.BITG))
        return lenet_q_c.Model()
        
    #Quantized Models A, B and C
    elif(model_name == SupportedModels.model_aq):
        model_a_q.BITW, model_a_q.BITA, model_a_q.BITG = map(int, precision.split(','))
        model_a_q.model_capacity = 16 #32
        print("Quantization with: {}, {}, {} for MODEL A".format(model_a_q.BITW, model_a_q.BITA, model_a_q.BITG))
        return model_a_q.Model()
    elif(model_name == SupportedModels.model_bq):
        model_b_q.BITW, model_b_q.BITA, model_b_q.BITG = map(int, precision.split(','))
        model_b_q.model_capacity = 32 #64
        print("Quantization with: {}, {}, {} for MODEL B".format(model_b_q.BITW, model_b_q.BITA, model_b_q.BITG))
        return model_b_q.Model()
    elif(model_name == SupportedModels.model_cq):
        model_c_q.BITW, model_c_q.BITA, model_c_q.BITG = map(int, precision.split(','))
        model_c_q.model_capacity = 64 #48 #96
        print("Quantization with: {}, {}, {} for MODEL C".format(model_c_q.BITW, model_c_q.BITA, model_c_q.BITG))
        return model_c_q.Model()
    else:
        print("nothing to return")

def get_mod_typ(training_model, precision):

    """
        This function returns SupportedModels Enum based on the string input. The string input is usually from a yaml file.

        Args:
        -----------------
        training_model: name of the model as specified in the user input from yaml file (string)

        precision: precision of the model (string of format 2,2,32 representing bitW, bitA and bitG). Is None for full precision


        Returns:
        ------------------
        A "SupportedModels" type enum
    
    """

    #Get the model architecture
    if(training_model.lower() == "lenet5" or training_model.lower() == "lenet5_a"):
       model_arch = SupportedModels.lenet5
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.lenet_q
    elif(training_model.lower() == "lenet5_b"):
       model_arch = SupportedModels.lenet5b
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.lenet_qb
    elif(training_model.lower() == "lenet5_c"):
       model_arch = SupportedModels.lenet5c
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.lenet_qc
    elif (training_model.lower() == "lenet5tf"):
        model_arch = SupportedModels.lenet5tf
        if(precision is not None):
           model_arch = SupportedModels.lenet_q
    #MODEL A, B , C MNIST
    elif (training_model.lower() == "model_a"):
        model_arch = SupportedModels.model_a
        if(precision is not None):
           model_arch = SupportedModels.model_aq
    elif (training_model.lower() == "model_b"):
        model_arch = SupportedModels.model_b
        if(precision is not None):
           model_arch = SupportedModels.model_bq
    elif (training_model.lower() == "model_c"):
        model_arch = SupportedModels.model_c
        if(precision is not None):
           model_arch = SupportedModels.model_cq  
    elif (training_model.lower() == "cw_mnist"):
        model_arch = SupportedModels.cw_mnist 
        #TODO: quantization support for this model
        if(precision is not None):
            raise Exception("Quantization of this model type is not supported.")
    else:
        print("unimplemented")
        return
    return model_arch



def get_mod_typ_cifar(training_model, precision):

    """
        This function returns SupportedModels Enum based on the string input. 
        The string input is usually from a yaml file.
        This function is specifically for CIFAR10 models only since the MNIST models were too many and made the code conjusted

        Args:
        -----------------
        training_model: name of the model as specified in the user input from yaml file (string)

        precision: precision of the model (string of format 2,2,32 representing bitW, bitA and bitG). Is None for full precision


        Returns:
        ------------------
        A "SupportedModels" type enum
    
    """
    #Get the model architecture
    if(training_model.lower() == "cifar_a" or training_model.lower() == "cifara"):
       model_arch = SupportedModels.cifar_a
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.cifar_aq
    elif(training_model.lower() == "cifar_b" or training_model.lower() == "cifarb"):
       model_arch = SupportedModels.cifar_b
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.cifar_bq
    elif(training_model.lower() == "cifar_c" or training_model.lower() == "cifarc"):
       model_arch = SupportedModels.cifar_c
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.cifar_cq

    #Get the resnet architecture
    elif(training_model.lower() == "resnet3" or training_model.lower() == "resnet_3"):
       model_arch = SupportedModels.resnet_3
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.resnet_3q
    elif(training_model.lower() == "resnet5" or training_model.lower() == "resnet_5"):
       model_arch = SupportedModels.resnet_5
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.resnet_5q
    elif(training_model.lower() == "resnet7" or training_model.lower() == "resnet_7"):
       model_arch = SupportedModels.resnet_7
       if(precision is not None):
           #if precision is specified get he quantized version of the model
           model_arch = SupportedModels.resnet_7q
    else:
        print("unimplemented")
        return
    return model_arch

def get_model_cifar(model_name, precision):
    """
    Returns the model definition of the selected model. Again, specifically for cifar

    Args:
    -------

    model_name: SupportedModels enum signifying the name of the model, as specified in the SupportedModels enum

    precision: Precision of the model. This is required because to get the quantized version of the model definition,
                we need to set the bit width in the graph

    Returns:
    ---------

    A modelDesc class implementation that contains 
    
        a. input/output placeholders

        b. Optimizer to be used during training

        c. Model description/ graph

    """
    #FP CIFAR models
    if(model_name == SupportedModels.cifar_a):
        print("Full precision CIFAR-10 covnet base Model A.")
        return cifar_convnet.Model()
    if(model_name == SupportedModels.cifar_b):
        print("Full precision CIFAR-10 covnet base Model B.")
        return cifar_convnet_b.Model()
    if(model_name == SupportedModels.cifar_c):
        print("Full precision CIFAR-10 covnet base Model C.")
        return cifar_convnet_c.Model()

    #FP CIFAR Resnet models
    if(model_name == SupportedModels.resnet_3):
        print("Full precision ResNet model with n = 3 and 20 layers")
        return cifar10_resnet.Model(3)
    if(model_name == SupportedModels.resnet_5):
        print("Full precision ResNet model with n = 5 and 32 layers")
        return cifar10_resnet.Model(5)
    if(model_name == SupportedModels.resnet_7):
        print("Full precision ResNet model with n = 7 and 44 layers.")
        return cifar10_resnet.Model(7)


        
    #Quantized versions of CIFAR10 models
    elif(model_name == SupportedModels.cifar_aq):
        cifar_convnet_q.BITW, cifar_convnet_q.BITA, cifar_convnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for CIFAR model A.".format(cifar_convnet_q.BITW, cifar_convnet_q.BITA, cifar_convnet_q.BITG))
        return cifar_convnet_q.Model()
    elif(model_name == SupportedModels.cifar_bq):
        cifar_convnet_qb.BITW, cifar_convnet_qb.BITA, cifar_convnet_qb.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for CIFAR model B.".format(cifar_convnet_qb.BITW, cifar_convnet_qb.BITA , cifar_convnet_qb.BITG))
        return cifar_convnet_qb.Model()
    elif(model_name == SupportedModels.cifar_cq):
        cifar_convnet_qc.BITW, cifar_convnet_qc.BITA, cifar_convnet_qc.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for CIFAR model C.".format(cifar_convnet_qc.BITW, cifar_convnet_qc.BITA , cifar_convnet_qc.BITG))
        return cifar_convnet_qc.Model()

    #Quantized versions of ResNet models
    elif(model_name == SupportedModels.resnet_3q):
        cifar10_resnet_q.BITW, cifar10_resnet_q.BITA, cifar10_resnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for Resnet model with n=3.".format(cifar10_resnet_q.BITW, cifar10_resnet_q.BITA, cifar10_resnet_q.BITG))
        return cifar10_resnet_q.Model(3)
    elif(model_name == SupportedModels.resnet_5q):
        cifar10_resnet_q.BITW, cifar10_resnet_q.BITA, cifar10_resnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for Resnet model with n=5.".format(cifar10_resnet_q.BITW, cifar10_resnet_q.BITA , cifar10_resnet_q.BITG))
        return cifar10_resnet_q.Model(5)
    elif(model_name == SupportedModels.resnet_7q):
        cifar10_resnet_q.BITW, cifar10_resnet_q.BITA, cifar10_resnet_q.BITG = map(int, precision.split(','))
        print("Quantization with: {}, {}, {} for Resnet model with n=7.".format(cifar10_resnet_q.BITW, cifar10_resnet_q.BITA , cifar10_resnet_q.BITG))
        return cifar10_resnet_q.Model(7)
    
    else:
        print("nothing to return")