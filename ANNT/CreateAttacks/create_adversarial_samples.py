from typing import Dict
from .attack_core import RestoreModel, create_classifier
from .setup_attacks import SetupAttacks
from .supported_attacks import SupportedAlgorithms
from tensorpack.utils import logger
from  ..visualize_data import *
import logging

import numpy as np
import pandas as pd

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import SaliencyMapMethod
from art.attacks.evasion import BoundaryAttack
from art.attacks.evasion import UniversalPerturbation
from art.attacks.evasion import CarliniL2Method, CarliniL0Method, CarliniLInfMethod
import art
from art.utils import load_mnist


def get_adversarial_samples(config, dataset, attack_type, targets, attack_params):    

    """
    This function will first create a model to attack using attack_model.py then will use the model to create adversarial examples.
    
    args:
    -------------
    
    config : The PredictConfig class instance which already has a session where the model which has model restoration data along with checkpoint location
    
    dataset : The dataset which should be used to create adversarial examples. Ideally a test_data set
    
    supportedAlgorithms : enum object which indicates which algorithm to use to create adversarial examples

    Returns:
    --------------

    Zipped tuples whose first element if the adversarial image and the second image is the 
    """
    #TODO: Extend this so that adversarial examples are created not only for given dataset but also for provided images

    # Restore the saved model and get all paramerters necessary to create adversarial examples
    attack_model = RestoreModel(config)
    #use the attack_params to create a classifier
    classifier = create_classifier(attack_model)
    if(attack_type == SupportedAlgorithms.FSGM):
        x_adv = CreateFGSMAttack(classifier, dataset, targets, attack_params)
        return x_adv
    elif(attack_type == SupportedAlgorithms.JSMA):
        x_adv = CreateJSMAttack(classifier, dataset, targets, attack_params)
        return x_adv
    elif(attack_type == SupportedAlgorithms.UAP):
        x_adv = CreateUAPAttack(classifier, dataset, attack_params)
        return x_adv
    elif(attack_type == SupportedAlgorithms.BA):
        x_adv = CreateSimBattack(classifier, dataset, targets, attack_params)
        return x_adv
    elif(attack_type == SupportedAlgorithms.CW_L2):
        x_adv = CreateCWattack_L2(classifier, dataset, targets, attack_params)
        return x_adv
    elif(attack_type == SupportedAlgorithms.CW_l0):
        x_adv = CreateCWattack_L0(classifier, dataset, targets, attack_params)
        return x_adv
    elif(attack_type == SupportedAlgorithms.CW_linf):
        x_adv = CreateCWattack_Linf(classifier, dataset, targets, attack_params)
        return x_adv
    else:
        return None


def CreateFGSMAttack(classifier:art.estimators.classification.TensorFlowClassifier, data:np.array, targets:list, attack_params:list):
    '''
    Craft FGSM attack. 

    Args:
    ---------   
    * classifier: a trained tf classifier    
    * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
    * attack_param:  [epsilon (attack step size): value between 0 and 1]
    '''  
    _targeted = False
    if(len(targets)>0):
        _targeted = True
    adv_crafter = FastGradientMethod(classifier, targeted= _targeted,  eps=attack_params[0])
    
    if(_targeted):
        x_test_adv = adv_crafter.generate(x=data, y= np.array(targets))
    else:
        x_test_adv = adv_crafter.generate(x=data)
        
    return x_test_adv   

def CreateJSMAttack(classifier:art.estimators.classification.TensorFlowClassifier, data:np.array, targets:list, attack_params:list):    
    '''
    Craft JSMA attack. 
    Args:
    ---------   
    * classifier: a trained tf classifier    
    * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
    * attack_param:  [theta: (distortion introduced per selected feature per iteration): value between 0 and 1, gamma (percentage of pixels to be distorted): value between 0 and 1]
    '''
    
    _targeted = False
    if(len(targets)>0):
        _targeted = True
        
    _theta = attack_params[0]
    _gamma = attack_params[1]
    adv_crafter = SaliencyMapMethod(classifier=classifier, theta=_theta, gamma=_gamma)
    logger.info("Theta: {}, Gamma: {}".format(_theta, _gamma))
    
    if(_targeted):
        x_test_adv = adv_crafter.generate(x=data, y= np.array(targets))
    else:
        x_test_adv = adv_crafter.generate(x=data)
    
    return x_test_adv 

def CreateUAPAttack(classifier, data, attack_params): 
    '''
    Craft UAP attack. 

    Args:
    ---------   
    * classifier: a trained tf classifier    
    * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
    * attack_param:  [xi: (maximum allowed perturbation): value between 0 and 1, max_iter (number of iteration to run the algorithm): positive integer, eps: (perturbation per step, specific to FGSM)]
    '''
    attacker_p = {"eps": attack_params[2]}    
    xi = attack_params[0]   
    _max_iter = attack_params[1]
    logger.info("eps: {}; xi: {}, max_iter: {}".format(attacker_p, xi, _max_iter))
    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    #handler = logging.StreamHandler()
    #formatter = logging.Formatter("[%(levelname)s] %(message)s")
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)
    adv_crafter = UniversalPerturbation(classifier, attacker="fgsm", attacker_params= attacker_p, eps=xi,max_iter=_max_iter)
    x_test_adv = adv_crafter.generate(x=data)
    noise = adv_crafter.noise
    print(noise.shape)
    logger.info("Fooling rate: {}".format(adv_crafter.fooling_rate))
    for uap in noise:
        plot_image(uap)    
    return x_test_adv 

def CreateSimBattack(classifier, data, targets:list, attack_params):    
    '''
    Craft BoundaryAttack attack. 

    Args:
    ---------   
    * classifier: a trained tf classifier    
    * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
    * attack_param: [max_iter (number of iteration to run the algorithm)]
    '''
    #(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    #print(x_train.shape)
    #print(data.shape)
    #Simba needs channel information which is not provided in mnist trainining model

    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)
    #handler = logging.StreamHandler()
    #formatter = logging.Formatter("[%(levelname)s] %(message)s")
    #handler.setFormatter(formatter)
    #logger.addHandler(handler)
    _targeted = False
    if(len(targets)>0):
        _targeted = True
    _max_itter = attack_params[0]
    logger.info("BA: Number of iterations: {}".format(_max_itter))
    adv_crafter =BoundaryAttack(estimator=classifier, max_iter=_max_itter, targeted=_targeted)
    
    if(_targeted):
        x_test_adv = adv_crafter.generate(x=data, y= np.array(targets))
    else:
        x_test_adv = adv_crafter.generate(x=data)
    
    return x_test_adv 


def CreateCWattack_L2(classifier, data, targets:list, attack_params):        
        '''
        Craft L2 version of the CW attack. 

        Args:
        ---------   
        * classifier: a trained tf classifier    
        * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
        * attack_param: [
            
            confidence_k: Confidence of adversarial examples where a higher value produces more distorted examples but classified with higher confidence as the target class,   
            
            targeted: Boolean. Set 0 for false 1 for true

            learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.

            max_iter: The maximum number of iterations.

            initial_const: The initial trade-off constant c to use to tune the relative importance of distance and confidence. If binary_search_steps is large, the initial constant is not important, as discussed in Carlini and Wagner (2016).

            batch_size: Size of the batch on which adversarial samples are generated.

            binary_search_steps: Number of times to adjust constant with binary search (positive value). If binary_search_steps is large, then the algorithm is not very sensitive to the value of initial_const. Note that the values gamma=0.999999 and c_upper=10e10 are hardcoded with the same values used by the authors of the method.

            max_halving: Maximum number of halving steps in the line search optimization.

            max_doubling: Maximum number of doubling steps in the line search optimization.
            ]
        '''
        
        _targeted = False
        if(len(targets)>0):
            _targeted = True
            
        confidence_k = attack_params[0]
        #targeted = attack_params[1]
        learning_rate = attack_params[1]       
        max_iter = attack_params[2]
        initial_const = attack_params[3]      
        batch_size = attack_params[4]
        binary_search_steps = attack_params[5]
        max_halving = attack_params[6]
        max_doubling = attack_params[7]
        adv_crafter =CarliniL2Method(classifier=classifier, 
                                                 confidence= confidence_k, 
                                                 targeted=_targeted, 
                                                 learning_rate=learning_rate,
                                                 binary_search_steps=binary_search_steps, 
                                                 max_iter=max_iter, 
                                                 initial_const=initial_const,
                                                 max_halving=max_halving,
                                                 max_doubling=max_doubling,
                                                 batch_size=batch_size)  

        
        if(_targeted):
            x_test_adv = adv_crafter.generate(x=data, y= np.array(targets))
        else:
            x_test_adv = adv_crafter.generate(x=data)        
        
        return x_test_adv

def CreateCWattack_Linf(classifier, data, targets:list, attack_params):        
        '''
        Craft L_infinity version of the CW attack. 

        Args:
        ---------   
        * classifier: a trained tf classifier    
        * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
        * attack_param: [
            
            confidence_k: Confidence of adversarial examples where a higher value produces more distorted examples but classified with higher confidence as the target class,   
            
            targeted: Boolean. Set 0 for false 1 for true

            learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.

            max_iter: The maximum number of iterations.

            initial_const: The initial value of constant c.

            batch_size: Size of the batch on which adversarial samples are generated.

            decrease_factor: The rate of shrinking tau, values in 0 < decrease_factor < 1 where larger is more accurate.

            largest_const: The largest value of constant c.

            const_factor: The rate of increasing constant c with const_factor > 1, where smaller more accurate.
            ]
        '''
        
        _targeted = False
        if(len(targets)>0):
            _targeted = True
            
        confidence_k = attack_params[0]
        #targeted = attack_params[1]
        learning_rate = attack_params[1]       
        max_iter = attack_params[2]
        initial_const = attack_params[3]      
        batch_size = attack_params[4]        
        decrease_factor = attack_params[5]
        largest_const = attack_params[6]
        const_factor = attack_params[7]
        adv_crafter =CarliniLInfMethod(classifier=classifier, 
                                                 confidence= confidence_k, 
                                                 targeted=_targeted, 
                                                 learning_rate=learning_rate,
                                                 max_iter=max_iter,                                                  
                                                 initial_const=initial_const,
                                                 decrease_factor=decrease_factor,
                                                 largest_const=largest_const,
                                                 const_factor=const_factor,
                                                 batch_size=batch_size)

        if(_targeted):
            x_test_adv = adv_crafter.generate(x=data, y= np.array(targets))
        else:
            x_test_adv = adv_crafter.generate(x=data)        
        
        return x_test_adv

def CreateCWattack_L0(classifier, data, targets:list, attack_params):        
        '''
        Craft L_0 version of the CW attack. 

        Args:
        ---------   
        * classifier: a trained tf classifier    
        * data: imageset (numpy array) which is used to craft adversarial examples. Usually images from test dataset are used.
        * attack_param: [
            
            confidence_k: Confidence of adversarial examples where a higher value produces more distorted examples but classified with higher confidence as the target class,   
            
            targeted: Boolean. Set 0 for false 1 for true

            learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results but are slower to converge.

            max_iter: The maximum number of iterations.

            initial_const: The initial value of constant c.

            batch_size: Size of the batch on which adversarial samples are generated.

            max_halving: Maximum number of halving steps in the line search optimization.

            max_doubling: Maximum number of doubling steps in the line search optimization.

            warm_start: Instead of starting gradient descent in each iteration from the initial image. we start the gradient descent from the solution found on the previous iteration.
            
            mask: The initial features that can be modified by the algorithm. If not specified, the algorithm uses the full feature set.

            binary_search_step: Number of times to adjust constant with binary search (positive value). If binary_search_steps is large, then the algorithm is not very sensitive to the value of initial_const. Note that the values gamma=0.999999 and c_upper=10e10 are hardcoded with the same values used by the authors of the method.
            ]
        '''
        _targeted = False
        if(len(targets)>0):
            _targeted = True
            
        confidence_k = attack_params[0]
        #targeted = attack_params[1]
        learning_rate = attack_params[1]       
        max_iter = attack_params[2]
        initial_const = attack_params[3]      
        batch_size = attack_params[4]
        max_halving = attack_params[5]
        max_doubling = attack_params[6]
        warm_start = attack_params[7]
        mask = attack_params[8]
        binary_search_steps = attack_params[9]
        adv_crafter =CarliniL0Method(classifier=classifier, 
                                                 confidence= confidence_k, 
                                                 targeted=_targeted, 
                                                 learning_rate=learning_rate,
                                                 max_iter=max_iter, 
                                                 initial_const=initial_const,
                                                 max_halving=max_halving,
                                                 max_doubling=max_doubling,
                                                 batch_size=batch_size,
                                                 warm_start=warm_start,
                                                 mask=mask,
                                                 binary_search_steps=binary_search_steps)

        if(_targeted):
            x_test_adv = adv_crafter.generate(x=data, y= np.array(targets))
        else:
            x_test_adv = adv_crafter.generate(x=data)        
        
        return x_test_adv

def create_benign_adv_map(x_test_adv, x_test):
    """
        Create a numpy array whose each element if a tuple containing original image and its adversarial counterpart

        Args:
        ------------------------
        x_test: original/ benign image
        x_test_adv: corresponding adversarial image

    """
    print("shape of adversarial image created {}, shape of the dataset {}:".format(x_test_adv.shape, x_test.shape))
    mapped = zip(x_test, x_test_adv)
    return mapped