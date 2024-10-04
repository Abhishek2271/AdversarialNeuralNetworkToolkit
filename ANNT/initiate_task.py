from .train_net import *
from .Inference import *
from .CreateAttacks  import *

def check_input_integrity(task_type, model_name, dataset, saved_model_location, attack_algorithm, attack_parameters):
    if None in (task_type, model_name, dataset, saved_model_location,  attack_algorithm, attack_parameters):
        raise Exception("Please provide all required inputs")
    elif (type(attack_parameters) != list):
        raise Exception("Attack parameters should always be list")
    elif all(not isinstance(var, str) for var in [task_type, model_name, dataset, saved_model_location,  attack_algorithm]):
        raise TypeError("Task type, model names, datasets, saved model locations, and attack algorithms should be strings")


def set_user_options(task_type:str=None, 
                     model_name:str=None, 
                     dataset:str=None, 
                     images:str=None,
                     saved_model_location:str=None, 
                     attack_algorithm:str=None, 
                     attack_parameters:list=None, 
                     epochs:int=30,
                     targets:list=None,
                     randomize:bool=None,
                     sample_size:int=None,
                     prefiltered:bool=None,
                     iterations:int=None,
                     bitwidth=None) -> dict:    
    """
    Sets selected options to a dictionary that will be used during operation

    args:
    -----------
    * task_type: could be training, attack, inference
    * model_name: predefined name of the model for instance  "resnet_3"
    * dataset: name of the dataset for instance "mnist" 
    * saved_model_location: location of a saved model
    * attack_algorithm: desired attack_algorithm, could be fgsm, ba, uap, jsma
    * attack_parameters = list of parameters required to create the attack. for example, for fgsm: [0.1]
    """
    _task_type = task_type #training, attack, inference
    _model_name = model_name  #name of the model
    _dataset = dataset #name of the dataset
    _saved_model_location = saved_model_location
    _attack_algorithm = attack_algorithm #attack_algorithm
    _attack_parameters =  attack_parameters #attack_parameters

    trans_attack_images = "" #location of images to attack a network during a transfer based attack

    settings = {'task': {'type': task_type}, 
    'training-options': {'model': model_name , 'dataset': dataset , 'epochs': epochs, 'precision': {'bitwidth': None}}, 
    'inference-options': {'load-model': saved_model_location, 'base-model': model_name, 'dataset':dataset ,'images': images, 'precision': {'bitwidth': bitwidth}}, 
    'attack-options': {'attack-mode': 'create', 'create-attack': {'load-model': saved_model_location, 'base-model': model_name, 'algorithm': attack_algorithm, 'randomize': randomize, 'sample-limit': sample_size, 'iterations': iterations, 'prefiltered': prefiltered , 'parameters': attack_parameters, 'targets': targets,'dataset': dataset, 'precision': {'bitwidth': bitwidth}}, 
    'transfer-attack': {'source': {'source-images': trans_attack_images, 'source-data': dataset}, 'target': {'target-model': saved_model_location, 'target-base-model': model_name, 'precision': {'bitwidth': bitwidth}}}}}
    
    return settings

def set_dataset_dir(dataset, dataset_dir=None):
    if(dataset_dir is None):
        if(dataset.lower() == "cifar10"):
            dataset_dir = "./CIFAR10Data"
        elif (dataset.lower() == "mnist"):
            dataset_dir == "./MnistData"

def begin(task_type:str, model_name:str, dataset:str, saved_model_location:str, attack_algorithm:str, attack_parameters:list) -> None:
    """
    Based on task type, begin selected operation

    args:
    -----------
    * task_type: could be training, attack, inference
    * model_name: predefined name of the model for instance  "resnet_3"
    * dataset: name of the dataset for instance "mnist" 
    * saved_model_location: location of a saved model
    * attack_algorithm: desired attack_algorithm, could be fgsm, ba, uap, jsma
    * attack_parameters = list of parameters required to create the attack. for example, for fgsm: [0.1]
    """

    check_input_integrity(task_type, model_name, dataset, saved_model_location, attack_algorithm, attack_parameters)
    
    trans_attack_images = "" #location of images to attack a network during a transfer based attack

    settings = {'task': {'type': task_type}, 
    'training-options': {'model': model_name , 'dataset': dataset , 'precision': {'bitwidth': None}}, 
    'inference-options': {'load-model': saved_model_location, 'base-model': model_name, 'images': dataset, 'precision': {'bitwidth': None}}, 
    'attack-options': {'attack-mode': 'create', 'create-attack': {'load-model': saved_model_location, 'base-model': model_name, 'algorithm': attack_algorithm, 'parameters': attack_parameters, 'dataset': dataset, 'precision': {'bitwidth': None}}, 
    'transfer-attack': {'source': {'source-images': trans_attack_images, 'source-data': dataset}, 'target': {'target-model': saved_model_location, 'target-base-model': model_name, 'precision': {'bitwidth': None}}}}}
    
    if(settings["task"]["type"].lower() == "training"):
        print("Initiating training on the selected network...")
        initiate_traning(settings)
    elif(settings["task"]["type"].lower() == "inference"):
        print("Performing inference on the selected model...")
        inference_net.initiate_inference(settings)
    elif(settings["task"]["type"].lower()== "attack"):
        print("Creating adversarial examples based on the selected algorithm...")
        CreateAttacks.attack_net.initiate_attack_creation(settings)

def begin_train(model_name:str, dataset:str, epochs:int=50) -> None:
    """
    Begin training process of selected model and dataset

    Args:
    -----------
    * model_name: predefined name of the model for instance  "resnet_3"
    * dataset: name of the dataset for instance "mnist" 
    """
    settings =  set_user_options(task_type="training", model_name=model_name, dataset=dataset, epochs=epochs)
    print("Initiating training on the selected network...")
    initiate_traning(settings)
    
def begin_inference(saved_model_location:str, model_name:str,  dataset:str, infer_images=None, bitwidth=None):
    """
    Begin inference on a selected model

    args:
    -----------
    * model_name: predefined name of the model for instance  "resnet_3"
    * dataset: name of the dataset for instance "mnist" 
    * saved_model_location: location of a saved model
    """
    settings =  set_user_options(task_type="inference", saved_model_location=saved_model_location, model_name=model_name, dataset=dataset, images=infer_images, bitwidth=bitwidth)
    print("Performing inference on the selected model...")
    accuracy_t1, error_t1, saved_corr_images, saved_false_images = inference_net.initiate_inference(settings)
    return accuracy_t1, error_t1, saved_corr_images, saved_false_images


def begin_attack(saved_model_location:str, 
                 model_name:str,  
                 attack_algorithm:str, 
                 attack_parameters:list, 
                 dataset:str, 
                 targets:list=[], 
                 randomize:bool=False, 
                 sample_size:int=500, 
                 prefiltered:bool=False, 
                 iterations:int=1, 
                 bitwidth:str=None):
    """
    Begin attack creation on selected model

    args:
    -----------
    * model_name: predefined name of the model for instance  "resnet_3"
    * dataset: name of the dataset for instance "mnist" 
    * saved_model_location: location of a saved model
    * attack_algorithm: desired attack_algorithm, could be fgsm, ba, uap, jsma
    * targets: a list of target labels. Empty means untargetted attacks, 
    * randomize: a bool, when true will select images randomly from the input dataset when creating adversarial samples, 
    * sample_size: number of adversarial samples to create, 
    * prefiltered: boolean, when true, will only create adversarial samples for those images in the dataset that are correctly predicted by the network, 
    * iterations: number of sets of adversarial samples (sample number) to create,
    * attack_parameters = list of parameters required to create the attack. for example, for fgsm: [0.1]

    returns:
    ------------
    * adv_samples: numpy array of adversarial samples [n, image dim]
    """

    settings =  set_user_options(task_type="attack", 
                                 saved_model_location=saved_model_location, 
                                 model_name=model_name, 
                                 attack_algorithm=attack_algorithm, 
                                 attack_parameters=attack_parameters, 
                                 dataset=dataset, 
                                 targets=targets,
                                 randomize=randomize,
                                 sample_size=sample_size,
                                 prefiltered=prefiltered,
                                 iterations=iterations,
                                 bitwidth=bitwidth)
    print("Creating adversarial examples based on the selected algorithm...")
    adv_samples= CreateAttacks.attack_net.initiate_attack_creation(settings)
    return adv_samples