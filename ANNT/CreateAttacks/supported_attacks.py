from enum import Enum


#List all supported algorithms
class SupportedAlgorithms(Enum):
    FSGM = "fgsm"
    DLV = "dlv" 
    UAP = "uap"
    JSMA = "jsma" 
    BA = "ba"
    CW_l0 = "cw_l0"
    CW_linf = "cw_linf"
    CW_L2 = "cw_l2"



