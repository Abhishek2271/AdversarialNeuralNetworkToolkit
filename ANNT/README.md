## Adversarial Neural Network Toolkit

The source code in the repository is a Python API that is able to fulfil a complete workflow of:
1. Traning full-precision and quantized neural networks.
2. Creating adversarial examples on these networks.
3. Transfering adversarial examples from the network where samples are created (source) to another (target) network.

The API uses Tensropack [(Y. Wu et al., 2016)](https://github.com/tensorpack) for training. Tensorpack is training framework which a part of TensorFlow 1.13 [(Abadi et al., 2016)](https://www.tensorflow.org) API.  
For quantization DoReFa-Net method [(Zhou et al., 2018)](https://arxiv.org/abs/1606.06160) is used
For adversarial attack generation ART [(Nicolae et al., 2019)](https://arxiv.org/abs/1807.01069 ) is used. 

The API is fairly simple to use. Details on how to use the api is in the [wiki](https://github.com/Abhishek2271/TransferabilityAnalysis/wiki) section.

# References
## Quantization is based on DoReFa-Net method as proposed in the paper: https://arxiv.org/abs/1606.06160  
Cited as:  
Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., & Zou, Y. (2018). DoReFa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv:1606.06160 [cs].

#### For quantization, the library available from the authors is used. This is available at:
https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net

The networks used are from the examples provided on the Tensorpack repository:   
Tensorpack cited as:  
Wu, Y. et al. (2016). Tensorpack. https://github.com/tensorpack.

#### Tensorpack models are available at:  
https://github.com/tensorpack/tensorpack/tree/master/examples

## Adversarial Examples are created using Adversarial Robustness Toolbox (ART) v. 1.5.1. Official paper: https://arxiv.org/abs/1807.01069  
Cited as:  
Nicolae, M.-I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., Zantedeschi, V., Baracaldo, N., Chen, B., Ludwig, H., Molloy, I. M., & Edwards, B. (2019). Adversarial robustness toolbox v1.0.0. arXiv:1807.01069

ART is one of the popular APIs for adversaral examples generation and supports a large number of attacks. It is open-source with large number of very well explained examples. Please check out their repository at:
https://github.com/Trusted-AI/adversarial-robustness-toolbox
