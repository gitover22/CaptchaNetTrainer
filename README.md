# A CNN training framework for captcha recognition

## introduction
This framework is designed for deep learning beginners, can be used to experience the training process of neural networks, experience tuning optimization, is very easy to deploy, and can run on your personal computer in a short time.

## overview
![voerview](./resource/overview.png)

## dataset
![dataset](./resource/datasets.png)
dataset generator can generate arbitrary datasets by Captcha package.

## test
![test params](./resource/params.png)

The pure numeric CAPTCHA, pure uppercase letter CAPTCHA and alphanuloid mixed CAPTCHA are tested, and resnet50 in the model pool is used for testing, and the accuracy is 99.7%,99.3%,96.5%

## some tips
- Supports cpu/gpu.
- pytorch framework.
- easy deploy on your python env.


## how to run
- git clone or download zip.
- Install the required packages.
- modify parameters for your model or add model.
- Use dataset generator to generate the dataset.
- Train and test the model.


## attention
- Console interaction, you can choose diyCNN,ResNet etc.
- The dataset has high complexity and contains noise, curve and other interference information.
- The project is not yet complete and there are multiple directions for optimization.
