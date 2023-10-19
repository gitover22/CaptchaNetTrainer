# Captchas identify items

## Introduction
Simple console interactive program, you can use your own built cnn network, or vgg network for training and testing.
Identification object: captcha generated verification code, upper and lower case letters and numbers, 5 digits in length, 
the recognition rate reached more than 98, the following figure is the dataset example.

![image](https://github.com/gitover22/captcha_recognition/assets/83172922/8eb749cc-ec1c-47b2-b100-9d7ae022a033)

## something you need know
This project uses pytorch framework with gpu acceleration. The dataset is captcha provided to generate.


## how to run
- Clone this pro.
- Install the required packages.
- Train and test the model.


## a few words
- Console interaction, you can choose vgg11, 13,16,19, etc.
- The dataset has high complexity and contains noise, curve and other interference information.
- The project is not yet complete and there are multiple directions for optimization.
