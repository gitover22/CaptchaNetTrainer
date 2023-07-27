# 验证码识别项目

#### 介绍
简单的控制台交互程序，可以使用自己搭建的cnn网络，或vgg网络进行训练和测试
识别对象：由captcha生成的验证码，大小写字母以及数字，5位长度，识别率达到98以上，下图是数据集示例

![image](https://github.com/gitover22/captcha_recognition/assets/83172922/8eb749cc-ec1c-47b2-b100-9d7ae022a033)

#### 软件架构
  使用pytorch框架,可使用gpu加速。数据集是captcha提供生成


#### 安装教程
1.  clone项目；
2.  安装所需软件包，如torch，tqdm等；
3.  训练，测试模型。


#### 使用说明
1.  控制台交互，可以选择vgg11，13，16，19等；
2.  数据集复杂程度高，含有噪点，曲线等干扰信息；
3.  项目尚不完善，有多个优化方向。
