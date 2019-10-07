# Source Seeking on a Nano drone
In this repository, we include all code necessary for inference of a full 8-bit quantized DQN policy on a BitCraze CrazyFlie. Our methodology is summarized as:

![Air Learning](readme_fig/pipeline_ss.png)

This repository consists of the following three parts:
  - Training: the settings required to train with the [Air Learning](https://github.com/harvard-edge/airlearning) platform. 
  - Conversion: the python code for conversion from a Tensorflow checkpoint (.cktp) to a piece of c++ code.
  - crazyflie-firmware: the crazyflie firmware enabling inference.
   

# The Project

**Paper**: https://arxiv.org/abs/1909.11236

**Title**: Learning to Seek: Autonomous Source Seeking with Deep Reinforcement Learning Onboard a Nano Drone Microcontroller

**Authors**: Bardienus P. Duisterhof, Srivatsan Krishnan, Jonathan J. Cruz, Colby R. Banbury, William Fu, Aleksandra Faust, Guido C. H. E. de Croon, Vijay Janapa Reddi

**Video**:

[![ss_vid](http://img.youtube.com/vi/wmVKbX7MOnU/0.jpg)](http://www.youtube.com/watch?v=wmVKbX7MOnU "Source Seeking Video")

# Install

The following instructions are tested on Ubuntu 18.04

**Training**

First of all, install the [Air Learning](https://github.com/harvard-edge/airlearning) platform according to the instructions on its github page. Verify your installation was successfull before proceeding.

Your installation will contain an [airlearning-rl](https://github.com/harvard-edge/airlearning-rl) folder, which is the folder we need to change to train a source-seeking agent. Simply do a:

```bash
cd ~/airlearning-rl
git fetch --all
git checkout source-seeking
```
This will provide you with all the algorithmic changes we made to [Air Learning](https://github.com/harvard-edge/airlearning).

**Conversion**

dependencies
```bash
pip3 install tensorflow matplotlib numpy 
```
**Crazyflie-firmware**

dependencies
```bash
sudo add-apt-repository ppa:team-gcc-arm-embedded/ppa
sudo apt-get update
sudo apt install gcc-arm-embedded
```
**Clone**
 
```bash
git clone https://github.com/harvard-edge/source-seeking --recursive
cd crazyflie-firmware 
git submodule init
git submodule update
```
# Use
**Training**

Once the source seeking branch in [airlearning-rl](https://github.com/harvard-edge/airlearning-rl) is checked out, you can proceed using airlearning in a normal fashion. Use the settings.py file to adjust the environment.

**Conversion**

Use [main.py](https://github.com/harvard-edge/source-seeking/blob/master/Conversion/main.py) in the conversion folder to call the functions needed. For the current settings, the provided model will be converted in a total of 1,000 iterations. Settings can be altered in [settings.py](https://github.com/harvard-edge/source-seeking/blob/master/Conversion/settings.py).

**CrazyFlie Firmware**

```bash
make clean
make -j4
make cload
```
**Note:** the crazyflie won't take-off without a light-sensor connected. We designed the code as such to verify sensor connectivity before commencing a run. If no sensor is present, alter [this](https://github.com/harvard-edge/crazyflie-firmware/blob/5a839d5f76ae6965b566fa2df69195ca921fb625/src/deck/drivers/src/tfmicrodemo.c) file, by commenting out all 'TS2591XXX' functions, and replace 'sensor_read' with something else (e.g., random values).

# Contact

For help, contact Bart (bduisterhof@g.harvard.edu)

**Contributors**

Bardienus Duisterhof (bduisterhof@g.harvard.edu)

Srivatsan Krishnan (srivatsan@seas.harvard.edu)

William Fu 
