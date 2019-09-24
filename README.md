# Source Seeking on a Nano drone
This repository contains all code necessary for inference of a full 8-bit quantized DQN policy on a BitCraze CrazyFlie.
The repository consists of two main parts:
  - Inference: the CrazyFlie firmware
  - Conversion: the python code for conversion from a Tensorflow checkpoint (.cktp)

# The Project
Abstract:

Fully autonomous navigation using nano drones has numerous application in the real world, ranging from search and rescue to source seeking. Nano drones are well-suited for source seeking because of their agility, low price, and ubiquitous character. Unfortunately, their constrained form factor limits flight time, sensor payload, and compute capability. These challenges are a crucial limitation for the use of source-seeking nano drones in GPS-denied and highly cluttered environments.
Hereby, we introduce a fully autonomous deep reinforcement learning-based light-seeking nano drone. The 33-gram nano drone performs all computation on-board the ultra-low-power microcontroller (MCU). We present the method for efficiently training, converting, and utilizing deep reinforcement learning policies. Our training methodology and novel quantization scheme allow fitting the trained policy in 3 kB of memory. The quantization scheme uses representative input data and input scaling to arrive at a full 8-bit model. Finally, we evaluate the approach in simulation and flight tests using a Bitcraze CrazyFlie, achieving 80\% success rate on average in a highly cluttered and randomized test environment. Even more, the drone finds the light source in 29\% fewer steps compared to a baseline simulation (obstacle avoidance without source information). To our knowledge, this is the first deep reinforcement learning method that enables source seeking within a highly constrained nano drone demonstrating robust flight behavior. Our general methodology is suitable for any (source seeking) highly constrained platform using deep reinforcement learning.

Video:

[![ss_vid](http://img.youtube.com/vi/wmVKbX7MOnU/0.jpg)](http://www.youtube.com/watch?v=wmVKbX7MOnU "Source Seeking Video")



# Install

Install dependencies
```bash
pip3 install tensorflow matplotlib numpy 
```
Clone repo
```bash
git clone https://github.com/harvard-edge/source-seeking
```
