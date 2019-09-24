'''
September 24 -- Bart Duisterhof
Harvard Edge Computing
Set of functions for converison of checkpoint to tflite 
Used for inference of Deep Reinforcement Learning on a nano drone

Contact: bduisterhof@g.harvard.edu

evaluate.py: running the quantized model with representative input, showing the output distribution
'''

from main import *

command = np.array([])
num_it = 10000            # number of (8-bit) inputs are run through the quantized net
num_zero = 0
for _ in range(num_it):
    loc = np.argmax(inference_tflite())
    command = np.append(command,loc)
    if(loc==0):
        num_zero += 1
print('Going straight in :',(num_zero/num_it)*100,'%')

fig = plt.figure()
weights = np.ones_like(command)/float(len(command))
plt.xticks(np.arange(3),('0','1','2'))
plt.xlabel('Action ID')
plt.ylabel('Action occurence')
plt.hist(command,bins=range(4), weights=weights, cumulative=False, alpha=0.75,histtype='bar',width=0.8)
plt.show()