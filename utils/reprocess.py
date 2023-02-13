import torch
import numpy as np
import torch.nn.functional as F

num_classes = 4

classes = [
 'FM',
 '8ASK',
 'AM-DSB-SC',
 '8PSK'
]

def output2class(output):
    print(classes[output.argmax(dim=-1)])

def output2odd(output):
    odds = F.softmax(output, dim=-1).cpu().detach().numpy()
    for i in range(num_classes):
        print("{} \t {:.2f}%".format(classes[i], odds[0,i]*100))
