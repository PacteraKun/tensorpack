import os, sys, argparse

filePath = sys.argv[1]

with open(filePath) as f:
    log = f.readlines()
count = 0
i = 0

file = open('evaluation_coco_mAP_same_threshold_cascade.txt', 'w')
while i < len(log):
    if '[EvalCallback] Will evaluate at epoch' in log[i]:
        print(log[i])
        file.write(log[i] + '\n')
    if 'Accumulating evaluation results' in log[i]:
        for j in range(i, i+14):
            print(log[j])
            file.write(log[j] + '\n')        
    i += 1
'''for i, line in enumerate(log):
    if 'Accumulating evaluation results' in line:
        count += 1
        print(i)
print(count)'''

