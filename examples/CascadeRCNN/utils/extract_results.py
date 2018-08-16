'''

Print and store mAP results of validatioin dataset from train log text file. 

'''

import os, sys, argparse

parser = argparse.ArgumentParser()
input = parser.add_argument('--input', help='input log file path', type=str)
output = parser.add_argument('--output', help='output txt file to store mAp results', type=str)

args = parser.parse_args()
if os.path.exists(args.input):
    with open(args.input) as f:
        try:
            log = f.readlines()
        except IOError:
            print("Could not read file:", args.input)

count = 0
i = 0

file = open(args.output, 'w')
while i < len(log):
    if '[EvalCallback] Will evaluate at epoch' in log[i]:
        print(log[i])
        file.write(log[i] + '\n')
    if 'Accumulating evaluation results' in log[i]:
        for j in range(i, i+14):
            print(log[j])
            file.write(log[j] + '\n')        
    i += 1
file.close()

