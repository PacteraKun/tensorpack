import os, sys, argparse

parser = argparse.ArgumentParser()
input = parser.add_argument('--input', help='input log file path', type=str)
output = parser.add_argument('--output', help='output txt file to store mAp results', type=str)

args = parser.parse_args()
with open(args.input) as f:
    log = f.readlines()
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
'''for i, line in enumerate(log):
    if 'Accumulating evaluation results' in line:
        count += 1
        print(i)
print(count)'''

