# usage examples: run "python contrast_acc.py log1.txt log2.txt log3.txt ..." to contrast two or more log.

import sys
from matplotlib import pyplot as plt
def parse_log(infile):
    f = open(infile,'r')
    train_iteration=[]
    train_loss=[]
    test_iteration=[]
    test_loss=[]
    test_accuracy=[]
    for line in f.readlines():
        if 'Iteration' in line and 'loss' in line:
            split_line = line.strip().split()
            train_iteration.append(int(split_line[split_line.index('Iteration') + 1][:-1]))
            train_loss.append(float(split_line[split_line.index('loss') + 2]))
            continue
        if 'Test net output' in line and 'accuracy' in line:
            split_line = line.strip().split()
            test_accuracy.append(float(split_line[split_line.index('accuracy') + 2]))
            continue
        if 'Iteration' in line and 'Testing net' in line:
            split_line = line.strip().split()
            test_iteration.append(int(split_line[split_line.index('Iteration') + 1][:-1]))
            continue
        if 'Test net output' in line and 'loss' in line:
            split_line = line.strip().split()
            test_loss.append(float(split_line[split_line.index('loss') + 2]))
            continue
    if len(test_iteration)!=len(test_accuracy):
        test_iteration.pop()
    if len(test_loss)!=len(test_accuracy):
        test_iteration.pop()
        test_accuracy.pop()
    return [train_iteration, train_loss, test_iteration, test_loss, test_accuracy]

def draw_acc(infiles):
    parse_res = [parse_log(file_) for file_ in infiles]
    fig = plt.figure(figsize=[25, 10])
    fig1 = fig.add_subplot(121)
    for i,color in zip(xrange(len(parse_res)),'rgbk'):
      fig1 = plt.plot(parse_res[i][2],parse_res[i][3],c = color,label = "Test loss "+str(i+1))
    plt.legend()
    fig2 = fig.add_subplot(122) 
    for i,color in zip(xrange(len(parse_res)),'rgbk'):  
      fig2 = plt.plot(parse_res[i][2],parse_res[i][4],c = color, label = "Test accuracy "+str(i+1))
    plt.legend(loc = 4)
    plt.show()
    
if __name__ == '__main__':
  draw_acc(sys.argv[1:])
