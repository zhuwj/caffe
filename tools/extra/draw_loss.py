#__author__ = 'hujie'
# usage example: run "python draw_loss.py log_0-15000.txt 7000 log_7000-15000.txt 10000 log_10000-15000.txt "
# if you have only one log to plot, just run "python draw_loss.py log.txt"

import sys
import time
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
        if 'Test net output' in line and 'loss3/top-1' in line:
            split_line = line.strip().split()
            test_accuracy.append(float(split_line[split_line.index('loss3/top-1') + 2]))
            continue
        if 'Iteration' in line and 'Testing net' in line:
            split_line = line.strip().split()
            test_iteration.append(int(split_line[split_line.index('Iteration') + 1][:-1]))
            continue
        if 'Test net output' in line and 'loss3/loss3' in line:
            split_line = line.strip().split()
            test_loss.append(float(split_line[split_line.index('loss3/loss3') + 2]))
            continue
    if len(test_iteration)!=len(test_accuracy):
        test_iteration.pop()
    if len(test_loss)!=len(test_accuracy):
        test_iteration.pop()
        test_accuracy.pop()
    return [train_iteration, train_loss, test_iteration, test_loss, test_accuracy]


def draw_loss(arg):
  train_iteration, train_loss, test_iteration, test_loss, test_accuracy = arg
  fig = plt.figure(figsize=[25, 10])
  fig1 = fig.add_subplot(121)
  plt.plot(train_iteration,train_loss,'b',label = 'train')
  plt.plot(test_iteration,test_loss,'r',label = 'test')
  plt.title('Train and Test loss')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.legend()

  fig2 = fig.add_subplot(122)
  plt.plot(test_iteration,test_accuracy,'r')
  plt.title('Test accuracy')
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy')
  timestr = time.strftime("%d_%m_%y_%H:%M")
  plt.savefig('losss_accu_' + timestr + '.png')
  plt.show()

def draw_loss_from_file(infile):
  arg = parse_log(infile)
  draw_loss(arg)


def concat_loss(file_list, iter_list):
    concat_result = [[],[],[],[],[]]
    index_test_flag = 1
    if len(file_list)!=len(iter_list):
        iter_list.append(parse_log(file_list[-1])[0][-1])
    for i,file in enumerate(file_list):
        result = parse_log(file)
        index_train = result[0].index(iter_list[i])
        if iter_list[i] not in result[2]:
          index_test_flag = 0
        else:
          index_test = result[2].index(iter_list[i])
        result[0] = result[0][:index_train]
        result[1] = result[1][:index_train]
        if(index_test_flag == 1):
          result[2] = result[2][:-1]
          result[3] = result[3][:-1]
          result[4] = result[4][:-1]
        for j in xrange(5):
            concat_result[j].extend(result[j])
    return concat_result;



if __name__ == '__main__':
    file_list=[]
    iter_list=[]
    for i,arg in enumerate(sys.argv):
        if i==0: continue
        if i%2 == 0:
            iter_list.append(int(arg))
        else:
            file_list.append(arg)
    result = concat_loss(file_list,iter_list)
    draw_loss(result)

