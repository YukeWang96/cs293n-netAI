#!/usr/bin/env python3
import os
import sys
from random import shuffle, sample

def merge_csvs(dirpath):
    dirpath = sys.argv[1]
    files = os.listdir(dirpath)
    foutput = open("merged.csv", "w")
    content = []
    header = False
    for fname in files:
        if ".csv" in fname:
            fpth = os.path.join(dirpath, fname)
            fp = open(fpth)
            cnt = 0
            for line in fp:
                if cnt == 0:
                    if not header:
                        foutput.write(line)
                        header = True
                    cnt += 1
                else:
                    foutput.write(line)
                    cnt += 1
    foutput.close()
    print("=> Complete Merge CSVs, Output to -- merge.csv") 

def shuffle_files(fpath):
    fp = open(fpath)
    fo = open("shuffle.csv", "w")
    content = []
    for line in fp:
        content.append(line.replace(" ", ""))
    
    # print(content)
    tmp = sample(content[1:], len(content) - 1)
    result = content[:1] + tmp

    for line in result:
        fo.write(line)

    print("=> Complete Shuffle data, Output to -- shuffle.csv")

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     raise ValueError("Usage: ./exe dir_path")
    # merge_csvs(sys.argv[1])
    shuffle_files(os.path.join(os.path.dirname(__file__), "train", "merged.csv"))