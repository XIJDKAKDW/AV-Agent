import argparse
import platform
import sys

system = platform.system()

if system == "Linux":
    sys.path.append(r"/home/changxiaosong/python/malwareTest")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test001")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test002")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test003")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test004")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/AV_Agent_reimpl/test004")
    sys.path.append(r"/home/changxiaosong/python/malwareTest/pr2_new_3")

from AV_Agent_reimpl import test001
from AV_Agent_reimpl import test002
from AV_Agent_reimpl import test003
from AV_Agent_reimpl import test004
from AV_Agent_reimpl.test001 import load_seqs_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='训练集文件路径')
    parser.add_argument('--test', required=True, help='测试集文件路径')
    args = parser.parse_args()

    train_file = args.train
    test_file = args.test
    seqs_test = load_seqs_from_file(test_file)

    test001.main(train_file,test_file)
    test002.main(seqs_test)
    test003.main(seqs_test)
    test004.main(seqs_test)