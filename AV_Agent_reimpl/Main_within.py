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

from AV_Agent_reimpl import test001
from AV_Agent_reimpl import test002
from AV_Agent_reimpl import test003
from AV_Agent_reimpl import test004
from AV_Agent_reimpl.test001 import load_seqs_from_file

if __name__ == '__main__':
    # train_file = r"/home/changxiaosong/python/malwareTest/123.txt"
    # test_file = r"/home/changxiaosong/python/malwareTest/123.txt"
    # train_file = r"D:\研究生\日常文件备份\123.txt"
    # test_file = r"D:\研究生\日常文件备份\123.txt"
    # train_file=r'/home/changxiaosong/python/malwareTest/train_0.5begin-dowgin-smsreg-smssend-.txt'
    # test_file=r'/home/changxiaosong/python/malwareTest/test_0.5begin-dowgin-smsreg-smssend-.txt'
    train_file=r'/home/changxiaosong/python/malwareTest/cic-train-within.txt'
    test_file=r'/home/changxiaosong/python/malwareTest/cic-test-within.txt'
    seqs_test = load_seqs_from_file(test_file)

    test001.main(train_file,test_file)
    test002.main(seqs_test)
    test003.main(seqs_test)
    test004.main(seqs_test)