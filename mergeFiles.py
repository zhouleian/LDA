#coding=utf-8
import sys
import os


print os.getcwd()
print sys.path[0]
def getFileList():
    segPath = sys.path[0] + '/seg_result'
    fileList = os.listdir(segPath)
    return fileList

def mergeFiles(fileList):
    """
    将所有文件合并到一个文件test.txt中
    :param fileList: 文件列表
    :return: 合并后的文件
    """
    filesPath = sys.path[0] + '/seg_result'
    testPath = os.getcwd() + '/LDA'
    #print testPath
    os.chdir(testPath)
    fTest = open('test.txt','w+')
    num = len(fileList)
    fTest.write(str(num) + '\n')
    for eachFile in fileList:
        fileName = filesPath + '/' + eachFile
        f = open(fileName, 'r')
        content = f.read()
        fTest.write(content + '\n')
    fTest.close()
fileList = getFileList()
print len(fileList)
mergeFiles(fileList)
path = os.getcwd()
print path
count = len(open(r'test.txt','rU').readlines())
print count