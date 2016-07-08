# coding=utf-8

"""将文件处理成一个字典"""
import codecs
import sys
import os
import numpy as np
#print os.getcwd()

def countKey(fileName):
    """
    将文件处理为一个字典
    :param fileName:要处理的文件
    :param resultName:处理之后的字典文件
    :return:
    """
    try:
        # 计算文件行数
        lineNums = len(open(fileName, 'rU').readlines())
        #print u'文件行数: ' + str(lineNums)
        # 统计格式 格式<Key:Value> <属性:出现个数>
        i = 0
        table = {}
        source = open(fileName, "r")
        #result = open(resultName, "w")
        while i < lineNums:
            line = source.readline()
            #line = line.rstrip('\n')
            #print line
            words = line.split(" ")  # 空格分隔，words是列表，保存文本中的所有的词，并且是用空格分开的词
            #print u'空格分割的列表===='
            #print str(words).decode('string_escape')
            # 字典插入与赋值
            for word in words:
                if word != "" and table.has_key(word):  # 如果存在次数加1
                    num = table[word]
                    table[word] = num + 1
                elif word != "":  # 否则初值为1
                    table[word] = 1
            i = i + 1
        return table
    except Exception, e:
        print 'Error:', e
    finally:
        source.close()
        #result.close()
        #print 'END\n\n'

def getLContent():
    """
    获得地点内容
    """
    with codecs.open('didian.txt', 'r', 'utf-8') as f:
        docs = f.readlines()
    return docs

def getLTimes(docs,table):
    """获取每一列的数据"""
    f = open('times.txt','w')
    docsLen = len(docs)
    times = np.zeros((docsLen,1),dtype='int')
    num = 0
    for i in xrange(docsLen):
        l = docs[i].encode('utf-8').strip()
        if table.has_key(l):
            times[num] = table[l]
            num += 1
            #f.write(l + ": " + str(table[l]) + '\n')
            f.write(str(table[l]) + '\n')
        else:
            f.write(str('0') + '\n')
            #f.write(l + ": " + str('0') + '\n')

    f.close()
    return times

def getFileList():
    """
    获得seg_result文件夹中的已经分好词的文件
    :return: 一系列文件
    """
    segPath = sys.path[0] + '/seg_result'
    fileList = os.listdir(segPath)
    return fileList
def allCountKey(fileList,docs):
    filesPath = sys.path[0] + '/seg_result'
    # 合并后文件的路径os.getcwd() = /home/zhoulei/travelogue
    deltaPath = os.getcwd() + '/delta'
    # print testPath
    os.chdir(deltaPath)
    docsLen = len(docs)
    delta = np.zeros((docsLen,1),dtype='int')
    for eachFile in fileList:
        fileName = filesPath + '/' + eachFile
        table = countKey(fileName)
        times = getLTimes(docs,table)
        if sum(delta[0]) == 0:
            delta = times
        else:
            delta = np.concatenate((delta,times),axis = 1)
    return delta

def main():
    fileList = getFileList()
    # docs ： 保存地点内容
    docs = getLContent()
    # delta数组初始化为一列0,在allCountKey 中组合,delta的大小会成为L*M
    delta = allCountKey(fileList,docs)

    print delta.shape
    # 出现在文档d中的所有地点数目
    d = 2
    #print sum(delta.T[d])






if __name__=='__main__':
    main()