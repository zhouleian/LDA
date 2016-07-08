# coding=utf-8
import logging
import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os
import delta

from collections import OrderedDict

# 获取当前路径
path = os.getcwd()
# 导入日志配置文件
logging.config.fileConfig("logging.conf")
# 创建日志对象
logger = logging.getLogger()
# loggerInfo = logging.getLogger("TimeInfoLogger")
# Consolelogger = logging.getLogger("ConsoleLogger")

# 导入配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")
# 文件路径
trainfile = os.path.join(path, os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path, os.path.normpath(conf.get("filepath", "wordidmapfile")))
paramfile = os.path.join(path, os.path.normpath(conf.get("filepath", "paramfile")))
topNfile = os.path.join(path, os.path.normpath(conf.get("filepath", "topNfile")))
tassginfile = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))
thetafile = os.path.join(path, os.path.normpath(conf.get("filepath", "thetafile")))
psifile = os.path.join(path, os.path.normpath(conf.get("filepath", "psifile")))
phifile = os.path.join(path, os.path.normpath(conf.get("filepath", "phifile")))
xifile = os.path.join(path, os.path.normpath(conf.get("filepath", "xifile")))
phiglfile = os.path.join(path, os.path.normpath(conf.get("filepath", "phiglfile")))
placeidmapfile = os.path.join(path,os.path.normpath(conf.get('filepath','placeidmapfile')))
# 模型初始参数
#L = int(conf.get('model_args','L'))
K = int(conf.get("model_args", "K"))
glK = int(conf.get("model_args", "glK"))
alpha = float(conf.get("model_args", "alpha"))
beta = float(conf.get("model_args", "beta"))
eta = float(conf.get("model_args", "eta"))
etagl= float(conf.get("model_args", "etagl"))
iter_times = int(conf.get("model_args", "iter_times"))
top_words_num = int(conf.get("model_args", "top_words_num"))
gammaloc = float(conf.get("model_args", "gammaloc"))
gammagl = float(conf.get("model_args", "gammagl"))

class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class DataPreProcessing(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()
        self.place2id = OrderedDict()
        self.placedocs = []

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")
    def cacheplaceidmap(self):
        with codecs.open(placeidmapfile,'w','utf-8') as f:
            #file = codecs.open('placeid.txt','w+','utf-8')
            for place,id in self.place2id.items():
                #f.write(place + '\n')
                #file.write(place +'\n')
                f.write(place + '\t' + str(id) + '\n')
            #file.close()

class LDAModel(object):
    def __init__(self, dpre,deltaNpArray):

        self.dpre = dpre  # 获取预处理参数
        self.deltaNpArray = deltaNpArray
        # 模型参数
        # 聚类个数K也就是主题个数，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta),delta,etaloc,etagl
        #
        self.glK = glK
        self.alpha = alpha
        self.etagl = etagl
        self.gammaloc = gammaloc
        self.gammagl = gammagl

        self.K = K
        self.beta = beta
        self.eta = eta
        self.delta = delta
        self.iter_times = iter_times
        self.top_words_num = top_words_num


        # 文件变量
        # V:词语总数 M:文档总数 L:地点总数
        # 分好词的文件trainfile = train.dat
        # 词对应id文件wordidmapfile
        # 每个主题topN词文件topNfile
        # 最后分派结果文件tassginfile
        # 模型训练选择的参数文件paramfile
        # 词语-局部主题分布文件 philocfile,V*locK的矩阵，p(w|z) = philoc[w][z]
        # 局部主题-地点分布文件 psifile,locK*L的矩阵，p(z|l) = psi[z][l]
        # 地点-文档分布文件 xifile，L * M 的矩阵，p(l|d) = xi[l][d]
        # 词语-全局主题分布文件 phiglgile,V*glK的矩阵，p(w|z') = phigl[w][z']
        # 全局主题-文档分布文件 thetafile, glK*M的矩阵，p(z'|d) = theta[z'][d]

        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.psifile = psifile
        self.phifile = phifile
        self.phiglfile = phiglfile
        self.xifile = xifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile
        self.placeidmapfile = placeidmapfile

        # nw,词语w在局部主题locTopic上的分布，W*K的矩阵，nw[w][z]词语w被分配到局部主题z的次数
        # nwsum 局部主题z包含的词语总数，K*1的矩阵，nwsum[z]局部主题z包含的词语总数
        self.nw = np.zeros((self.dpre.words_count,self.K),dtype='int')
        self.nwsum = np.zeros(self.K,dtype='int')
        # nlz,同时分配到地点l和局部主题z的词语数，L*K的矩阵，nl[l][z]
        # nlsum,地点l包含的总的词语数，L*1
        self.nlz = np.zeros((self.dpre.places_count,self.K),dtype='int')
        self.nlsum = np.zeros(self.dpre.places_count,dtype='int')
        # ndl，M*L，nd[d][l]文档d被分配到地点l(必然也在某个局部主题中）的词语个数
        # nd，M*K，nd[d][z]文档d中备份裴到主题z的总词数
        self.ndl = np.zeros((self.dpre.docs_count,self.dpre.places_count),dtype='int')
        self.nd = np.zeros((self.dpre.docs_count,self.K),dtype='int')
        self.Z = np.array(
            [[0 for y in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])
        # M*doc.size()，Z[i][j]表示第i文档的第j个词分配的主题文档中词的主题分布
        self.P = np.array([[0 for memoryview in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])
        # M*doc_size(),P[i][j] 表示第i文档的第j个词语分配的主题文档中的词的地点分布

        # 随机为文档中的词语分配主题，更新计数向量ndsum,nwsum, 计数矩阵nd，nw
        for d in xrange(len(self.Z)):#self.Z = Z的行数，也就是文档数目
            #self.ndsum[d] = self.dpre.docs[d].length #因为是初始化为0的矩阵，所以文档d包含的词个数等于文档的词个数
            for w in xrange(self.dpre.docs[d].length):
                #从主题编号0-K-1中随机选取一个主题
                topic = random.randint(0, self.K - 1)
                place = random.randint(0,self.dpre.places_count -1)
                #print '-------=============',place
                self.Z[d][w] = topic
                self.P[d][w] = place
                #被分配给topic的词w的自增1
                word = self.dpre.docs[d].words[w]
                self.nw[word][topic] += 1
                self.nwsum[topic] += 1
                self.nlz[place][topic] += 1
                self.nlsum[place] += 1
                self.ndl[d][place] += 1
                self.nd[d][topic] += 1

        #phi : x*y,x：词语数目，y：主题数目，psi:x*y,x:每一行表示主题 y:地点 xi:x:地点 y: 文档
        self.phi = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.words_count)])
        self.psi = np.array([[0.0 for y in xrange(self.dpre.places_count)] for x in xrange(self.K)])
        self.xi = np.array([[0.0 for y in xrange(self.dpre.docs_count)] for x in xrange(self.dpre.places_count)])
    def sampling(self, d, w):
        """
        Gibbs Sampling为当前词重新分配主题,和地点
        :param d: 文档
        :param w: 当前词
        """
        topic = self.Z[d][w] #当前的主题
        place = self.P[d][w]
        word = self.dpre.docs[d].words[w] #对应位置的词的ID

        self.nw[word][topic] -= 1
        self.nwsum[topic] -= 1
        self.nlz[place][topic] -= 1
        self.nlsum[place] -= 1
        self.ndl[d][place] -= 1
        self.nd[d][topic] -= 1

        topicP = self.computeTransProbTopic(d,w,place)
        newTopic = self.multSampleTopic(topicP)
        #print 'topic=='
        placeP = self.computeTransProbPlace(d,w,newTopic,place)
        newPlace = self.multSamplePlace(placeP)
        #print 'placep = '

        change =[]
        change.append(newTopic)
        change.append(newPlace)
        self.nw[word][newTopic] += 1
        self.nwsum[newTopic] += 1
        self.nlz[newPlace][newTopic] += 1
        self.nlsum[newPlace] += 1
        self.ndl[d][newPlace] += 1
        self.nd[d][newTopic] += 1
        return change

    def computeTransProbTopic(self,d,w,place):
        """
        对第d个文档的第w个词语计算Gibbs Sampling过程中的传递概率
        :param d: 文档
        :param w: 词在文档中的编号
        """
        #用于平滑
        Weta = self.dpre.words_count * self.eta
        Kbeta = self.K * self.beta
        word = self.dpre.docs[d].words[w]  # 对应位置的词的ID
        topicP = (self.nw[word] + self.eta)/(self.nwsum + Weta) * \
            (self.nlz[place] + self.beta)/(self.nlsum[place] + Kbeta)
        # p,概率向量 double类型，存储采样的临时变量,p是马尔可夫传递概率，p[z]表示当前词语被分配到主题z的概率
        #print 'topic-len',len(topicP)
        #print 'topic.shape = ',topicP.shape
        return topicP
    def multSampleTopic(self,proList):
        """
        从多项分布中proList采样，proList表示剔除当前词之后的主题分布
        :param proList: 多项分布，在这里是马尔可夫传递概率
        """
        for k in xrange(1, self.K):
            proList[k] += proList[k-1]
        newTopic = 0
        u = random.uniform(0, proList[self.K - 1])
        for topic in xrange(self.K):
            if proList[topic] > u:
                newTopic = topic
                break
        return newTopic

    def computeTransProbPlace(self, d, w,topic,place):
        """
        对第d个文档的第w个词语计算Gibbs Sampling过程中的传递概率
        :param d: 文档
        :param w: 词在文档中的编号
        """
        # 用于平滑
        Kbeta = self.K * self.beta
        mu = 0.01
        #print '----',len(self.nlz.T[topic])
        #print self.nlz.shape

        #print mu*self.deltaNpArray[place][d]
        #print '=====================',sum(mu*self.deltaNpArray.T[d])
        placeP = (self.nlz.T[topic] + self.beta)/(self.nlsum + Kbeta) *\
                  (self.ndl[d] + mu*self.deltaNpArray[place][d])/(self.nd[d][topic] + sum(mu*self.deltaNpArray.T[d]))
        # p,概率向量 double类型，存储采样的临时变量,p是马尔可夫传递概率，p[z]表示当前词语被分配到主题z的概率
        #print '****************',len(placeP)
        #print placeP
        return placeP
    def multSamplePlace(self, proList):
        """
        从多项分布中proList采样，proList表示剔除当前词之后的地点分布
        :param proList: 多项分布，在这里是马尔可夫传递概率
        """
        for l in xrange(1, self.dpre.places_count):
            proList[l] += proList[l - 1]
        newPlace = 0
        u = random.uniform(0, proList[self.dpre.places_count - 1])
        for place in xrange(self.dpre.places_count):
            if proList[place] > u:
                newPlace = place
                break
        return newPlace
    def est(self):
        # Consolelogger.info(u"迭代次数为%s 次" % self.iter_times)
        #对于每次迭代，每篇文档中的每个词语，分配新的主题，并更新z矩阵
        for x in xrange(self.iter_times):
            print 'item = ',x
            for d in xrange(self.dpre.docs_count):
                for w in xrange(self.dpre.docs[d].length):
                    change = self.sampling(d,w)
                    topic = change[0]
                    place = change[1]
                    self.Z[d][w] = topic
                    self.P[d][w] = place
        logger.info(u"迭代完成。")
        logger.debug(u"计算词语-主题分布")
        self.computePhi()
        logger.debug(u"计算主题-地点分布")
        self.computePsi()
        # logger.debug(u'计算地点-文档分布')
        # self.computeXi()
        logger.debug(u"保存模型")
        self.save()

    def computePhi(self):
        """
        计算phi也就是p(w|z)
        """
        for k in xrange(self.K):
            self.phi.T[k] = (self.nw.T[k] + self.eta)/(self.nwsum[k] + self.dpre.words_count*self.eta)
    def computePsi(self):
        """
        计算psi也就是p(z|l)
        """
        for l in xrange(self.dpre.places_count):
            self.psi.T[l] = (self.nlz[l] + self.beta)/(self.nlsum[l] + self.K*beta)
    # def computeXi(self):
    #     """
    #     计算xi也就是p(l|d)
    #     """
    #     mu = 0.1
    #     for d in xrange(self.dpre.docs_count):
    #         self.xi.T[d] = (self.nd[d] + mu*self.deltaNpArray[l][d])/(self.nd[d] + sum(mu*self.deltaNpArray.T[d]))

    def save(self):
        # 保存psi主题-地点分布
        logger.info(u'主题-地点分布已保存到%s' % self.psifile)
        with codecs.open(self.psifile,'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.dpre.places_count):
                    f.write(str(self.psi[x][y]) + '\t')
                f.write('\n')
        # 保存phi词-主题分布
        logger.info(u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for x in xrange(self.dpre.words_count):
                for y in xrange(self.K):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')

        logger.info(u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile, 'w', 'utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('eta=' + str(self.eta) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')
        # 保存每个主题topic的词
        logger.info(u"主题topN词已保存到%s" % self.topNfile)

        with codecs.open(self.topNfile, 'w', 'utf-8') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in xrange(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = []
                twords = [(n, self.phi[n][x]) for n in xrange(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in xrange(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')
        # 保存最后退出时，文章的词分派的主题的结果
        logger.info(u"文章-词-主题分派结果已保存到%s" % self.tassginfile)
        with codecs.open(self.tassginfile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')
                f.write('\n')
        logger.info(u"模型训练完成。")


def preprocessing():
    logger.info(u'载入数据......')
    with codecs.open(trainfile, 'r', 'utf-8') as f:
        docs = f.readlines()
    logger.debug(u"载入完成,准备生成字典对象和统计文本数据...")
    dpre = DataPreProcessing()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split()#将tmp也就是docs中的一行，即一个文档的内容，根据里面的空格分开成为tmp列表，len(tmp)就是一篇文档中词语数
            # 生成一个文档对象
            doc = Document()
            for item in tmp: #对tmp中的每一项，也就是对每篇文档中每一个词语
                if dpre.word2id.has_key(item):
                    doc.words.append(dpre.word2id[item])
                else:
                    dpre.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            dpre.docs.append(doc)
        else:
            pass
    dpre.docs_count = len(dpre.docs) #dpre.docs 是一个列表的列表,len(dpre.docs)表示docs里有几个小列表，也就是有几个文档
    dpre.words_count = len(dpre.word2id)# 整个训练集合中的不重复的词语个数
    logger.info(u"共有%s个文档" % dpre.docs_count)
    logger.info(u'%s个词语'%dpre.words_count)
    dpre.cachewordidmap()
    logger.info(u"词与序号对应关系已保存到%s" % wordidmapfile)

    with codecs.open('didian.txt', 'r', 'utf-8') as f:
        placedocs = f.readlines()
    #print '--------------',len(placedocs)
    itemsidex = 0
    for line in placedocs:
        #print line
        if line != "":
            tmp = line.strip()
            # 生成一个文档对象
            doc = Document()
            if dpre.place2id.has_key(tmp):
                doc.words.append(dpre.place2id[tmp])
            else:
                dpre.place2id[tmp] = itemsidex
                doc.words.append(itemsidex)
                itemsidex += 1
            doc.length = len(tmp)
            dpre.placedocs.append(doc)
        else:
            pass
    dpre.places_count = len(dpre.place2id)  # 整个训练集合中的不重复的词语个数
    logger.info(u"共有%s个地点" % dpre.places_count)
    dpre.cacheplaceidmap()
    logger.info(u"词与序号对应关系已保存到%s" % placeidmapfile)
    return dpre

def getdeltaNpArray():
    # 引入deltaNpArray
    fileList = delta.getFileList()
    didian = delta.getLContent()
    # delta数组初始化为一列0,在allCountKey 中组合,最后会多出最前面的一列全0的列，
    deltaNpArray = delta.allCountKey(fileList, didian)
    return deltaNpArray
def run():
    dpre = preprocessing()
    deltaNpArray = getdeltaNpArray()
    lda = LDAModel(dpre,deltaNpArray)
    lda.est()


if __name__ == '__main__':
    run()