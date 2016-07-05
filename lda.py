# coding=utf-8
import logging
import logging.config
import ConfigParser
import numpy as np
import random
import codecs
import os

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
thetafile = os.path.join(path, os.path.normpath(conf.get("filepath", "thetafile")))
phifile = os.path.join(path, os.path.normpath(conf.get("filepath", "phifile")))
paramfile = os.path.join(path, os.path.normpath(conf.get("filepath", "paramfile")))
topNfile = os.path.join(path, os.path.normpath(conf.get("filepath", "topNfile")))
tassginfile = os.path.join(path, os.path.normpath(conf.get("filepath", "tassginfile")))
# 模型初始参数
K = int(conf.get("model_args", "K"))
alpha = float(conf.get("model_args", "alpha"))
beta = float(conf.get("model_args", "beta"))
iter_times = int(conf.get("model_args", "iter_times"))
top_words_num = int(conf.get("model_args", "top_words_num"))


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

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w', 'utf-8') as f:
            for word, id in self.word2id.items():
                f.write(word + "\t" + str(id) + "\n")


class LDAModel(object):
    def __init__(self, dpre):

        self.dpre = dpre  # 获取预处理参数
        # 模型参数
        # 聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
        #
        self.K = K
        self.beta = beta
        self.alpha = alpha
        self.iter_times = iter_times
        self.top_words_num = top_words_num
        #
        # 文件变量
        # 分好词的文件trainfile = train.dat
        # 词对应id文件wordidmapfile
        # 文章-主题分布文件thetafile，M*K的矩阵，p(z|d) = theta[d][z]
        # 词-主题分布文件phifile,K*V的矩阵，p(w|z) = phi[z][w]
        # 每个主题topN词文件topNfile
        # 最后分派结果文件tassginfile
        # 模型训练选择的参数文件paramfile
        #
        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile
        # p,概率向量 double类型，存储采样的临时变量,p是马尔可夫传递概率，p[z]表示当前词语被分配到主题z的概率
        # nw,词word在主题topic上的分布， W*K的矩阵,nw[w][z]表示词w被分配到主题z的概率
        # nwsum,每个topic的词的总数，K*1的向量，nwsum[z] 表示主题z中包含的词语的个数
        # nd,每个doc中各个topic的词的总数，M*K的矩阵,nd[d][z]表示文档d中被分配到主题z的词的个数
        # ndsum,每个doc中词的总数，M*1的向量，ndsum[d]表示文档d中包含的词的个数
        #self.p = np.zeros(self.K)
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype="int")
        self.nwsum = np.zeros(self.K, dtype="int")
        self.nd = np.zeros((self.dpre.docs_count, self.K), dtype="int")
        self.ndsum = np.zeros(dpre.docs_count, dtype="int")
        self.Z = np.array(
            [[0 for y in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])
        # M*doc.size()，Z[i][j]表示第i文档的第j个词分配的主题文档中词的主题分布

        # 随机为文档中的词语分配主题，更新计数向量ndsum,nwsum, 计数矩阵nd，nw
        for d in xrange(len(self.Z)):
            self.ndsum[d] = self.dpre.docs[d].length #因为是初始化为0的矩阵，所以文档d包含的词个数等于文档的词个数
            for w in xrange(self.dpre.docs[d].length):
                #从主题编号0-K-1中随机选取一个主题
                topic = random.randint(0, self.K - 1)
                self.Z[d][w] = topic
                #被分配给topic的词w的自增1
                self.nw[self.dpre.docs[d].words[w]][topic] += 1
                self.nd[d][topic] += 1
                self.nwsum[topic] += 1
        #theta:x*y的数组，K是主题，每一列表示一个主题，dpre.docs_count:文档数目，phi：x*y的数组dpre.words_count：词语数目
        self.theta = np.array([[0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count)])
        self.phi = np.array([[0.0 for y in xrange(self.dpre.words_count)] for x in xrange(self.K)])

    def sampling(self, d, w):
        """
        Gibbs Sampling为当前词重新分配主题
        :param d: 文档
        :param w: 当前词
        """
        topic = self.Z[d][w] #当前的主题
        word = self.dpre.docs[d].words[w] #对应位置的词的ID
        self.nw[word][topic] -= 1
        self.nd[d][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[d] -= 1

        p = self.computeTransProb(d,w)
        newTopic = self.multSample(p)

        self.nw[word][newTopic] += 1
        self.nwsum[newTopic] += 1
        self.nd[d][newTopic] += 1
        self.ndsum[d] += 1
        return newTopic

    def computeTransProb(self,d,w):
        """
        对第d个文档的第w个词语计算Gibbs Sampling过程中的传递概率
        :param d: 文档
        :param w: 词在文档中的编号
        """
        #用于平滑
        Wbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha
        word = self.dpre.docs[d].words[w] #对应位置的词的ID
        p = np.zeros(self.K)
        p = (self.nw[word] + self.beta) / (self.nwsum + Wbeta) * \
                 (self.nd[d] + self.alpha) / (self.ndsum[d] + Kalpha)
        return p
    def multSample(self,proList):
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

    def est(self):
        # Consolelogger.info(u"迭代次数为%s 次" % self.iter_times)
        #对于每次迭代，每篇文档中的每个词语，分配新的主题，并更新z矩阵
        for x in xrange(self.iter_times):
            for d in xrange(self.dpre.docs_count):
                for w in xrange(self.dpre.docs[d].length):
                    topic = self.sampling(d,w)
                    self.Z[d][w] = topic
        logger.info(u"迭代完成。")
        logger.debug(u"计算文章-主题分布")
        self.computeTheta()
        logger.debug(u"计算词-主题分布")
        self.computePhi()
        logger.debug(u"保存模型")
        self.save()

    def computeTheta(self):
        """
        计算theta也就是p(z|d),M*K
        """
        for d in xrange(self.dpre.docs_count):
            self.theta[d] = (self.nd[d] + self.alpha) / (self.ndsum[d] + self.K * self.alpha)
    def computePhi(self):
        """
        计算phi即p(w|z),K*W
        :return:
        """
        #slef.nw.T代表的是nw矩阵的转置
        for k in xrange(self.K):
            self.phi[k] = (self.nw.T[k] + self.beta) / (self.nwsum[k] + self.dpre.words_count * self.beta)

    def save(self):
        # 保存theta文章-主题分布
        logger.info(u"文章-主题分布已保存到%s" % self.thetafile)
        with codecs.open(self.thetafile, 'w') as f:
            for x in xrange(self.dpre.docs_count):
                for y in xrange(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
        # 保存phi词-主题分布
        logger.info(u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile, 'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
        # 保存参数设置
        logger.info(u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile, 'w', 'utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
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
                twords = [(n, self.phi[x][n]) for n in xrange(self.dpre.words_count)]
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
            tmp = line.strip().split()
            # 生成一个文档对象
            doc = Document()
            for item in tmp:
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
    dpre.docs_count = len(dpre.docs)
    dpre.words_count = len(dpre.word2id)
    logger.info(u"共有%s个文档" % dpre.docs_count)
    dpre.cachewordidmap()
    logger.info(u"词与序号对应关系已保存到%s" % wordidmapfile)
    return dpre


def run():
    dpre = preprocessing()
    lda = LDAModel(dpre)
    lda.est()


if __name__ == '__main__':
    run()