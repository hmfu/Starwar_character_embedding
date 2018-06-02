import os
import numpy as np
import tensorflow as tf
import pickle

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

class DataFormater():
    def __init__(self):
        self.seqList = None
        self.tokenDict = None
        self.tokenNum = None
        self.indToTokDict = None
        self.code1Arr = None
        self.code2Arr = None
        self.wtFuncArr = None
        self.labelArr = None
        self.dataTup = None

    def loadSeq(self, fileNameList):
        self.seqList = []
        for fileName in fileNameList:
            seq = []
            with open(fileName) as inFile:
                next(inFile)
                for line in inFile:
                    token = line.rstrip().split(' ')[1][1:-1]
                    seq += [token]
            self.seqList += [seq]
        
    def buildTokenDict(self):
        tokenSet = set()
        for seq in self.seqList:
            for token in seq:
                tokenSet.add(token)
        self.tokenDict = {}
        nextCode = 0
        for token in tokenSet:
            self.tokenDict[token] = nextCode
            nextCode += 1

        self.indToTokDict = {v: k for k, v in self.tokenDict.items()}
        
        self.tokenNum = nextCode
                
    def tokenToCode(self):
        nl = [[self.tokenDict[seq[ind]] for ind in range(len(seq)) if ind == 0 or seq[ind] != seq[ind-1]] for seq in self.seqList]
        self.seqList = nl

    def buildData(self, windowSize, alpha, xMax):
        pairScoreDict = {}

        for seq in self.seqList:
            for ind in range(len(seq) - windowSize):
                code1 = seq[ind]
                for offset in range(1, windowSize + 1):
                    code2 = seq[ind + offset]
                    tup = (min(code1, code2), max(code1, code2))
                    pairScoreDict[tup] = pairScoreDict.get(tup, 0) + 1 / offset

        tupList = list(pairScoreDict.keys())
        tupNum = len(tupList)

        self.code1Arr = np.empty(shape = (tupNum, 1))
        self.code2Arr = np.empty(shape = (tupNum, 1))
        self.wtFuncArr = np.empty(shape = (tupNum, 1))
        self.labelArr = np.empty(shape = (tupNum, 1))
        
        for tupInd in range(tupNum):
            tup = tupList[tupInd]
            score = pairScoreDict[tup]
            self.code1Arr[tupInd, 0] = tup[0]
            self.code2Arr[tupInd, 0] = tup[1]
            self.wtFuncArr[tupInd, 0] = np.minimum(score / xMax, 1.0)
            self.labelArr[tupInd, 0] = np.log(score)

        self.dataTup = (self.code1Arr, self.code2Arr, self.wtFuncArr, self.labelArr)

    def saveMetaData(self, fileName):
        with open(fileName, 'w') as outFile:
            for ind in range(self.tokenNum):
                outFile.write(str(self.indToTokDict[ind]) + '\n')
        print ('meta data %d is saved.' % fileName)

class GloveModel():
    def __init__(self):
        self.embedLookupTable = None
        self.biasLookupTable = None
        self.embedLookupTableArr = None
        self.biasLookupTableArr = None
        self.inputCode1PH = None
        self.inputCode2PH = None
        self.wtFuncPH = None
        self.labelPH = None
        self.optimizer = None
        self.sess = None
        self.loss = None

    def buildModel(self, embedDim, tokenNum):
        self.embedLookupTable = tf.Variable(tf.random_normal([tokenNum, embedDim], stddev = 0.1, name = 'embedLookupTable'))
        self.biasLookupTable = tf.Variable(tf.random_normal([tokenNum, 1], stddev = 0.001, name = 'biasLookupTable'))

        self.inputCode1PH = tf.placeholder('int32', [None, 1])
        self.inputCode2PH = tf.placeholder('int32', [None, 1])
        self.inputWtFuncPH = tf.placeholder('float32', [None, 1])
        self.inputLabelPH = tf.placeholder('float32', [None, 1])

        embed1PH = tf.nn.embedding_lookup(self.embedLookupTable, self.inputCode1PH)
        embed2PH = tf.nn.embedding_lookup(self.embedLookupTable, self.inputCode2PH)
        bias1PH = tf.nn.embedding_lookup(self.biasLookupTable, self.inputCode1PH)
        bias2PH = tf.nn.embedding_lookup(self.biasLookupTable, self.inputCode2PH)

        self.loss = tf.reduce_sum(self.inputWtFuncPH * tf.square(embed1PH * embed2PH + bias1PH + bias2PH - self.inputLabelPH))

    def trainModel(self, learningRate, epochNum, evalEpochNum, dataTup):
        code1Arr, code2Arr, wtFuncArr, labelArr = dataTup
        feedDict = {self.inputCode1PH:code1Arr,self.inputCode2PH:code2Arr,self.inputWtFuncPH:wtFuncArr,self.inputLabelPH:labelArr}

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(self.loss)
        with tf.Session(config = config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            
            for epochInd in range(epochNum):
                sess.run(self.optimizer, feed_dict = feedDict)
                
                if epochInd % evalEpochNum == 0:
                    print ('epochInd: %d, loss: ' % epochInd, sess.run(self.loss, feed_dict = feedDict))
            
            self.embedLookupTableArr = sess.run(self.embedLookupTable)
            self.biasLookupTableArr = sess.run(self.biasLookupTable)

    def saveEmbedding(self, fileName, tokDict):
        with open(fileName, 'wb') as outFile:
            pickle.dump((self.embedLookupTableArr, self.biasLookupTableArr, tokDict), outFile, protocol=pickle.HIGHEST_PROTOCOL) 
            print ('data tuple saved as %s.' % fileName)

class Visualizer():
    def __init__(self):
        pass

    def saveGraph(self, FileWriterDir, dataTupFileName, metaDataPath):
        with open(dataTupFileName, 'rb') as inFile:
            embedLookupTableArr, biasLookupTableArr, tokDict = pickle.load(inFile)
        
        with tf.Session(config = config) as sess:

            embedLookupTableArrVar = tf.Variable(embedLookupTableArr)
            sess.run(embedLookupTableArrVar.initializer)
            
            writer = tf.summary.FileWriter(FileWriterDir, sess.graph)
            projectorConfig = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embedding = projectorConfig.embeddings.add()
            embedding.metadata_path = metaDataPath
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, projectorConfig)
            
            saver_embed = tf.train.Saver([embedLookupTableArrVar])
            saver_embed.save(sess, os.path.join(FileWriterDir, './embedding.ckpt'), 1)

        writer.close()
        print ('graph for visualization saved to %d' % FileWriterDir)

if __name__ == '__main__':
    df = DataFormater()

    fileList = ['SW_EpisodeIV.txt', 'SW_EpisodeV.txt', 'SW_EpisodeVI.txt']
    
    df.loadSeq([os.path.join('star-wars-movie-scripts/', fileName) for fileName in fileList])
    df.buildTokenDict()
    df.tokenToCode()
    df.buildData(windowSize = 5, alpha = 0.75, xMax = 10)
    df.saveMetaData(fileName = 'metaData.csv')

    gm = GloveModel()
    gm.buildModel(embedDim = 10, tokenNum = df.tokenNum)
    gm.trainModel(learningRate = 0.01, epochNum = 1000, evalEpochNum = 50, dataTup = df.dataTup)
    gm.saveEmbedding(fileName = 'tableTup.pkl', tokDict = df.indToTokDict)

    v = Visualizer()
    v.saveGraph(FileWriterDir = './graphs/embedding', dataTupFileName = 'tableTup.pkl', metaDataPath = 'metaData.csv')
