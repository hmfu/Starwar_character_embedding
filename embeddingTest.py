import numpy as np
import pickle

class EmbeddingTest():
    def __init__(self, fileName):
        self.embedTable = None
        self.biasTable = None
        self.indToTokDict = None
        self.tokDict = None
        self.tokenNum = None

    def loadData(self, fileName):
        with open(fileName, 'rb') as inFile:
            self.embedTable, self.biasTable, self.indToTokDict = pickle.load(inFile)
        
        self.tokenList = sorted(list(self.indToTokDict.values()))
        self.tokDict = {v:k for k, v in self.indToTokDict.items()}
        self.tokenNum = len(self.tokenList)

    def getCloseRank(self, token, useDist):
        ind = self.tokDict[token]

        if useDist:
            distArr = np.linalg.norm(self.embedTable - self.embedTable[ind], axis = 1)
            return sorted([(self.indToTokDict[ind], distArr[ind]) for ind in range(self.tokenNum)], key = lambda x:x[1])
        else:
            simArr = np.sum(self.embedTable[ind] * self.embedTable, axis = 1) + self.biasTable[:, 0] + self.biasTable[ind, 0]
            return sorted([(self.indToTokDict[ind], simArr[ind]) for ind in range(self.tokenNum)], key = lambda x:x[1], reverse=1)

if __name__ == '__main__':
    et = EmbeddingTest(fileName = 'tableTup.pkl')
    et.loadData(fileName = 'tableTup.pkl')
    
    token = input('which character are you searching for? list: \n' + str(et.tokenList) + '\n')
    
    print ('Top candidates: ')
    print ("{:<15}".format('character'), 'inner product or distance')
    for tup in et.getCloseRank(token, useDist = 0):
        print ("{:<15}".format(str(tup[0])), tup[1])
