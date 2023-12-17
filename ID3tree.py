import math
import operator
import matplotlib.pyplot as plt
import Tree_show

def createDataset():

    dataSet =[
        # 14个样本，4个属性
        ['severe', 'hard', 'high', 'Yes', 'Yes'],
         ['strong', 'well', 'well', 'No', 'Yes'],
         ['normal', 'hard', 'high', 'Yes', 'Yes'],
         ['normal', 'hard', 'low', 'No', 'Yes'],
         ['strong', 'hard', 'well', 'Yes', 'Yes'],
         ['strong', 'well', 'well', 'Yes', 'No'],
         ['normal', 'well', 'high', 'Yes', 'No'],
         ['normal', 'hard', 'low', 'No', 'No'],
         ['severe', 'hard', 'well', 'No', 'Yes'],
         ['severe', 'hard', 'high', 'No', 'Yes'],
         ['normal', 'hard', 'well', 'Yes', 'No'],
         ['strong', 'well', 'high', 'Yes', 'Yes'],
         ['strong', 'well', 'low', 'Yes', 'Yes'],
         ['severe', 'well', 'high', 'No', 'No']]

    # 特征值列表

    # 前四列的名字（特征列）分别为心绞痛，呼吸，心率，头晕，是否得心脏病
    labels = ['心绞痛', '呼吸', '心率', '头晕']
    return dataSet,labels

def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLable=featVec[-1]

        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0
        labelCounts[currentLable]+=1
    #print(labelCounts)

    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*math.log(prob,2)

    return shannonEnt

def splitDataSet(dataSet,axis,value):
    retDataSet=[]

    for featVec in dataSet:

        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1

    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0

        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy

        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return  bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #print(sortedClassCount)
    print(type(sortedClassCount))
    print(sortedClassCount)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(dataSet):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

 # # 给定新的症状，来预测是否有得心脏病的风险
def predict_play(tree, new_dic):
        """
        根据构造的决策树，对未知数据进行预测
        :param tree: 决策树（根据已知数据构造的）
        :param new_dic: 一条待预测的数据
        :return:返回叶子节点，也就是最终的决策
        """
        while type(tree).__name__ == "dict":
            key = list(tree.keys())[0]
            tree = tree[key][new_dic[key]]
        return tree



dataSet,labels=createDataset()
myTree=createTree(dataSet,labels)
Tree_show.createPlot(myTree)
print(myTree)
print(predict_play(myTree, {'心绞痛': 'strong', '呼吸': 'hard', '心率': 'low', '头晕': 'Yes'}))

