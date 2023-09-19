import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import math
import random
class ItemBasedCF:
    def __init__(self,data,item_data):
        self.data=data
        self.trainData={}
        self.testdata={}
        self.item_data=item_data
        self.threshold=0.8
        self.itemSimMatrix=[]
    def preprocessData(self):#有问题
        traindata_list={}
        testdata_list={}
        item_all=[]
        for user in self.data:
            for item in self.data[user]:
                item_all.append(item[0])##string格式
        item_all=list(set(item_all))
        for user in self.data:#用户遍历
            traindata_list.setdefault(user, {})
            testdata_list[user]={}
            get_item=[]
            for item in self.data[user]:
                get_item.append(item[0])
            choose_num=int(0.3*len(get_item))##这里有问题
            choose_no=random.sample(list(range(0,len(get_item))),choose_num)
            # tt=random.sample(choose_no,choose_num)

            for i in range(len(item_all)):
                if item_all[i] in get_item:
                    traindata_list[user][item_all[i]]=1 
                if item_all[i] in get_item:
                    testdata_list[user][item_all[i]]=1 
                else:
                    traindata_list[user][item_all[i]]=0
                    testdata_list[user][item_all[i]]=0
        #决定测试集先随机按7:3的比例划分
        self.trainData=traindata_list
        self.testdata=testdata_list

    def itemSimilarity(self):
        self.itemSimMatrix=dict()
        item_item_matrix=dict()
        item_user_matrix=dict()
        for user,items in self.trainData.items():
            for itemId,source in items.items():
                item_user_matrix.setdefault(itemId,0)
                item_user_matrix[itemId]+=1
                item_item_matrix.setdefault(itemId,{})
                for i in items.keys():
                    if i==itemId:
                        continue
                    item_item_matrix[itemId].setdefault(i, 0)
                    # 计算同时给两个物品打分的人数，并对活跃用户进行惩罚
                    item_item_matrix[itemId][i] += 1 / math.log(1 + len(items) * 1.0)
        for itemId, relatedItems in item_item_matrix.items():
            self.itemSimMatrix.setdefault(itemId, dict()) # 初始化self.itemSimMatrix[itemId]
            for relatedItemId, count in relatedItems.items():
                self.itemSimMatrix[itemId][relatedItemId] = count / math.sqrt(item_user_matrix[itemId] * item_user_matrix[relatedItemId])
            # 归一化
            sim_max = max(self.itemSimMatrix[itemId].values())
            for item in self.itemSimMatrix[itemId].keys():
                self.itemSimMatrix[itemId][item] /= sim_max

    def evaluate(self):
        rank = dict()
        test_pred={}
        for user in self.testdata:
            test_pred[user]={}
            interacted_items = self.trainData.get(user, {}) # 当前用户已经交互过的item
        # 遍历用户评分的物品列表
            for itemId, score in interacted_items.items(): # 取出每一个当前用户交互过的item
            # 取出与物品itemId最相似的K个物品及其评分
                for i, sim_ij in sorted(self.itemSimMatrix[itemId].items(), key=lambda x: x[1], reverse=True):
                # 如果这个物品j已经被用户评分了，舍弃
                    # if i in interacted_items.keys() and interacted_items[i]==1:
                        # continue
                # 对物品ItemID的评分*物品itemId与j的相似度之和
                    rank.setdefault(i, 0)
                    rank[i] += score * sim_ij
                    if rank[i]>=self.threshold:
                        test_pred[user][i]=1
            for item in self.item_data:
                if item not in test_pred[user]:
                    test_pred[user][item]=0
        y_true=[]
        y_pred=[]
        for user in self.testdata:
            for item in self.item_data:
                if item in self.testdata[user]:
                    y_true.append(self.testdata[user][item])
                else:
                    y_true.append(0)
        for user in test_pred:
            for item in self.item_data:
                if item in test_pred[user]:
                    y_pred.append(test_pred[user][item])
                else:
                    y_pred.append(0)
        # 堆排序，推荐权重前N的物品
        tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    # 计算 ROC 曲线和 AUC
        # fpr, tpr, thresholds = roc_curve(y_true)
        # roc_auc = auc(fpr, tpr)
        print("------evaluation-------")
        print("precision:",precision,"recall:",recall,"f1:",f1)
    def recommend(self,user_id,k,N):
        rank = dict()
        interacted_items = self.trainData.get(user_id, {}) # 当前用户已经交互过的item
        # 遍历用户评分的物品列表
        for itemId, score in interacted_items.items(): # 取出每一个当前用户交互过的item
            # 取出与物品itemId最相似的K个物品及其评分
            for i, sim_ij in sorted(self.itemSimMatrix[itemId].items(), key=lambda x: x[1], reverse=True)[0:k]:
                # 如果这个物品j已经被用户评分了，舍弃
                if i in interacted_items.keys() and interacted_items[i]==1:
                    continue
                # 对物品ItemID的评分*物品itemId与j的相似度 之和
                rank.setdefault(i, 0)
                rank[i] += score * sim_ij
        # 堆排序，推荐权重前N的物品
        return dict(sorted(rank.items(), key=lambda x:x[1], reverse=True)[0:N])
