import pandas as pd
import mysql
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from gensim.models import Word2Vec,KeyedVectors
from sklearn.metrics import roc_curve, auc
import mysql.connector
from config.settings import *
import time
from sklearn.model_selection import train_test_split
import pickle
import argparse
import random
from util.item_cf import ItemBasedCF
from util.get_data import get_dc_components,get_sms_api_log,get_business_order
def hot_calculate(conn,component_id_list,system_log,month,N):#统计一下每个月点击率最高的元件
    item_dict={}
    cursor=conn.cursor()
    for data in system_log:
        component_name=data[4]
        component_id=data[3]
        merchant_name=data[6]
        component_form=1 
        call_time=data[8]
        if component_id not in component_id_list:
            continue
        if component_id in component_id_list and component_id_list[component_id]["status"]!=21:
            continue
        item_dict.setdefault(data[3], {"merchant_name":merchant_name,"call_number":0,"component_form":component_form,"component_name":component_name})
        # 将字符串转换为 datetime 对象
        # 提取月份
        call_month = call_time.month
        if call_month==month:
            item_dict[data[3]]["call_number"]+=1
    sorted_dict = dict(sorted(item_dict.items(), key=lambda item: item[1]["call_number"], reverse=True))
    ##recommended_result is
    recommend_dict=[]
    for index, da in enumerate(sorted_dict):
        local_time=time.localtime(time.time())
        recommend_time=time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        new_da=(da,sorted_dict[da]["component_name"],sorted_dict[da]["component_form"],recommend_time,index,sorted_dict[da]["merchant_name"])
        recommend_dict.append(new_da)
        if index>=N:
            break
    #写入表内
    # 查询表是否存在
        # 执行创建数据表的 SQL 命令
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS zz_de_admin.recommended_result_hot(
        component_id BIGINT(18) NOT NULL,
        component_name VARCHAR(255) NOT NULL,
        component_form BIGINT(18) NOT NULL,
        recommended_time VARCHAR(20) NOT NULL,
        recommended_score INT NOT NULL,
        merchant_name VARCHAR(255) NOT NULL
    )
'''
    cursor.execute(create_table_query)
        # 提交事务
    conn.commit()
    print("-----清除recommended_result_hot数据------")
    cursor1 = conn.cursor()
    delete_query = "DELETE FROM zz_de_admin.recommended_result_hot"
    cursor1.execute(delete_query)
    conn.commit()
    cursor1.close()
    print("-----插入recommended_result_hot数据------")
    cursor2 = conn.cursor()
    insert_query = "INSERT INTO zz_de_admin.recommended_result_hot (component_id, component_name,component_form,recommended_time,recommended_score,merchant_name) VALUES (%s,%s,%s,%s,%s,%s)"
    cursor2.executemany(insert_query, recommend_dict)
    # 提交事务
    conn.commit()
    cursor2.close()
    print("-----finish!------")
    return recommend_dict
def deal_sql(conn,recommend_dict):
    cursor=conn.cursor()
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS zz_de_admin.recommended_result(
        user_id BIGINT(18),
        user_name VARCHAR(20) NOT NULL,
        component_id BIGINT(18),
        component_name VARCHAR(255) NOT NULL,
        component_form BIGINT(18) NOT NULL,
        description VARCHAR(255) NOT NULL,
        merchant_id VARCHAR(20) NOT NULL,
        merchant_name VARCHAR(20) NOT NULL,
        recommended_source VARCHAR(50) NOT NULL,
        recommended_time VARCHAR(20) NOT NULL,
        recommended_score FLOAT NOT NULL
    )
    '''
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    print("-----清除recommended_result数据------")
    cursor1 = conn.cursor()
    delete_query = "DELETE FROM zz_de_admin.recommended_result"
    cursor1.execute(delete_query)
    conn.commit()
    cursor1.close()
    print("-----插入recommended_result数据------")
    cursor2 = conn.cursor()
    insert_query = "INSERT INTO zz_de_admin.recommended_result (user_id, user_name,component_id,component_name,component_form,description,merchant_id,merchant_name,recommended_source,recommended_time,recommended_score) VALUES (%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s)"
    cursor2.executemany(insert_query, recommend_dict)
    # 提交事务
    conn.commit()
    cursor2.close()
    print("-----finish!------")
def get_table(conn,name:str):
    cursor1=conn.cursor()
    query = "SELECT * FROM "+name
    data=[]
    cursor1.execute(query)
    data=cursor1.fetchall()
    cursor1.close()
    return data
def build_connection(parser):
# 建立连接
    conn = mysql.connector.connect(
    host=parser.host,  # 远程MySQL服务器地址
    port=parser.port,
    user=parser.user,      # 数据库用户名
    password=parser.password  # 数据库密码
)
    return conn
def deal_output(model_name,data,data_info):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
        print("----recommend----")
        data_value=model.predict_proba(data)##输入特征有问题
        for da,sco in zip(data_info,data_value):
            da["recommended_score"]=sco[1]
            if "decision_tree_model.pkl" in model_name:
                da["recommended_source"]="decision_tree_model"
            elif "logistic_regression_model.pkl" in model_name:
                da["recommended_source"]="logistic_regression_model"
        sorted_list_desc = sorted(data_info, key=lambda x: (x['user'], x['recommended_score']), reverse=True)
        user_counts = {}
        result_list = []
        for item in sorted_list_desc:
            user = item['user']
            if user in user_counts:
                user_counts[user] += 1
            else:
                user_counts[user] = 1
            if user_counts[user] <= args.recommend_num:
                result_list.append(tuple(item.values()))
        deal_sql(conn,result_list)
def get_info():
    #会员用户表
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',help='sql服务器地址',default=SQL_HOST)
    parser.add_argument('--port',help='sql服务器端口地址',default=SQL_PORT)
    parser.add_argument('--user',help='数据库用户名',default=SQL_USER)
    parser.add_argument('--password',help='数据库密码',default=SQL_PASSWORD)
    parser.add_argument('--choose_model',help='选用的模型(decision、lc、filter)',default=CHOOSE_MODEL)
    parser.add_argument('--modes',help='测试/训练/输出（test/train/output/hot_culculate）',default=MODES)
    parser.add_argument('--recommend_num',help='推荐数量',default=RECOMMEND_NUM)
    parser.add_argument('--cal_month',help='热点统计月份',default=RECOMMEND_MONTH)
    parser.add_argument('--recall_num',help='召回数量',default=RECALL_NUM)
    parser.add_argument('--limit_percent',help='限制概率',default=LIMIT_PERCENT)
    parser.add_argument('--hot',help='是否进行热点统计',default=HOT_CALCULATE)
    args = parser.parse_args()
    conn=build_connection(args)
    #我需要获取的信息：1、所有元件信息[从里面需要分析出：元件特征、新元件、所有元件列表]
                    #2、系统调用日志：用户ID、被调用元件id
                    #3、所有用户信息：
    #先获取所有元件
    dc_data=get_table(conn,"zz_de_admin.dc_components")
    component_id_list={}
    st=time.time()
    wv_from_text = KeyedVectors.load_word2vec_format("model/sgns.wiki.bigram-char", binary=False)
    print("time:",time.time()-st)
    component_id_list=get_dc_components(dc_data,wv_from_text)
    item_list=component_id_list.keys()#所有item列表
    # 从用户系统日志获得买方用户ID：user_id  ##component_id：被调用元件id
    user={}
    data=[]
    name="zz_de_admin.sms_api_log_0"
    data=get_table(conn,name)
    user,use_component=get_sms_api_log(data,0,user,item_list,wv_from_text)
    no_use_component=list(set(item_list)-set(use_component))
    '''
    #先不要这部分数据了，这部分数据对不上。先只用0的。
    for i in range(1,15):
        name="zz_de_admin.sms_api_log_"+str(i)
        data=get_table(conn,name)
        user=get_sms_api_log(data,i,user,item_list,wv_from_text)
    '''
    #获取订单元件表.先不获了
    # bs_data=get_table(conn,"zz_de_admin.business_order")
    # user=get_business_order(bs_data,user)
    name="zz_dc_cloud.sso_member"
    sso_member=get_table(conn,name)
    sso_member_id={}
    for da in sso_member:
        sso_member_id[da[1]]=da[10]
    #user是用户调用的物品id，item_list是所有item列表，component_id_list从dc_components里读取的物品信息，sso_member是用户
    return args,user,item_list,component_id_list,sso_member_id,conn,data,no_use_component
def deal_user_name(user_data,sso_member_id):#这个每次是否一样
    return list(set(list(user_data)+list(sso_member_id)))
    # return list(set(list(sso_member_id)))
def concat_model_data(user_data,user_recall,component_id_list,item_list,sso_member_id):
    data=[]
    data_info=[]
    user_feature=deal_user_name(user_data.keys(),sso_member_id.keys())#其实这里有问题
    call_list={}
    status_list=[]
    for user in user_data:
        for info in user_data[user]:#这有问题，这有大问题！！
            call_list[info[0]]=info[1:]
    for user in user_recall:
        item_used=[]
        for item in user_data[user]:
            item_used.append(item[0])
        for item in user_recall[user]:
            user_name=""
            if user in sso_member_id:
                user_name=sso_member_id[user]
            component_name=""
            description=""
            component_form=""
            if item not in component_id_list:
                continue
            if item in item_used:
                continue
            if item in component_id_list and component_id_list[item]["status"]!=21:##
                continue
            status_list.append(component_id_list[item]["status"])
            if item in component_id_list:
                component_name=component_id_list[item]["component_name"]
                description=component_id_list[item]["description"]
            local_time=time.localtime(time.time())
            recommend_time=time.strftime("%Y-%m-%d %H:%M:%S", local_time)
            merchant_id=component_id_list[item]["merchant_id"]
            component_form=component_id_list[item]["component_form"]
            #user_id, user_name,component_id,component_name,description,merchant_id,merchant_name,recommend_time,recommended_score
            new_info={"user":user,"user_name":user_name,"component_id":item,"component_name":component_name,"component_form":component_form,"description":description,"merchant_id":merchant_id,"merchant_name":"","recommended_source":"","recommended_time":recommend_time,"recommended_score":0}
            data_info.append(new_info)
            new_data=[user_feature.index(user)]
            if item in item_list:##这没必要啊
                new_data.extend(component_id_list[item]['vec'])
            else:
                new_data.extend(300*[0])
                # 这部分是删掉了一部分特征
            # if item in call_list:
                # new_data.extend(call_list[item])
            # else:
                # new_data.extend(600*[0])
                # if len(new_data)!=3011:
                    # print(user)
            data.append(new_data)     
    return data,data_info    
def preporcess_model_data(user_data,component_id_list,item_list,sso_member_id,ts):
    data=[]
    label=[]
    user_feature=deal_user_name(user_data.keys(),sso_member_id.keys())
    all_item_list=[]
    call_list={}
    for user in user_data:#call_list是干啥的
        for info in user_data[user]:
            all_item_list.append(info[0])
            call_list[info[0]]=info[1:]
    for user in user_data:
        deal_list=user_data[user]
        user_item=[]
        for info in deal_list:#给每一个user生成一个数值表示
            new_data=[user_feature.index(user)]
            item=info[0]
            user_item.append(item)
            new_data.extend(info[1:])
            # if item in item_list:
                # new_data.extend(component_id_list[item]['vec'])
                # new_data.extend(info[1:])
            # else:
                # new_data.extend(component_id_list[item])##不存在的值取平均值 0。这里要设置一个函数。还没写！！
                # new_data.extend(2410*[0])
                # new_data.extend(info[1:])
            data.append(new_data)
            label.append(1)
        #0怎么取？？？从其他调度+dc_component里的item[就从dc_component就行]
        data_lens=len(user_item)
        user_item=list(set(user_item))
        difference=list(set(all_item_list)-set(user_item))#不是item_list
        if data_lens<len(difference):
            neg_item=random.sample(difference,data_lens)
        else:
            neg_item=random.sample(difference,len(difference))
        for neg in neg_item:
            new_data=[user_feature.index(user)]   
            if neg in item_list:
                new_data.extend(component_id_list[item]['vec'])
                # new_data.extend(info[1:])
            else:
                new_data.extend(call_list[item])
            data.append(new_data)
            label.append(0)

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=ts, stratify=label, random_state=42)
    return X_train,Y_train,X_test,Y_test
def evaluate(y_true,y_pred,y_pred_proba):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    # 计算 ROC 曲线和 AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print("------evaluation-------")
    print("precision:",precision,"recall:",recall,"f1:",f1,"auc:",roc_auc)
if __name__ == "__main__": 
    # 数据处理
    args,user_data,item_data,component_id_list,sso_member_id,conn,system_log,no_use_component=get_info()
    recommend_dict=[]
    ##计算热度
    if HOT_CALCULATE==True:
        recommend_dict=hot_calculate(conn,component_id_list,system_log,RECOMMEND_MONTH,RECOMMEND_NUM)
    if args.modes=="output":#就是这里改
        ##做多路召回。一个是基于物品的协同过滤，一个是没使用过的，一个是热度榜
        ibcf=ItemBasedCF(user_data,item_data)#先测试这个准确性。没加特征  #21
        ibcf.preprocessData()
        ibcf.itemSimilarity()
        new_user_data={}
        for user in user_data:
            topN = ibcf.recommend(user, k=SIMILARITY_NUM, N=RECALL_NUM)  # 输出格式item的id和评分
            for data in topN:
                new_user_data.setdefault(user,[])
                new_user_data[user].append(data)
        recommend_list=[]
        for da in recommend_dict:
            recommend_list.append(da[0])
        for user in user_data:
            new_user_data[user].extend(recommend_list)
            new_user_data[user].extend(no_use_component)#多扩几个

        #千分之五的限制
       # for user in new_user_data:
        #    if len(new_user_data[user])<int(LIMIT_PERCENT*len(list(item_data))):
         #       del new_user_data[user]
        data,data_info=concat_model_data(user_data,new_user_data,component_id_list,item_data,sso_member_id)
        if args.choose_model=="decision":
            deal_output('model/decision_tree_model.pkl',data,data_info)
        elif args.choose_model=="lc":
            deal_output('model/logistic_regression_model.pkl',data,data_info)
    elif args.modes=="train":
        if args.choose_model=="decision":
            # 创建逻辑回归模型
            X_train,Y_train,X_test,Y_test=preporcess_model_data(user_data,component_id_list,item_data,sso_member_id,0.2)
            dtc = DecisionTreeClassifier()
            dtc.fit(X_train, Y_train)
            y_predict = dtc.predict(X_test)
            y_pred_proba=dtc.predict_proba(X_test)[:, 1]
            # evaluate(Y_test,y_predict,y_pred_proba)
            # print('predict: ', y_predict[:10])
            with open('model/decision_tree_model.pkl', 'wb') as model_file:
                pickle.dump(dtc, model_file)
        elif args.choose_model=="lc":
            X_train,Y_train,X_test,Y_test=preporcess_model_data(user_data,component_id_list,item_data,sso_member_id,0.2)
            lr_model = LogisticRegression()
            # 训练模型
            lr_model.fit(X_train, Y_train)
            # 预测测试集
            y_pred = lr_model.predict(X_test)
            y_pred_proba=lr_model.predict_proba(X_test)[:, 1]
            # evaluate(Y_test,y_pred,y_pred_proba)
            # print('predict: ', y_pred[:10])
            with open('model/logistic_regression_model.pkl', 'wb') as model_file:
                pickle.dump(lr_model, model_file)
    elif args.modes=="test":
    ##逻辑回归做推荐
        if args.choose_model=="lc":
            X_train,Y_train,X_test,Y_test=preporcess_model_data(user_data,component_id_list,item_data,0.2)
            with open('model/logistic_regression_model.pkl', 'rb') as file:
                dtc = pickle.load(file)
                y_predict = dtc.predict(X_test)
                y_pred_proba=dtc.predict_proba(X_test)[:, 1]
                evaluate(Y_test,y_predict,y_pred_proba)
        elif args.choose_model=="decision":
            X_train,Y_train,X_test,Y_test=preporcess_model_data(user_data,component_id_list,item_data,0.2)
            with open('model/decision_tree_model.pkl', 'rb') as file:
                dtc = pickle.load(file)
                y_predict = dtc.predict(X_test)
                y_pred_proba=dtc.predict_proba(X_test)[:, 1]
                evaluate(Y_test,y_predict,y_pred_proba)
    conn.close()        






