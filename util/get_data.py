import jieba
from gensim.models import Word2Vec,KeyedVectors
import numpy as np
#dc_components使用的特征（问题是，部分元件没有这些特征）：
    #元件身份认证：component_id[0]
    #需要nlp embedding的特征：1、component_name[1] 2、component_e_name[3] 3、description[4] 4、category_names[7]、5、root_category_name[9]、6、datasource[10]、7、data_provider[11]、8、service_description[30]、
    #直接使用的特征：9、component_type[12] 10、component_form[13]11、data_frequency[14] 12、status[21]
    #需要处理的特征：13、production_mode[14] 14、service_type[18]
def deal_production_mode(text):
    if text=="nonRealTime":
        return 0
    else:
        return 1
def deal_service_type(text):#利用one-hot编码
    num_classes=5#暂时不知道是几
    num=text.split(',')
    one_hot_vector = [0] * num_classes
    for i in num:
        one_hot_vector[int(i)]=1
    return one_hot_vector
def deal_text_embedding(text,wv_from_text):#获得句向量嵌入
    if text is None:
        seg_text=''
    else:
        seg_text = list(jieba.cut(text))
    # model=Word2Vec.load()
    word_embeddings=[wv_from_text[word] for word in seg_text if word in wv_from_text]
    if not word_embeddings:
        sentence_embedding = np.zeros(wv_from_text.vector_size)
    else:
    # 计算句子的嵌入向量（取平均）
        sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding
def get_dc_components(dc_data,wv_from_text)->dict:
    component_id_list={}
    for u in dc_data:
        new_data=[]
        #nlp embedding特征
        nlp_list=[1,3,4,7,9,10,11,30]
        for i in nlp_list:
            new_data.extend(deal_text_embedding(u[i],wv_from_text))
        #直接使用的特征
        new_data.extend([int(u[12]),int(u[13]),int(u[15]),int(u[21])])
        #处理特征
        new_data.append(deal_production_mode(u[14]))
        new_data.extend(deal_service_type(u[18]))
        component_id_list[u[0]]={}
        component_id_list[u[0]]["vec"]=new_data#这里有问题
        component_id_list[u[0]]["component_name"]=u[1]
        component_id_list[u[0]]["description"]=u[4]
    return component_id_list
def deal_id(data):
    return list(set(data))
#sms_api_log使用的特征：
    #用户身份认证：user_id[1]
    #元件调用：component_id[3]
    #使用特征（优先用这个）：
        #nlp特征：component_name[3]、merchant_name[5]、这两个重复了啊
        #直接使用的特征(先不用这个了，不然不知道咋整)：out_data_size[11] result[12] identification_status[21]
def get_sms_api_log(data,ways,user,item_list,wv_from_text)->dict:
    a=0
    for da in data:
        user_id=da[1]
        if ways==0:
            component_id=da[2]
        else:
            component_id=da[3]
        if user_id not in user:
            user[user_id]=[]
            # user[user_id]=[]
        new_data=[component_id]
        if component_id in item_list:
            a+=1
        #处理nlp特征
        if ways==0:
            new_data.extend(deal_text_embedding(da[4],wv_from_text))
            new_data.extend(deal_text_embedding(da[6],wv_from_text))
        else :
            new_data.extend(deal_text_embedding(da[3],wv_from_text))
            new_data.extend(deal_text_embedding(da[5],wv_from_text))
        # new_data.append(da[11])
        # new_data.append(da[12])
        # new_data.append(da[21])
        #元件id、其他特征
        user[user_id].append(new_data)
    return user
#先不管这个了
def get_business_order(bs_data,user)->dict:
    buyer_name_list=[]
    merchant_id_list=[]
    for da in bs_data:
        user_id=da[3]
        buyer_name=da[4]
        merchant_id=da[5]
        if user_id in user:
            if buyer_name!=None:
                user[user_id].setdefault("buyer_name",buyer_name)
            if merchant_id!=None:
                user[user_id].setdefault("merchant_id",merchant_id)
            buyer_name_list.append(buyer_name)
            merchant_id_list.append(merchant_id)
    buyer_name_list=deal_id(buyer_name_list)
    merchant_id_list=deal_id(merchant_id_list)
    for user_id in user:
        if "buyer_name" in user[user_id]:
            user[user_id]["buyer_name"]=buyer_name_list.index(user[user_id]["buyer_name"])
        if "merchant_id" in user[user_id]:
            user[user_id]["merchant_id"]=merchant_id_list.index(user[user_id]["merchant_id"])
    return user
     
