import pandas as pd
items = pd.read_csv('market_basket.txt' ,sep="\t")



itemsets = items.groupby(['ID'])\
.agg({'Product': lambda x: x.ravel().tolist()}).reset_index()
print(itemsets)

print("q2")
print(items.head(10))

print(items.shape)
print(itemsets.shape)

list = items.groupby('Product').agg(lambda x: ','.join(x))
products = []
for name in list.index:
    products.append(name)

import numpy as np
binaryTable=np.zeros((len(itemsets.ID),len(products)))
ligne=0
for i in itemsets.Product:
    #fixer la colonne
    colonne=0
    for j in products:
        if(j in i) :
            binaryTable[ligne,colonne] = 1
        else :
            binaryTable[ligne,colonne] = 0
        colonne = colonne+1
    ligne=ligne+1
df = pd.DataFrame(binaryTable, index=itemsets.ID,columns=products)
print(df)

print("q5")
print(pd.crosstab(index = items.ID , columns=items.Product))
print("q6")
print(itemsets.head(30))

products[:3]

def load_data_set():
    list = items.groupby('ID').agg(lambda x: ','.join(x))
    transactions = list.values.tolist()
    dataset=[]
    for j in transactions :
        for k in j:
            k=k.split(',')
            dataset.append(k)
    return dataset

def create_C1(data_set):
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1

def is_apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True

def create_Ck(Lksub1, k):
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 =  [*Lksub1, ]
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = [*list_Lksub1[i],]
            l2 = [*list_Lksub1[j],]
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck

def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk

def generate_L(data_set, k, min_support):
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data

def generate_assoc_rules(L, support_data, min_conf):
    assoc_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in assoc_rule_list:
                        assoc_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return assoc_rule_list

data_set = load_data_set()
L, support_data = generate_L(data_set, k=4, min_support=0.025)
for Lk in L:
    print ("="*50)
    print ("frequent " + str(len([*Lk,][0])) + "-itemsets(frequent_itemsets)\t\tsupport(support)")
    print ("="*50)
    for freq_set in Lk:
        print (freq_set, support_data[freq_set])

big_rules_list = generate_assoc_rules(L, support_data, min_conf=0.7)
print("Les associations fortes")
for item in big_rules_list:
    print(item[0], "=>", item[1], "conf(Confiance): ", item[2])

data = []
allItemsets = []

for i in L:
    for j in i:
        data.append(j)
        if (len(data) == 15):
            break
    if (len(data) == 15):
        break
print("print L")
for i in data:
    print(i, "\n")

freq_items=[]
for i in L :
        for j in i :
            freq_items.append(set(j))

def is_inclus(x,items):
  return items.issubset(x)


x = {'Aspirin'}
print("start q10")
print(freq_items)
for itemset in freq_items:
  if(is_inclus(itemset , x)):
    print(itemset)
print("end q10")

y = {'Aspirin' , 'Eggs'}
print(freq_items)
print("start q11")
for itemset in freq_items:
  if(is_inclus(itemset , y)):
    print(itemset)
print("end q11")

from apyori import apriori

dataset = pd.read_csv('market_basket.txt' ,sep="\t")

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_array = te.fit(itemsets.Product).transform(itemsets.Product)
dd = pd.DataFrame(te_array , columns=te.columns_)
print(dd)

print("start q12")
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(dd , min_support=0.025 ,max_len=4, use_colnames = True)
print(frequent_itemsets)

from mlxtend.frequent_patterns import association_rules
res = association_rules(frequent_itemsets , metric="confidence" , min_threshold =0.75 )
print(res)
print("end q12")
print("start q13")
print(res.head())
print("end q13")

res1 = res[res['lift'] > 7]
print("start q14")
print(res1)
print("end q14")

print("start q15")
print(res1[res1['consequents'] == {'2pct_Milk'}])
print("end q15")



