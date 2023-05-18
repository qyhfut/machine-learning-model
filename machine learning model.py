import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import lightgbm as gbm
from sklearn.neural_network import MLPClassifier


data = pd.read_csv(r"ldampnet_vector3.csv")



# 模型输入
# X = data.iloc[:,0:10] #LDA特征

# X = data.iloc[:,10:778] #MPNet特征

X = data.iloc[:,778:781] #sentiment特征


# X = data.iloc[:,0:778] #LDA-MPNet特征


# X = data.iloc[:,0:781] #LDA-MPNet-sentiment特征


y = data.iloc[:,-1]
print(X)


'''
数据集划分
'''
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=10)
# X_train_vali, X_test, y_train_vali, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

'''
不均匀采样
'''
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=10)#random_statei相当于随机数种子的作用
X_train, y_train = smt.fit_resample(X_train, y_train)
X_train = pd.DataFrame(X_train, columns=list(X_train.columns))
print("nums of train/test set: ", len(X_train), len(X_test), len(y_train), len(y_test))


'''
训练模型
'''

# print('--- KNN model ---')
# classifier=KNeighborsClassifier() # ovr:一对多策略C=2,kernel='rbf',gamma=10,decision_function_shape='ovr'
# classifier.fit(X_train, y_train) #ravel函数在降维时默认是行序优先
# preds = classifier.predict(X_test)
# y_preds = classifier.predict_proba(X_test)[:,1]  ###预测概率



# print('--- LogisticRegression model ---')
# clf = LogisticRegression(random_state=10)
# clf.fit(X_train, y_train)
# preds = clf.predict(X_test)
# y_preds = clf.predict_proba(X_test)[:,1]  ###预测概率




# print('--- NB model ---')
# gnb = GaussianNB()
# preds = gnb.fit(X_train, y_train).predict(X_test)
# y_preds = gnb.fit(X_train, y_train).predict_proba(X_test)[:,1]



# print('--- SVM model ---')
# classifier=svm.SVC(random_state=10, probability=True) # ovr:一对多策略C=2,kernel='rbf',gamma=10,decision_function_shape='ovr'
# classifier.fit(X_train, y_train) #ravel函数在降维时默认是行序优先
# preds = classifier.predict(X_test)
# y_preds = classifier.predict_proba(X_test)[:,1]  ###预测概率



# print('--- Decision Tree model ---')
# dt = tree.DecisionTreeClassifier(random_state=10)
# dt.fit(X_train, y_train)
# preds = dt.predict(X_test)
# y_preds = dt.predict_proba(X_test)[:,1]  ###预测概率



# print('--- Random-forest model ---')
# forest = RandomForestClassifier(random_state=10)
# forest.fit(X_train, y_train)
# preds = forest.predict(X_test)
# y_preds = forest.predict_proba(X_test)[:,1]  ###预测概率



# print('--- GBDT model ---')
# gbdt = GradientBoostingClassifier(random_state=10)
# gbdt.fit(X_train, y_train)
# preds = gbdt.predict(X_test)
# y_preds = gbdt.predict_proba(X_test)[:,1]  ###预测概率



print('--- AdaBoost model ---')
adaboost = AdaBoostClassifier(random_state=10)
adaboost.fit(X_train, y_train)
preds = adaboost.predict(X_test)
y_preds = adaboost.predict_proba(X_test)[:,1]  ###预测概率



# print('--- XGBoost model ---')
# model = xgb.XGBClassifier(random_state=10)
# model.fit(X_train, y_train)
# preds= model.predict(X_test)
# y_preds = model.predict_proba(X_test)[:,1]  ###预测概率



# print('--- lightgbm model ---')
# lgbm = gbm.LGBMClassifier(random_state=10)
# lgbm.fit(X_train, y_train)
# preds = lgbm.predict(X_test)
# y_preds = lgbm.predict_proba(X_test)[:,1]  ###预测概率



# y_preds=np.array(y_preds)
# y_test=np.array(y_test)
# for i in range(preds.shape[0]):
#     if round(y_preds[i])!=y_test[i]:
#         print(y_test[i],y_preds[i])


#模型性能相关指标
acc = accuracy_score(y_test, preds)
precision = precision_score(y_test,preds)
recall = recall_score(y_test,preds)
f1 = f1_score(y_test, preds)
auc1 = roc_auc_score(y_test, y_preds)
print("Accuracy: ", acc)
print("Precision:",precision)
print("Recall:",recall)
print("F1 score: ", f1)
print("AUC: ", auc1)


#ROC绘制
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_test, y_preds) #计算真正率和假正率
roc_auc =auc(fpr, tpr) #计算auc的值


# plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show()


# fpr2 = np.array(fpr)
# test1 = pd.DataFrame(fpr2)
# test1.to_csv("./output/fpr_adaboost_all.csv")
#
# tpr2 = np.array(tpr)
# test2 = pd.DataFrame(tpr2)
# test2.to_csv("./output/tpr_adboost_all.csv")