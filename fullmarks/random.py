import pickle

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

from config import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# todo:1.加载数据
data = pd.read_csv(config.data.train_pre_path)
# 加载特征
words = data['words']
# 加载标签
label = data['label']
# todo: 特征工程
# 特征提取器实例化
stop_words = open(config.data.stopwords_path, encoding='utf-8').read().split()
transforms = TfidfVectorizer(stop_words=stop_words)
features = transforms.fit_transform(words)
# todo:2.处理数据
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=52)
# todo:3.模型训练
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
# 交叉验证网格搜索自动调参
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(x_train, y_train)
grid_search.best_score_
grid_search.best_params_
best_estimator = grid_search.best_estimator_
# for _ in tqdm(range(1), '训练中'):
#     model.fit(x_train, y_train)
# todo:4.模型预测
y_pred = best_estimator.predict(x_test)
# todo:5.模型评估
print(f'acc:->{accuracy_score(y_true=y_test, y_pred=y_pred)}')
print(f"precision:->{precision_score(y_true=y_test, y_pred=y_pred, average='micro')}")
print(f"recall:->{recall_score(y_true=y_test, y_pred=y_pred, average='micro')}")
print(f"fl:->{f1_score(y_true=y_test, y_pred=y_pred, average='micro')}")
print(classification_report(y_true=y_test, y_pred=y_pred))

# todo:6.保存模型
with open(config.data.random_model_save_path, 'wb') as f:
    pickle.dump(model, f)

with open(config.data.tf_save_path, 'wb') as f:
    pickle.dump(features, f)
