import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('train.csv')

X = df_train[['discourse_id', 'essay_id', 'discourse_text', 'discourse_type']]
y = df_train['discourse_effectiveness']

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25)

X_train['discourse_effectiveness'] = y_train
X_val['discourse_effectiveness'] = y_val

X_train.to_csv('./data_stratified/train_stratified.csv')
X_val.to_csv('./data_stratified/validation_stratified.csv')