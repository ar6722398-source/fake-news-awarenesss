import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

p=argparse.ArgumentParser(); p.add_argument('--data',default='data/news.csv'); p.add_argument('--text-col',default='text'); p.add_argument('--label-col',default='label'); a=p.parse_args()
df=pd.read_csv(a.data).dropna(subset=[a.text_col,a.label_col])
df[a.label_col]=df[a.label_col].astype(str).str.upper().str.strip()
if len(df)<10: raise ValueError('Dataset is too small. Add a real labelled dataset before training.')
if df[a.label_col].nunique()<2: raise ValueError('Dataset must contain at least two labels, e.g. REAL and FAKE.')
Xtr,Xte,ytr,yte=train_test_split(df[a.text_col].astype(str),df[a.label_col],test_size=.2,random_state=42,stratify=df[a.label_col])
model=Pipeline([('tfidf',TfidfVectorizer(stop_words='english',max_df=.9,min_df=1,ngram_range=(1,2),max_features=50000)),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))])
model.fit(Xtr,ytr); pred=model.predict(Xte); joblib.dump(model,'fake_news_pipeline.joblib')
report=f"Rows: {len(df)}\nTrain rows: {len(Xtr)}\nTest rows: {len(Xte)}\nAccuracy: {accuracy_score(yte,pred):.4f}\n\nClassification report:\n{classification_report(yte,pred)}\nConfusion matrix:\n{confusion_matrix(yte,pred)}\n"
open('model_metrics.txt','w').write(report); print(report)
