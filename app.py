import sqlite3
from pathlib import Path
import joblib
import streamlit as st
DB='fake_news.db'; MODEL='fake_news_pipeline.joblib'

def db_init():
    with sqlite3.connect(DB) as c:
        c.execute('CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, news_text TEXT NOT NULL, prediction TEXT NOT NULL, confidence REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')

def save(text,label,confidence):
    with sqlite3.connect(DB) as c:
        c.execute('INSERT INTO predictions(news_text,prediction,confidence) VALUES(?,?,?)',(text,label,confidence))

def load_model():
    if not Path(MODEL).exists(): return None
    return joblib.load(MODEL)

def history():
    with sqlite3.connect(DB) as c:
        return c.execute('SELECT created_at,prediction,confidence,substr(news_text,1,120) FROM predictions ORDER BY id DESC LIMIT 50').fetchall()

st.set_page_config(page_title='Fake News Awareness', page_icon='📰', layout='wide')
db_init(); model=load_model()
st.title('📰 Fake News Awareness & Detection System')
st.caption('Mini project: educational ML prototype. A prediction is not proof that a claim is true or false.')
page=st.sidebar.radio('Navigate',['Check News','Awareness','History','Model Info'])

if page=='Check News':
    st.subheader('Check news text')
    text=st.text_area('Paste a headline or news article',height=220,placeholder='Enter news text here...')
    if st.button('Analyse',type='primary'):
        if not text.strip(): st.warning('Please enter some news text.')
        elif model is None: st.error('Model file not found. Run: python train_model.py --data data/news.csv')
        else:
            pred=str(model.predict([text])[0])
            confidence=None
            if hasattr(model,'predict_proba'):
                probs=model.predict_proba([text])[0]; confidence=float(max(probs))*100
            if pred=='REAL': st.success(f'Prediction: REAL' + (f' ({confidence:.1f}% model confidence)' if confidence else ''))
            else: st.error(f'Prediction: POTENTIALLY FAKE' + (f' ({confidence:.1f}% model confidence)' if confidence else ''))
            save(text,pred,confidence)
            st.info('Verify important claims with the original source and multiple reputable sources. ML can be wrong, especially on unfamiliar topics.')
elif page=='Awareness':
    st.subheader('Fake-news awareness checklist')
    st.markdown('''1. **Check the original source** and whether the page is authentic.\n2. **Read beyond the headline**; headlines can omit context.\n3. **Check author, date and evidence**, including links to primary material.\n4. **Cross-check the same claim** with multiple reputable outlets or official sources.\n5. **Inspect images and quotes in context**; old media can be reused with a new claim.\n6. **Do not share solely because the content matches your expectations.**''')
elif page=='History':
    st.subheader('Recent local predictions')
    rows=history()
    if rows: st.dataframe(rows,use_container_width=True,column_config={0:'Time',1:'Prediction',2:'Confidence %',3:'Text preview'})
    else: st.info('No predictions stored yet.')
else:
    st.subheader('Model information')
    st.markdown('This project uses **TF-IDF text features + Logistic Regression** in a Scikit-learn pipeline. Training and evaluation are performed by `train_model.py`.')
    if Path('model_metrics.txt').exists(): st.code(Path('model_metrics.txt').read_text())
