"""
pip install streamlit
Run by: `streamlit run server.py`
"""

# import lib
import streamlit as st
import json
from utils import segment_documents
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import pipeline
import pandas as pd
import torch
import time

st.set_page_config(
    page_title="Question Answering System",
    page_icon="❓",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'HCMUS Mathematic and Computer Science seminar report'
    },
)


@st.cache(allow_output_mutation=True)
def load_data(dataset='train_reg'):
    data_file = open('data/'+dataset+'.json', 'r', encoding='utf-8-sig')
    data = json.load(data_file)['data']
    paragraphs = []
    questions = []
    for row in data:
        for paragraph in row['paragraphs']:
            paragraphs.append(paragraph['context'])
            for qa in paragraph['qas']:
                questions.append((qa['question'], qa['answers'][0]['text']))
    corpus = segment_documents(paragraphs)
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return corpus, questions, bm25


def get_bm25_score(query, corpus, bm25):
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores


@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("nguyenvulebinh/vi-mrc-large")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "nguyenvulebinh/vi-mrc-large")
    have_gpu = torch.cuda.is_available()
    qa_model = pipeline("question-answering", model=model,
                        tokenizer=tokenizer, device=0 if have_gpu else -1)
    return qa_model


st.sidebar.title("Question Answering System")
question = st.sidebar.text_input("Question", placeholder="Nhập câu hỏi")
st.sidebar.markdown("""
Example:
- thủ đô nước Việt Nam tên là gì?
- đâu là chiến thắng quân sự lớn nhất trong cuộc kháng chiến chống Pháp?
"""
                    )
datasets = [
    {
        "name": "train_reg",
        "w0": 0.064,
        "w1": 0.936
    },
    {
        "name": "train_wiki",
        "w0": 0.002,
        "w1": 0.998
    },
   {
        "name": "train_translated_squad_25",
        "w0": 0.387,
        "w1":0.613
    },
]

dataset = st.sidebar.selectbox(
    "Dataset", datasets, format_func=lambda x: x['name'])
corpus, questions, bm25 = load_data(dataset['name'])
top_k = st.sidebar.slider("Top K documents", 1, 30, 10)
w0 = st.sidebar.slider("Retrieval score weight", 0., 1.,
                       dataset['w0'], 0.001, format="%.3f")
w1 = st.sidebar.slider("Predict score weight", 0., 1.,
                       dataset['w1'], 0.001, format="%.3f")
qa_model = load_model()


if question:
    st.title(question)
    start_time = time.time()
    loading = st.text("Đang tìm câu trả lời...")
    bar = st.progress(0)

    doc_scores = get_bm25_score(question, corpus, bm25)
    top_k_idx = doc_scores.argsort()[-top_k:][::-1]

    results = []
    for i, idx in enumerate(top_k_idx):
        bar.progress((i+1)/top_k)
        retrieval_score = doc_scores[idx]
        result = qa_model(question=question, context=corpus[idx], topk=5)
        for r in result:
            r['predict_score'] = r['score']
            del r['score']
        data = {
            "context": corpus[idx],
            "answers": result,
            "retrieval_score": retrieval_score,
            "score": w0*retrieval_score + w1*result[0]['predict_score']
        }
        results.append(data)

    results = sorted(
        results, key=lambda x: x['score'], reverse=True)

    tab1, tab2 = st.tabs(["Result", "Top K documents"])
    with tab1:
        for i, result in enumerate(results):
            answer_start = result['answers'][0]['start']
            answer_end = result['answers'][0]['end']
            context = result['context']
            context = context[:answer_start] + '`' + \
                context[answer_start:answer_end] + '`' + context[answer_end:]
            col1, col2 = st.columns(2)
            col1.header("Context")
            col1.write(context)
            col2.header("Answer")
            col2.write('Score: ' + str(result['score']))
            col2.write('Retrieval score: ' + str(result['retrieval_score']))
            col2.write('Best answer: ' + result['answers'][0]['answer'])
            df = pd.DataFrame(result['answers'])
            # show full number
            df = df.style.format({'start': '{:,.0f}', 'end': '{:,.0f}',
                                  'predict_score': '{:,.15f}'})
            col2.dataframe(df, use_container_width=True)
    with tab2:
        for i, idx in enumerate(top_k_idx):
            st.write('Document ' + str(i+1))
            st.write(corpus[idx])
            st.write('Score: ' + str(doc_scores[idx]))

    end_time = time.time()
    loading.text("Run time: " +
                 str(round(end_time-start_time, 2)) + "s")
    bar.empty()
