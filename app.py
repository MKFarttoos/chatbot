
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
import re
import torch
import wikipedia
import pandas as pd
import numpy as np
import base64
import io
import zipfile
import gdown
import faiss
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# روابط Google Drive للتحميل
SENTENCE_MODEL_ID = "1OwngVgezpWbYzW3Rtw-aoV1CfTS5iMOU"
BERT_MODEL_ID = "1jW75lsnhDeXuFabwhj-d9VDOpLVV2ADK"
DATASET_ID = "1_CiUUzDookPuyAF1_qa3j-UWMOCkdFfS"

# تحميل وفك الضغط
def download_and_extract(file_id, output_dir, zip_name):
    zip_path = os.path.join(output_dir, zip_name)
    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

# تحميل الملفات
download_and_extract(SENTENCE_MODEL_ID, ".", "sentence_model.zip")
download_and_extract(BERT_MODEL_ID, ".", "bert_arabic_qa_model.zip")
if not os.path.exists("Merged_Arabic_QA_Dataset_Final.csv"):
    gdown.download(f"https://drive.google.com/uc?id={DATASET_ID}", "Merged_Arabic_QA_Dataset_Final.csv", quiet=False)

# إعداد ويكيبيديا
wikipedia.set_lang("ar")

# تحميل البيانات
df = pd.read_csv("Merged_Arabic_QA_Dataset_Final.csv", low_memory=False)
df = df.dropna(subset=['question', 'answer', 'category'])

# TF-IDF
vectorizer = TfidfVectorizer().fit(df['question'])

# FAISS
semantic_model = SentenceTransformer("sentence_model")
question_embeddings = semantic_model.encode(df['question'].tolist(), convert_to_numpy=True)
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# BERT
model_path = "bert_arabic_qa_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)

# Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Arabic QA Chatbot"
qa_log = []

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("🤖 بوت عربي للأسئلة والأجوبة", className="text-center mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("❓ السؤال:"),
            dcc.Input(id="question", type="text", placeholder="اكتب سؤالك هنا", style={"width": "100%"}),
            html.Br(), html.Br(),
            dbc.Button("🔍 استخرج الجواب", id="submit", color="primary", style={"width": "100%"}),
            html.Br(), html.Br(),
            html.Div(id="output", style={"whiteSpace": "pre-wrap", "fontWeight": "bold", "color": "#1d3557"}),
            html.Br(),
            html.Audio(id='audio', controls=True, src=''),
            html.Hr(),
            html.H5("📜 سجل الأسئلة السابقة:"),
            html.Ul(id="qa_history")
        ], width=12)
    ])
], fluid=True)

def normalize_question(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"ماهو|ايش|وش", "ما هو", text)
    text = text.replace("؟", "")
    return text.strip()

def text_to_audio(text):
    tts = gTTS(text, lang="ar")
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/mpeg;base64,{b64}"

@app.callback(
    [Output("output", "children"),
     Output("audio", "src"),
     Output("qa_history", "children")],
    [Input("submit", "n_clicks")],
    [State("question", "value")]
)
def answer(n_clicks, question):
    if not n_clicks or not question:
        return "❗ الرجاء إدخال السؤال.", "", []

    question = normalize_question(question)
    q_vec = vectorizer.transform([question])
    similarities = cosine_similarity(q_vec, vectorizer.transform(df['question']))[0]
    top_idx = similarities.argmax()
    if similarities[top_idx] >= 0.75:
        matched_answer = df.iloc[top_idx]['answer']
        matched_category = df.iloc[top_idx]['category']
        msg = f"✅ الجواب (TF-IDF - تصنيف: {matched_category}): {matched_answer}"
        qa_log.append(f"❓ {question} → ✅ {matched_answer} (TF-IDF)")
        return msg, text_to_audio(matched_answer), [html.Li(x) for x in qa_log]

    query_vec = semantic_model.encode([question], convert_to_numpy=True)
    D, I = index.search(query_vec, k=1)
    best_score = 1 - (D[0][0] / 4)
    best_idx = I[0][0]
    if best_score >= 0.75:
        matched_answer = df.iloc[best_idx]['answer']
        matched_category = df.iloc[best_idx]['category']
        msg = f"✅ الجواب (FAISS - تصنيف: {matched_category}): {matched_answer}"
        qa_log.append(f"❓ {question} → ✅ {matched_answer} (FAISS)")
        return msg, text_to_audio(matched_answer), [html.Li(x) for x in qa_log]

    try:
        search_results = wikipedia.search(question)
        if not search_results:
            msg = "❌ عذرًا، لم أتمكن من العثور على إجابة مناسبة لهذا السؤال."
            qa_log.append(f"❓ {question} → {msg}")
            return msg, "", [html.Li(x) for x in qa_log]

        page = wikipedia.page(search_results[0])
        context = page.content[:1500]
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        answer_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))
        if not answer_text.strip():
            msg = "❌ عذرًا، لم أتمكن من العثور على إجابة مناسبة لهذا السؤال."
            qa_log.append(f"❓ {question} → {msg}")
            return msg, "", [html.Li(x) for x in qa_log]
        msg = f"📚 المصدر: ويكيبيديا\n✅ الجواب: {answer_text.strip()}"
        qa_log.append(f"❓ {question} → ✅ {answer_text.strip()} (ويكيبيديا)")
        return msg, text_to_audio(answer_text.strip()), [html.Li(x) for x in qa_log]
    except:
        msg = "❌ عذرًا، لم أتمكن من العثور على إجابة مناسبة لهذا السؤال."
        qa_log.append(f"❓ {question} → {msg}")
        return msg, "", [html.Li(x) for x in qa_log]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
