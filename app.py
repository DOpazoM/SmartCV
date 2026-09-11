import streamlit as st
import fitz
import pickle
import torch

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "DOpazo38/smartcv-model"

@st.cache_resource
def cargar_modelo():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    archivo = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="codificador_de_etiquetas.pkl"
    )

    with open(archivo, "rb") as f:
        le = pickle.load(f)

    return tokenizer, model, le


tokenizer, model, le = cargar_modelo()


def leer_pdf(pdf_file):

    texto = ""

    doc = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    for pagina in doc:
        texto += pagina.get_text()

    return texto


def clasificar(texto):

    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    top = torch.topk(probs, 5)

    resultados = []

    for score, idx in zip(
        top.values[0],
        top.indices[0]
    ):

        resultados.append({
            "Cargo":
            le.inverse_transform([idx.item()])[0],
            "Probabilidad":
            f"{score.item()*100:.2f}%"
        })

    return resultados


st.title("SmartCV Wood Industry")

pdf = st.file_uploader(
    "Suba un CV PDF",
    type=["pdf"]
)

if pdf:

    texto = leer_pdf(pdf)

    ranking = clasificar(texto)

    st.subheader("Ranking de cargos")
    st.json(ranking)

    st.subheader("Texto extraído")
    st.text_area(
        "",
        texto,
        height=300
    )
