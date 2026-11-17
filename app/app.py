import time
import streamlit as st
import pandas as pd
from transformers import pipeline

# Configuração da página
st.set_page_config(
    page_title="Classificador de Reviews PT-BR | BERTimbau",
    page_icon="📊",
    layout="centered"
)

MODEL_ID = "GetulioCO/bertimbau-ptbr-review-classifier"

@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        top_k=None
    )

st.title("Classificador de Avaliações em PT-BR")
st.markdown(
    f"Inferência em tempo real utilizando **BERTimbau Fine-Tuned** "
    f"([`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})) para triagem automatizada de feedback de clientes."
)

classifier = load_classifier()

# Exemplos rápidos estilo delivery / e-commerce
st.subheader("Testar com exemplos prontos")
col1, col2, col3 = st.columns(3)

exemplos = {
    "positivo": "Entrega super rápida! O lanche chegou quentinho, bem embalado e exatamente como pedi.",
    "neutro": "A comida estava gostosa, mas demorou quase 40 minutos além do previsto para chegar.",
    "negativo": "Pedido veio completamente errado, embalagem violada e o suporte não resolveu meu problema."
}

if "texto_input" not in st.session_state:
    st.session_state["texto_input"] = exemplos["positivo"]

if col1.button("Exemplo Positivo", use_container_width=True):
    st.session_state["texto_input"] = exemplos["positivo"]
if col2.button("Exemplo Neutro", use_container_width=True):
    st.session_state["texto_input"] = exemplos["neutro"]
if col3.button("Exemplo Negativo", use_container_width=True):
    st.session_state["texto_input"] = exemplos["negativo"]

user_text = st.text_area(
    "Digite ou edite uma avaliação de cliente:",
    value=st.session_state["texto_input"],
    height=110
)

if st.button("Analisar Sentimento", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Por favor, insira um texto para análise.")
    else:
        start_time = time.perf_counter()
        raw_outputs = classifier(user_text)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Ordena por maior probabilidade
        sorted_outputs = sorted(raw_outputs, key=lambda x: x["score"], reverse=True)
        top_label = sorted_outputs[0]["label"]
        top_score = sorted_outputs[0]["score"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predição Principal", top_label)
        m2.metric("Confiança", f"{top_score * 100:.1f}%")
        m3.metric("Latência (CPU)", f"{latency_ms:.0f} ms")

        st.subheader("Distribuição de Probabilidades")
        for item in sorted_outputs:
            label = item["label"]
            score = item["score"]
            st.write(f"**{label}** — `{score * 100:.2f}%`")
            st.progress(float(score))