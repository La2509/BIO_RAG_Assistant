import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Bio-Research Assistant", page_icon="🧬")

# --- BARRA LATERAL (Setup) ---
with st.sidebar:
    st.header("⚙️ Configuração")
    st.markdown("Este assistente usa *RAG* para ler papers científicos.")
    
    # Tenta pegar a chave automática do arquivo de segredos
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("Chave de API carregada! ✅")
    else:
        # Se não tiver arquivo de segredos, pede manual
        api_key = st.text_input("Cole sua OpenAI API Key", type="password")

# --- FUNÇÕES CIENTÍFICAS ---

def get_pdf_text(pdf_docs):
    """Lê o PDF bruto e extrai o texto."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Quebra o texto em pedaços (Chunks) para a IA processar."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    """Transforma texto em números (Vetores) para busca semântica."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    """Configura o cérebro da IA para responder baseada no contexto."""
    llm = ChatOpenAI(temperature=0.3, openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# --- INTERFACE (FRONTEND) ---

st.title("🧬 Bio-RAG: Leitor de Artigos Científicos")
st.markdown("Faça upload de um artigo (PDF) e faça perguntas sobre a metodologia, resultados ou conclusões.")

# 1. Upload
pdf_docs = st.file_uploader("Carregue seu PDF aqui", accept_multiple_files=True)

# Botão de Processamento
if st.button("Processar Artigo 🧪"):
    if not api_key:
        st.error("Por favor, insira a chave da API na barra lateral.")
    elif not pdf_docs:
        st.warning("Carregue um arquivo PDF primeiro.")
    else:
        with st.spinner("Analisando e vetorizando o conteúdo..."):
            # Extração e Vetorização
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vector_store(text_chunks, api_key)
            
            # Salva na memória da sessão
            st.session_state.vectorstore = vectorstore
            st.session_state.conversation = True
            st.success("Processamento concluído! Pode perguntar.")

# 2. Área de Perguntas
if "conversation" in st.session_state:
    st.divider()
    user_question = st.text_input("Pergunte ao artigo (Ex: Qual foi a conclusão principal?)")
    
    if user_question:
        with st.spinner("Consultando a literatura..."):
            # Busca os trechos relevantes
            docs = st.session_state.vectorstore.similarity_search(user_question)
            
            # Gera a resposta
            chain = get_conversation_chain(st.session_state.vectorstore, api_key)
            response = chain.run(input_documents=docs, question=user_question)
            
            st.write("### 🤖 Resposta:")
            st.info(response)