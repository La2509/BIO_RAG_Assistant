# 🧬 Bio-RAG: Assistente de Pesquisa com IA

> **Uma ferramenta de Leitura Científica Acelerada usando Retrieval-Augmented Generation (RAG).**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-orange)

## 🎯 Sobre o Projeto

O **Bio-RAG** é uma aplicação desenvolvida para otimizar o tempo de pesquisadores e cientistas. Unindo meu background acadêmico (**PhD em Bioquímica**) com minha transição para **Engenharia da Computação**, criei esta ferramenta que permite a análise rápida de artigos científicos (papers) em formato PDF.

O sistema utiliza a técnica de **RAG (Retrieval-Augmented Generation)** para "ler" o PDF e permitir que o usuário faça perguntas específicas sobre metodologias, resultados e conclusões, recebendo respostas baseadas estritamente no conteúdo do arquivo enviado.

## 🚀 Funcionalidades

- 📄 **Upload de PDFs:** Suporte para carregamento de artigos científicos completos.
- 🔍 **Extração Inteligente:** Processamento de texto usando `PyPDF2` e fragmentação com `LangChain`.
- 🤖 **Q&A Interativo:** Chatbot capaz de responder perguntas contextuais sobre o artigo.
- 🧠 **Memória de Contexto:** Utiliza vetores (Embeddings) para encontrar os trechos mais relevantes do texto antes de responder.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Interface (Frontend):** Streamlit
- **Orquestração de LLM:** LangChain (Community & Core)
- **Modelos de IA:** OpenAI (GPT-3.5-turbo)
- **Banco Vetorial:** FAISS (Facebook AI Similarity Search)

## ⚙️ Como Executar o Projeto

Pré-requisitos: Python instalado e uma chave de API da OpenAI.

```bash
# 1. Clone o repositório
git clone [https://github.com/La2509/Bio-RAG-Assistant.git](https://github.com/SEU-USUARIO/Bio-RAG-Assistant.git)

# 2. Entre na pasta do projeto
cd Bio-RAG-Assistant

# 3. Crie um ambiente virtual (Opcional, mas recomendado)
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Configure suas credenciais
# Crie um arquivo .secrets.toml na pasta .streamlit ou insira a chave na interface.

# 6. Execute a aplicação
python -m streamlit run bio_rag.py