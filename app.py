# app.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import chromadb
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.exceptions import LoginError
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from app_config import AVAILABLE_MODELS, DEFAULT_MODEL_NAME, UNIFIED_SYSTEM_INSTRUCTION, QUESTIONS

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Document Chat", layout="wide",
    initial_sidebar_state="expanded", page_icon="assets/zensys.png"
)

# --- CONFIGURATION (from st.secrets) ---
PERSIST_DIR = st.secrets["PERSIST_DIR"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
CHAT_HISTORY_DIR = "chat_history"


def get_chat_filepath(username):
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    safe_username = "".join(c for c in username if c.isalnum())
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_username}_chat.json")

def save_chat_history(username, messages):
    filepath = get_chat_filepath(username)
    with open(filepath, 'w') as f: json.dump(messages, f, indent=2)

def load_chat_history(username):
    filepath = get_chat_filepath(username)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return []
    return []

# --- AUTHENTICATION ---
authenticator = stauth.Authenticate(
    credentials=st.secrets['credentials'].to_dict(),
    cookie_name=st.secrets['cookie']['name'],
    cookie_key=st.secrets['cookie']['key'],
    cookie_expiry_days=st.secrets['cookie']['expiry_days']
)

if not st.session_state.get("authentication_status"):
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        try:
            authenticator.login(captcha=True, single_session=True, clear_on_submit=True)
        except LoginError as e:
            if "Captcha entered incorrectly" in str(e):
                st.error('Captcha is incorrect')
                st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during login: {e}")
            st.session_state["authentication_status"] = False
            st.stop()
        if st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
            st.stop()
        elif st.session_state["authentication_status"] is None:
            st.stop()

def load_css(css_file_path):
    try:
        with open(css_file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {css_file_path}. Create './assets/style.css'")
load_css("assets/style.css")

# --- MAIN APP ---
def main_app():
    username = st.session_state['username']

    def handle_question_click(question):
        st.session_state.user_input = question

    with st.sidebar:
        authenticator.logout('Logout', 'main', key='logout_button', use_container_width=True)
        st.markdown("---")

        with st.container(border=True):
            model_options = list(AVAILABLE_MODELS.keys())
            if 'selected_model_name' not in st.session_state:
                st.session_state.selected_model_name = DEFAULT_MODEL_NAME

            selected_model_name = st.selectbox(
                "Choose an AI Model:", options=model_options,
                index=model_options.index(st.session_state.selected_model_name),
                key="model_selector"
            )
            if selected_model_name != st.session_state.selected_model_name:
                st.session_state.selected_model_name = selected_model_name
                st.toast(f"Model changed to: {selected_model_name}", icon="🤖")

            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                save_chat_history(username, [])
                st.toast("Chat history cleared!", icon="🧹")
                st.rerun()

        st.markdown("---")
        st.markdown("## ❓ Select a Question")
        with st.container(border=True, height=380):
            for i, question in enumerate(QUESTIONS):
                st.button(
                    question, key=f"q_btn_{i}",
                    on_click=handle_question_click,
                    args=(question,),
                    use_container_width=True
                )

    st.header("Chat with Your Documents")
    st.caption(f"⚡ Using Model: {st.session_state.selected_model_name}")

    @st.cache_resource
    def load_query_engine(model_name):
        model_id = AVAILABLE_MODELS[model_name]

        if "gemini" in model_id:
            llm = GoogleGenAI(model=model_id, api_key=st.secrets["GEMINI_API_KEY"])
        elif "gpt" in model_id:
            llm = OpenAI(model=model_id, api_key=st.secrets["OPENAI_API_KEY"])
        elif "claude" in model_id:
            llm = Anthropic(model=model_id, api_key=st.secrets["ANTHROPIC_API_KEY"])
        else:
            st.error(f"Model '{model_name}' not recognized. Please check app_config.py.")
            st.stop()

        Settings.llm = llm
        Settings.embed_model = OpenAIEmbedding(
            model_name=st.secrets["EMBED_MODEL"],
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_collection(name=COLLECTION_NAME))
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        return index.as_query_engine(
            similarity_top_k=10,
            text_qa_template=PromptTemplate(UNIFIED_SYSTEM_INSTRUCTION),
        )

    query_engine = load_query_engine(st.session_state.selected_model_name)

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history(username)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = None
    if prompt := st.chat_input("Ask a question about your documents..."):
        user_input = prompt
    elif "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.pop("user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                with st.spinner(f"Thinking with {st.session_state.selected_model_name}..."):
                    response = query_engine.query(user_input)
                    answer = response.response
                    source_nodes = response.source_nodes
                st.markdown(answer)

                with st.expander("View Sources"):
                    if source_nodes:
                        for i, node in enumerate(source_nodes):
                            st.write(f"**Source {i+1}:** `{node.metadata.get('source_file', 'N/A')}` (Score: {node.score:.4f})")
                            st.text_area(
                                label=f"Source Content {i+1}", value=node.get_text(),
                                height=150, key=f"source_{i}", label_visibility="collapsed"
                            )
                    else:
                        st.info("No sources were retrieved for this query.")

                st.session_state.messages.append({"role": "assistant", "content": answer})
                save_chat_history(username, st.session_state.messages)

            except Exception as e:
                error_message = f"An error occurred while generating the response: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                save_chat_history(username, st.session_state.messages)

if st.session_state.get("authentication_status"):
    main_app()