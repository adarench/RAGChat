import os
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.llms import LLMMetadata
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor, MetadataReplacementPostProcessor
from llama_index.core import StorageContext, load_index_from_storage

st.set_page_config(page_title="Conversation", page_icon="📈")

st.header("Chat with your data 💬 📚")


@st.cache_resource(show_spinner=False)
def get_index():
    api_key = "your-api-key-here"

    llm = OpenAI(model="gpt-4o", api_key=api_key)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                       query_instruction='Represent this sentence for searching relevant passages:')
    Settings.llm = llm
    Settings.embed_model = embed_model

    persist_dir = "./index/"
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    # Load the index from the saved directory
    index = load_index_from_storage(storage_context)

    return index


index = get_index()

# Hyperparameter settings
hyperparams = st.expander("Set hyperparams")
hyperparams.selectbox('Select the number of TopK chunks', [idx + 1 for idx in range(20)], key='option', index=8)

# System prompt that guides the LLM responses
SYSTEM_PROMPT = """You are an expert at marketing strategy and planning for large organizations. Answer the questions based on the presented context only."""
hyperparams.text_area("Write a system prompt", value=SYSTEM_PROMPT, key='system_prompt')
hyperparams.slider(
    'Select a similarity cutoff',
    0.0, 100.0, 50.0, step=1.0, key='similarity_cutoff')

if "messages" not in st.session_state.keys():  
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question!"}
    ]

if st.chat_input("Your question", key='prompt'): 
    st.session_state.messages.append({"role": "user", "content": st.session_state.prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                memory = ChatMemoryBuffer.from_defaults(token_limit=3600)
                postprocessor = SimilarityPostprocessor(similarity_cutoff=st.session_state.similarity_cutoff / 100)
                
                st.session_state['chat_engine'] = index.as_chat_engine(verbose=True,
                                                                       similarity_top_k=st.session_state.option,
                                                                       node_postprocessors=[postprocessor,
                                                                                            MetadataReplacementPostProcessor(
                                                                                                target_metadata_key="window")
                                                                                            ],
                                                                       chat_mode=ChatMode.CONTEXT,
                                                                       system_prompt=st.session_state.system_prompt,
                                                                       memory=memory)

                history = [ChatMessage(role=m['role'], content=m['content']) for m in st.session_state.messages[1:-1]]
                response = st.session_state.chat_engine.stream_chat(st.session_state.prompt, history)
                st.session_state['response'] = response
                placeholder = st.empty()
                full_response = ''
                source_nodes = st.session_state['response'].source_nodes
                for item in st.session_state['response'].response_gen:
                    full_response += item
                    placeholder.write(f"{full_response.strip()}")
                placeholder.write(f"{full_response.strip()}")

                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

                if 'response' in st.session_state:
                    source_nodes = st.session_state['response'].source_nodes
                    expander = st.expander("See sources")

                    for idx, n in enumerate(source_nodes):
                        expander.markdown("""---""")
                        expander.markdown(
                            f"##### SOURCE {idx + 1}:\n\n**TEXT**: {n.text},\n\n**SIMILARITY**: {n.score * 100:.2f}%\n\n**METADATA**:")
                        expander.json(n.metadata, expanded=False)

        except Exception as e:
            st.error("An error occurred: {e}".format(e=e))
