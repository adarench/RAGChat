import os
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.memory import ChatMemoryBuffer
from tempfile import NamedTemporaryFile
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SimilarityPostprocessor

st.set_page_config(page_title="Upload and Chat", page_icon="📈")
st.header("Upload and chat with your data 💬 📚")

def deduplicated_documents(documents):
    documents = {document.text.strip().lower(): document for document in documents}
    return list(documents.values())

@st.cache_resource(show_spinner=False)
def get_index():
    api_key = "your-api-key-here"

    llm = OpenAI(model="gpt-4o", api_key=api_key)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                       query_instruction='Represent this sentence for searching relevant passages:')
    Settings.llm = llm
    Settings.embed_model = embed_model

    data_dir = 'data'

    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist. Please ensure the data directory is available.")

    documents = SimpleDirectoryReader(data_dir).load_data()
    documents = deduplicated_documents(documents)
    node_parser = SentenceWindowNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex.from_documents(nodes)
    
    return index, [], []

index, file_names, file_paths = get_index()

directory = './temp'
if not os.path.exists(directory):
    os.makedirs(directory)

if "activate_chat" not in st.session_state:
    st.session_state.activate_chat = False

docs = st.file_uploader('⬆️ Upload your documents & Click to process',
                        accept_multiple_files=True,
                        type=['pdf', 'txt'])

pick_files = st.expander("Pick files")
pick_files.multiselect(
    "Select up to 10 files:",
    file_paths,
    max_selections=10, key='md_file_names'
)

if st.button('Process'):

    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question!"}
    ]
    files = [NamedTemporaryFile(dir=directory, suffix=f".{doc.type.split('/')[-1]}") for idx, doc in enumerate(docs)]

    for idx, doc in enumerate(docs):
        files[idx].write(doc.getbuffer())
    with st.spinner('Processing'):
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        documents = SimpleDirectoryReader(input_files=[directory + '/' + file.name.split('temp/')[-1] for file in files]
                                                      + st.session_state.md_file_names).load_data()

        documents = deduplicated_documents(documents)

        for document in documents:
            document.excluded_llm_metadata_keys = ["page_label", "file_name", "file_path", "creation_date",
                                                   "last_modified_date", "last_accessed_date", "file_type",
                                                   "file_size"]
            document.excluded_embed_metadata_keys = ["page_label", "file_name", "file_path", "creation_date",
                                                     "last_modified_date", "last_accessed_date", "file_type",
                                                     "file_size"]

        nodes = node_parser.get_nodes_from_documents(documents)
        index_chat = VectorStoreIndex.from_documents(documents)

        st.session_state.index_chat = index_chat
        st.session_state.activate_chat = True

        for file in files:
            file.close()

if st.session_state.activate_chat:
    hyperparams = st.expander("Set hyperparams")
    hyperparams.selectbox('Select the number of TopK chunks', [idx + 1 for idx in range(20)], key='option', index=8)

    SYSTEM_PROMPT = """You are an expert at marketing strategy and planning for large organizations. Answer the questions based on the presented context only."""
    hyperparams.text_area("Write a system prompt", value=SYSTEM_PROMPT, key='system_prompt')
    hyperparams.slider(
        'Select a similarity cutoff',
        0.0, 100.0, 50.0, step=1.0, key='similarity_cutoff')

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
                    
                    st.session_state['chat_engine'] = st.session_state.index_chat.as_chat_engine(verbose=True,
                                                                                                 similarity_top_k=st.session_state.option,
                                                                                                 node_postprocessors=[postprocessor,
                                                                                                                      MetadataReplacementPostProcessor(
                                                                                                                          target_metadata_key="window")
                                                                                                                     ],
                                                                                                 chat_mode=ChatMode.CONTEXT,
                                                                                                 system_prompt=st.session_state.system_prompt,
                                                                                                 memory=memory)

                    history = [ChatMessage(role=m['role'], content=m['content']) for m in
                               st.session_state.messages[1:-1]]
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
