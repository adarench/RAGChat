from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer, util
import os


class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_chunk_size: int = 512, overlap_size: int = 50):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer  # Assume the tokenizer aligns with the model
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk_text(self, text, metadata):
        sentences = text.split('. ')
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for i in range(len(sentences) - 1):
            current_chunk.append(sentences[i])
            current_chunk_size += len(self.tokenizer.tokenize(sentences[i]))

            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1])
            if current_chunk_size > self.max_chunk_size or similarity < 0.7:
                chunks.append({
                    'text': '. '.join(current_chunk),
                    'metadata': metadata
                })
                current_chunk = current_chunk[-self.overlap_size:]
                current_chunk_size = sum([len(self.tokenizer.tokenize(s)) for s in current_chunk])

        if current_chunk:
            chunks.append({
                'text': '. '.join(current_chunk),
                'metadata': metadata
            })

        return chunks

    def get_nodes_from_documents(self, documents):
        nodes = []
        for document in documents:
            nodes.extend(self.chunk_text(document.text, document.metadata))
        return nodes


def deduplicated_documents(documents):
    documents = {document.text.strip().lower(): document for document in documents}
    return list(documents.values())


def load_data():
    api_key = "your-api-key-here"

    # Initialize the LLM and embedding models
    llm = OpenAI(model="gpt-4o", api_key=api_key)

    reader = SimpleDirectoryReader('./data', recursive=True, required_exts=['.pdf', '.txt', '.docx'])
    documents = reader.load_data(show_progress=True)
    documents = deduplicated_documents(documents)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                       query_instruction='Represent this sentence for searching relevant passages:')

    # Set global configurations
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Initialize chunker
    chunker = SemanticChunker()
    nodes = chunker.get_nodes_from_documents(documents)

    # Create the index and save it
    index = VectorStoreIndex.from_documents(nodes)

    # Save the index to the disk
    persist_dir = "./index/"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    index.save_to_disk(persist_dir)

    return index

