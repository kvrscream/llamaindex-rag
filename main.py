from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import os


llm = Ollama(model='llama3.2:latest', request_timeout=120.0)
# ollama_embedding = OllamaEmbedding(
#     model_name="llama3.2:1b",
#     base_url="http://localhost:11434",
#     ollama_additional_kwargs={"mirostat": 0},
# )
Settings.llm = llm
# Settings.embed_model = ollama_embedding
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


path = os.path.join('docs')

text_splitter = SentenceSplitter(chunk_size=300, chunk_overlap=80)
Settings.text_splitter = text_splitter

documents = SimpleDirectoryReader(path).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)

retriever = VectorIndexRetriever(
  index=vector_index,
  similarity_top_k=20
)

syntetizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
  retriever=retriever,
  response_synthesizer=syntetizer,
  node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.4)],
)

def create_prompt(query):
  prompt = f'''
    Você é um assistente especialista em análise de documentos financeiros. \n
    Seu objetivo é fornecer respostas somente sobre com base no documento vetorizado fornecido. Em hipose alguma, deve-se fornecer informações fora desse contexto.\n
    Seja sempre o mais cordial possível e com bom humor. 
    \n
    \n
    Data a pergunta: {query}. Responda com base nos documentos
  '''
  return prompt

while True:
  query = input('Em que posso ajudar? ')

  if query == 'fim':
    break
  else:
    response = query_engine.query(create_prompt(query=query))
    print(response)