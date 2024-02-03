from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

pdfreader = PdfReader('budget_speech.pdf')

# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
        
# We need to split the text using Character Text Split such that it should not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

document_search = FAISS.from_texts(texts, embeddings)

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1e-10})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=document_search.as_retriever(), return_source_documents=False)

query = "priorities of this budget"

result = qa.run({"query": query})
print(result)