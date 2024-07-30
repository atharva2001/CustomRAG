from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_community.vectorstores import FAISS 
from fileLoader import FileLoader 



class DataLoader:
    def __init__(self, file_path: str):
        self.embeddings = HuggingFaceBgeEmbeddings(model_name="all-minilm-l6-v2")
        self.file = FileLoader(file_path)

    def dataLoader(self):
        docs = self.file.fileLoader()

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        return self.vectorstore.as_retriever()

        