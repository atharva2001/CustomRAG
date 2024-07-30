from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader 

class FileLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def fileLoader(self):
        loader = TextLoader(self.file_path)
        documents = loader.load() 
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=16)
        self.docs = text_splitter.split_documents(documents)

        return self.docs 