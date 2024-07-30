from dataLoader import DataLoader 

from langchain.chains import history_aware_retriever 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.retrieval import create_retrieval_chain 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

import os 

class CustomRAG:
    def __init__(self, filePath: str):
        # load_dotenv()
        self.chat_history = []
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key="AIzaSyDgxQKvx1M1XM4vMObBFw7xV-dZ6rZuYck")
        dataLoad = DataLoader(filePath)
        self.retriever = dataLoad.dataLoader()
        system_prompt = "You are an AI bot. You will be given an context. You task is to answer the questions base on the context provided and the chat history.\
            If the input/question in outside of the context you won't answer just say 'I dont know please provide more context!'."
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )



        


        history = history_aware_retriever.create_history_aware_retriever(self.llm, self.retriever, self.prompt)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt+"\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retriever_chain = create_retrieval_chain(history, document_chain)







