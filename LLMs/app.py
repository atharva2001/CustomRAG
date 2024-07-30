from fastapi import FastAPI
import uvicorn 
from logic import CustomRAG
from langchain_core.messages import AIMessage, HumanMessage 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Custom RAG")
obj = CustomRAG("trial.txt")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

obj.chat_history.extend(
        [
            HumanMessage(
                content="You are an AI bot!",
                name = "human"
            ),
            AIMessage(
                content="Hello! I am an AI bot. How can I help you today?",
                name = "system"
            )
        ]
    )

@app.get("/", tags=["Home"])
def index():
    return {
        "Message": "Welcome to the fantasctic world of AI"
    }

@app.get("/query", tags=["Query"])
def query(question: str):
    response = obj.retriever_chain.invoke(
        {
            "chat_history": obj.chat_history,
            "input": question
        }
    )

    obj.chat_history.extend(
        [
            HumanMessage(
                content=question,
                name="human"
            ),
            AIMessage(
                content=response["answer"],
                name="system"
            )
        ]
    )


    return {
        question: response["answer"]
    }



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)