from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi import FastAPI
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langserve import add_routes


# 1. Create prompt template
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

app = FastAPI(
    title="My LLM API",
    description="My first LLM API",
    version="1.0",
)


add_routes(
    app,
    chain,
    path="/chain"
    
)


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)

    