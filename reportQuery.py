from dotenv import load_dotenv
from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from langchain import OpenAI

load_dotenv()

documents = SimpleDirectoryReader('reports').load_data()

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# build index
index = KeywordTableIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist()
query_engine = index.as_query_engine()

# Query
def pdfQuery(message):

    response = query_engine.query(message)

    print(response)



#pdfQuery("Give me all the times where HR Max is greater 80 and the time where the HR Min is less than 50")

