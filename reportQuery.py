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

# get response from query
query_engine = index.as_query_engine()

# Query

def pdfQuery(message):

    response = query_engine.query(message)

    print(response)

pdfQuery("What is patient's date of birth?")


"""pdfQuery("at  hour 11:00 , what the HR min , max  and mean value of HR")
pdfQuery("Which hour has  highest HR Max")
pdfQuery("Which hour has min HR Min")
pdfQuery("Give the list of Ventricular Runs, and when did they occur")
"""