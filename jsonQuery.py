import os
import openai
from dotenv import load_dotenv
from pathlib import Path
from langchain import OpenAI
from llama_index import download_loader
from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

class JSONQuery:
    def __init__(self, cfg):
        self._cfg = cfg
        # PDF Folder

        self.JSONReader = download_loader("JSONReader")
        loader = self.JSONReader()
        self.documents = loader.load_data(Path(cfg))

        # define LLM
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0))
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

        self.index = None
        self.query_engine = None

    #Build Index from storage folder
    def initStorage(self):
        index = KeywordTableIndex.from_documents(self.documents, service_context=self.service_context)

        index.storage_context.persist()
        self.query_engine = index.as_query_engine()

    #Load Index from storage folder
    def loadStorage(self):
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = load_index_from_storage(storage_context)
        self.query_engine = self.index.as_query_engine()

    #Query
    def pdfQuery(self, message):
        
        response = self.query_engine.query(message)

        print(response)

if __name__ == "__main__":
    ex = {
        "dir":"reports"
    }

obj = JSONQuery(cfg='./sensortest5.json')
#obj.initStorage()
obj.loadStorage()

obj.pdfQuery("")

#obj.pdfQuery("Who are students that have a math score greater 90 and a science score is greater than 90?")

#"Plot Sarah's math scores over time."
"""
Sarah's Math Scores Over Time:
90 (1686862367000)
91 (1687035167000)
92 (1687207967000)
"""