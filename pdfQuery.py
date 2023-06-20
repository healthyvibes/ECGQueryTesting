from dotenv import load_dotenv
from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    PromptHelper,
    load_index_from_storage,
)
from langchain import OpenAI

load_dotenv()


class PDFQuery:
    def __init__(self, cfg):

        self._cfg = cfg
        # PDF Folder
        self.documents = SimpleDirectoryReader('%s' % self._cfg["dir"]).load_data()

        # define LLM
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0))
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

        self.index = None
        self.query_engine = None

        self.max_input_size    = 4096
        self.num_outputs       = 512
        self.chunk_size_limit  = 600
        self.max_chunk_overlap = 20

        self.prompt_helper =     prompt_helper = PromptHelper(
            max_input_size=self.max_input_size, 
            num_output=self.num_outputs, 
            max_chunk_overlap=self.max_chunk_overlap, 
            chunk_size_limit=self.chunk_size_limit)


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

obj = PDFQuery(cfg=ex)
#obj.initStorage()
obj.loadStorage()
obj.pdfQuery("What is the patient's date of birth?")