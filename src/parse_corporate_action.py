from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_aws import ChatBedrock
from typing import Optional
from pydantic import BaseModel, Field
import boto3
import os

# Fetch AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN')

def get_pdf_content(documents):
    raw_text = []
    for document in documents:
        loader = PyPDFLoader(document)
        raw_text.extend(loader.load())
    return raw_text

def get_vectorstore_text_splitter(text):
    # convert text to chunks of data
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # create vector embeddings
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_vectorstore_recursive_text_splitter(docs):
    # convert text to chunks of data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    # create vector embeddings
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def save_vector_store(documents, store_name):
    extracted_text = get_pdf_content(documents)
    vectorstore = get_vectorstore_recursive_text_splitter(extracted_text)
    vectorstore.save_local(store_name)

def load_vector_store(store_name):
    store = FAISS.load_local(store_name, embeddings,allow_dangerous_deserialization=True)
    return store

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

llm = ChatBedrock(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region='us-west-2',
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs=dict(temperature=0)
)

class Position:
    account: str
    security: str
    shares: int
    price: float
    def __init__(self, account, security, shares, price):
        self.account = account
        self.security = security
        self.shares = shares
        self.price = price
    def __str__(self):
        return f"Account={self.account} Security={self.security} Shares={self.shares} Price={self.price}"
    def __repr__(self):
        return f"Account={self.account} Security={self.security} Shares={self.shares} Price={self.price}"

class CorpAction(BaseModel):
    security: str = Field(description="full company name")
    original_shares: int = Field(description="shares before the change")
    new_shares: int = Field(description="shares after the change")
    effective_date: str = Field(description="effective date")
    ratio: Optional[float] = Field(
            default=None, description="ratio of the share change"
        )

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    event_type: str
    corp_action: CorpAction

# Define application steps
def retrieve(state: State):
    vectorstore = load_vector_store('dtcc')
    retrieved_docs = vectorstore.similarity_search(state['question'],k=4,search_type="similarity")
    event_type = 'reverse_split'
    return {"context": retrieved_docs, "event_type": event_type}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    prompt = PromptTemplate.from_template(
    """
    You are an assistant for question-answering tasks.
    Question: {question}
    Answer:
    """ )
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    structured_llm = llm.with_structured_output(CorpAction)
    corp_action = structured_llm.invoke(messages)
    return { "answer": response.content, "corp_action": corp_action}
def check_corpaction__type(state:State):
    if state['event_type'] == 'spit':
        return "process_split"
    else:
        return "process_reverse_split"

def process_split(state:State):
    pass

def process_reverse_split(state:State):
    corp_action = state['corp_action']
    if corp_action.ratio is None:
        corp_action.ratio = corp_action.new_shares/corp_action.original_shares

    positions = [
        Position(account='123', security='111, Inc.', shares = 1000, price = 0.05),
        Position(account='ABC', security='111, Inc.', shares = 500, price = 0.20)
    ]
    print("\nPosition before corporate action:\n")
    print(positions)
    positions = [Position(account=p.account,security=p.security,shares=p.shares/corp_action.ratio,price=p.price) for p in positions ]
    print("\nPosition after corporate action:\n")
    print(positions)
    print('\n')
if __name__ == "__main__":
    # https://www.sec.gov/search-filings
    # https://www.sec.gov/ix?doc=/Archives/edgar/data/84246/000008424624000025/tmb-20241107x8k.htm
    urls = ['https://www.sec.gov/Archives/edgar/data/1749864/000101915525000003/ratio424b3.htm']

    #save_vector_store(['pdf/YI_ADR_CorpAction.pdf'], 'dtcc')
    # Mac workaround:
    save_vector_store(['/Users/elizabethharasymiw/Documents/Github/dtcch-2025-abn-clearing-chicago/src/pdf/YI_ADR_CorpAction.pdf'], 'dtcc')

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_node("process_split", process_split)
    graph_builder.add_node("process_reverse_split", process_reverse_split)
    graph_builder.add_conditional_edges("generate", check_corpaction__type)
    graph_builder.add_edge('process_split', END)
    graph_builder.add_edge('process_reverse_split', END)
    graph = graph_builder.compile()

    # Visualize your graph
    try:
        png_graph = graph.get_graph().draw_mermaid_png()
        with open("dtcc-corpaction.png", "wb") as f:
            f.write(png_graph)

    except Exception as e:
        print(str(e))
        pass
    response = graph.invoke({"question": "How is American Depositary share changed due to this filing?"})

    print(response["answer"])