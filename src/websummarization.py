from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_aws import ChatBedrock
import boto3
import os

# Fetch AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.environ.get('AWS_SESSION_TOKEN')

boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    #profile_name='216989139036_hackathon-participant'
)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-west-2',
    #credentials_profile_name='216989139036_hackathon-participant'
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs =  {
    "max_tokens": 512,
    "temperature": 0.0,
}

claude_3_client = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

urls =["https://www.sec.gov/Archives/edgar/data/1738906/000095010325000279/dp223261_ex9901.htm"]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
chunk_size = 3000
chunk_overlap = 200

text_splitter = CharacterTextSplitter(
    # separator = "\n\n"
    chunk_size=chunk_size, # Maximum size of a chunk
    chunk_overlap=chunk_overlap, # Maintain continuity, have some overlap of chunks
    length_function=len, # Count number of characters to measure chunk size
)

texts = text_splitter.split_text(data[0].page_content)

# Create Document objects for each text chunk
documents = [Document(page_content=t) for t in texts[:]]
docs = {
    "input_documents": documents,
    "input": "stm32"
}

chain = load_summarize_chain(llm=claude_3_client, chain_type='map_reduce')  # verbose=True optional to see what is getting sent to the LLM

response =  chain.invoke(docs)

print(response["output_text"])