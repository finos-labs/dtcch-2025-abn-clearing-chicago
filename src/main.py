from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
import boto3

boto3.setup_default_session(profile_name='216989139036_hackathon-participant')
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)

llm = ChatBedrock(
    credentials_profile_name='216989139036_hackathon-participant',
    region='us-west-2',
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs=dict(temperature=0),
    # other params...
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)
