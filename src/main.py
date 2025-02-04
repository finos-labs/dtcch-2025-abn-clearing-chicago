from langchain_aws import ChatBedrock

model = ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    beta_use_converse_api=True)

model.invoke("Hello, world!")
