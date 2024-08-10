from utilities.utils import get_answer,create_qa_chain
import pinecone

# Get Answers

questions = [
    "What is the business outlook for Apple?",
    "What are the risk factors impacting Apple?",
    "What product brings Apple the most revenue? What percentage of the revenue does that contribute?",
    "What are Apple's key products and services?",
    "What are the most significant events in Apple's history?",
    "What are the most significant events identified in this report?",
    "Were any new risks identified in this report? If so, what were the new risks?",
    "Should I buy $10K in Apple stock?",
    "When was this report released?",
    "What period does this cover?",
    "Who are the key executives of this company?",
    "Who are the board of directors?"
]


# Initiate index
pinecone.init(api_key="", environment="gcp-starter")
index = pinecone.Index("vectordb1")

chain = create_qa_chain(db_name='vectordb1', emb_model="text-embedding-ada-002", openai_api_key='', with_meta_filteriing=False)

answer,source_docs = get_answer(chain,question=questions[0])

print(answer)
