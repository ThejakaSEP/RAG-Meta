import json
import pandas as pd
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import spacy
import yahooquery as yq
from yahooquery import Ticker


def load_data(file_path='data/0000320193-23-000106.txt'):

    # Parsing the JSON string
    f = open(file_path, "r")

    data = json.loads(f.read())

    # Extracting the ticker symbol
    ticker_symbol = data["ticker"]

    # Creating a DataFrame with the required columns
    df = pd.DataFrame(columns=["Section", "Ticker Symbol", "Text"])

    # Iterating over the JSON data and extracting the relevant fields
    for key, value in data.items():
        if key.startswith("item"):
            df = df.append({
                "Section": key,
                "Ticker Symbol": ticker_symbol,
                "Text": value
            }, ignore_index=True)

    return df


def create_vector_db(db_name,delete_if_exists=False):
    pinecone.init(api_key="0d2a8be4-2ec0-432d-9fb5-df5bb3290c8a", environment="gcp-starter")

    if delete_if_exists:
        pinecone.delete_index(db_name)

    if db_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=db_name,
            metric='cosine',
            dimension=1536 # 1536 dim of text-embedding-ada-002
        )
    else:
        print("Index Already exists")

    index = pinecone.Index(db_name)

    return index


def create_embeddings(df=None,index=None,emb_model="text-embedding-ada-002",openai_api_key='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'):
    
    embed_model = OpenAIEmbeddings(model=emb_model,openai_api_key=openai_api_key)

    tokenizer = tiktoken.get_encoding('cl100k_base')
    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Create Embeddings
    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in tqdm(df.iterrows()):
        # first get metadata fields for this record
        metadata = {
            'section': record['Section'],
            'ticker': record['Ticker Symbol'],
            # 'source': record['Text']
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['Text'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j,"text":text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed_model.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed_model.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

    print(index.describe_index_stats())


def create_qa_chain(db_name,emb_model,openai_api_key,with_meta_filteriing=False):
    index = pinecone.Index(db_name)

    # Create Embed_model
    embed_model = OpenAIEmbeddings(model=emb_model,openai_api_key=openai_api_key)
    
    text_field = "text"

    vectorstore = Pinecone(
        index,embed_model.embed_query, text_field
    )

    # completion llm
    llm = ChatOpenAI(
        openai_api_key='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',
        model_name='gpt-3.5-turbo',
        temperature=0.6
    )

    if with_meta_filteriing:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5,'filter': {'section':'item1'}}) 
    
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 

    
    # Change Prompt
    prompt_template = """
    As a Q&A agent, please answer the question below, using only the context below.
    If the context does not provide information, just return "Information not found in Context".

    question: {question}

    context: {context}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


    chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs = {"prompt": prompt},
    return_source_documents=True)

    return chain

def get_answer(chain,question):
    result = chain({"query": question})

    answer = result['result']
    source_docs = result['source_documents']

    return answer,source_docs

def get_orgs(text:str):
    try:
        nlp = spacy.load('en_core_web_md')
    except OSError:
        from spacy.cli import download
        download('en_core_web_md')
        nlp = spacy.load('en_core_web_md')

    doc = nlp(text)
    return [x.text for x in (nlp(doc)).ents if x.label_ == "ORG"]


def get_ticker(name: str, preferred_exchanges=['NYQ', 'NMS']):
    try:
        data = yq.search(name)
    except ValueError: # Will catch JSONDecodeError
        print(name)
    else:
        quotes = data['quotes']
        if len(quotes) == 0:
            return 'No Symbol Found'


        symbol = quotes[0]['symbol']
        found = False
        for quote in quotes:
             if quote['exchange'] in preferred_exchanges:
                symbol = quote['symbol']
                found = True
                break
        if not found:
            for quote in quotes:
                print(f"not found: {quote['exchange']}")
        return symbol
    
    def get_tickers_from_text(text:str):
        orgs = get_orgs(text)
        
        tickers = [{'ticker': get_ticker(x), 'name': x} for x in orgs]

        return tickers
