from utilities.utils import *

def setup_vector_index():
    df = load_data(file_path='data/0000320193-23-000106.txt')
    index = create_vector_db(db_name='vectordb1', delete_if_exists=False)
    create_embeddings(df=df, index=index, emb_model="text-embedding-ada-002", openai_api_key='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')
    return create_qa_chain(db_name='vectordb1', emb_model="text-embedding-ada-002", openai_api_key='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', with_meta_filteriing=False)

# This ensures that the following code runs only if set_vector_index.py is run directly, not when imported
if __name__ == '__main__':
    chain = setup_vector_index()
    