import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # parse the input query
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args=parser.parse_args()
    query_text = args.query_text

    # find the related docs in Chroma
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return
    
    # merge the query and choma search result as the LLM prompt input
    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template(context=context_text, question=query_text)
    print(prompt)
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # formatting the answer
    sources = [doc.metadata.get('source', None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()