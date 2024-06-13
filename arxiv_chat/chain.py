from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.retrievers import ArxivRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = Ollama(model = 'qwen2')

retriever = ArxivRetriever(load_max_docs = 5,
                           top_k_results = 5,
                           doc_content_chars_max = 5000)

memory = ConversationBufferWindowMemory(
    memory_key = 'chat_history',
    k = 5,
    return_messages = True
)

def format_docs(docs):
    return '\n\n'.join(
        f'arXiv {idx + 1}:\n{doc.page_content}' for idx, doc in enumerate(docs)
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'You are a computer scientist who is good at math and can explain complex concepts in simple words. You \
            help answer questions based on articles from the arXiv. If you know fundamental works, rely primarily on \
            them when answering. When answering, consider the following information: \
            {context}',
        ),
        MessagesPlaceholder('chat_history', optional = True),
        (
            'human',
            '{question}'
        )
    ]
)

agent = (
    {'context': RunnablePassthrough() | retriever | format_docs,
     'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )