import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
# from langchain_deepseek import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"
loader = TextLoader(markdown_path, encoding="utf-8")
docs = loader.load()
# print(type(markdown_path),type(loader),type(docs))
# <class 'str'> <class 'langchain_community.document_loaders.text.TextLoader'> <class 'list'>

# print(type(docs[0]))  # <class 'langchain_core.documents.base.Document'>
# print(docs[0].metadata)  # {'source': '../../data/C1/markdown/easy-rl-chapter1.md'}
# print(docs[0].page_content[:50])  # '强化学习（Reinforcement Learning）概述\n
# print(docs[0].get_lc_namespace())  # ['langchain', 'schema', 'document']

text_splitter = RecursiveCharacterTextSplitter()
texts = text_splitter.split_documents(docs)
# print(type(text_splitter),type(texts))  # <class 'langchain_text_splitters.recursive_character_text_splitter.RecursiveCharacterTextSplitter'> <class 'list'>

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(texts)

question = "文中举了哪些例子？"

retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# print(type(retrieved_docs), len(retrieved_docs))  # <class 'list'>, 3
# print(retrieved_docs[0].page_content)   

prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”
  
上下文:
{context}

问题: {question}

回答:"""
                                          )

llm = ChatOpenAI(
    model="deepseek-flash",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
