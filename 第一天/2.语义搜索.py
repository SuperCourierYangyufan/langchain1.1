from itertools import chain
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 加载
load_dotenv()

# 初始化
# chat = ChatOpenAI(
#     model_name=os.getenv("OPEN_AI_CHAT_MODEL"),
#     openai_api_key=os.getenv("OPEN_AI_API_KEY"),
#     openai_api_base=os.getenv("OPEN_AI_BASE_URL"),
#     verbose=True
# )

# web加载
loader: PlaywrightURLLoader = PlaywrightURLLoader(urls=["https://news.baidu.com/"], remove_selectors=["header", "footer"])
docs = loader.load()

# 切割器
textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=True
)

# 切割
allSplit = textSplitter.split_documents(docs)


# 向量大模型
embedding = OpenAIEmbeddings(
    model=os.getenv("OPEN_AI_EMBEDDING_MODEL"),
    api_key=os.getenv("OPEN_AI_API_KEY"),
    base_url=os.getenv("OPEN_AI_BASE_URL")
)

# 向量库
chroma = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db"
)

# 存储
chroma.add_documents(documents=allSplit)

# 相似度查询
results = chroma.similarity_search("今日新闻")
print(results)

# 分数相似度查询
results = chroma.similarity_search_with_score("今日新闻")
print(results)

# 向量相似度查询
results = chroma.similarity_search_by_vector(embedding.embed_query("今日新闻"))
print(results)


from langchain_core import chain
@chain
def retrierver(query: str) -> List[Document]:
    return chroma.similarity_search(query=query,k=1)
