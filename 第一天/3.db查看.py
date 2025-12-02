import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 加载环境变量
load_dotenv()

# 初始化向量模型
embedding = OpenAIEmbeddings(
    model=os.getenv("OPEN_AI_EMBEDDING_MODEL"),
    api_key=os.getenv("OPEN_AI_API_KEY"),
    base_url=os.getenv("OPEN_AI_BASE_URL")
)

def list_collection(db_path: str = "./chroma_langchain_db", collection_name: str = "example_collection"):
    """
    查询 Chroma 数据库中的所有数据
    
    参数:
        db_path: 数据库路径
        collection_name: 集合名称
    """
    # 初始化 Chroma 客户端
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=db_path
    )
    
    # 获取底层集合对象
    collection = chroma._collection
    
    # 查询所有数据
    results = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    # 获取记录总数
    total_count = len(results['ids'])
    
    print(f"\n{'='*60}")
    print(f"数据库路径: {db_path}")
    print(f"集合名称: {collection_name}")
    print(f"总记录数: {total_count}")
    print(f"{'='*60}\n")
    
    # 遍历并输出每条记录
    for i in range(total_count):
        print(f"\n--- 记录 {i + 1}/{total_count} ---")
        print(f"ID: {results['ids'][i]}")
        print(f"文档内容: {results['documents'][i][:100]}...")  # 只显示前100个字符
        print(f"元数据: {results['metadatas'][i]}")
        print(f"向量维度: {len(results['embeddings'][i]) if results['embeddings'][i] is not None else 0}")
        print("-" * 60)
    
    return {
        "total_count": total_count,
        "ids": results['ids'],
        "documents": results['documents'],
        "metadatas": results['metadatas']
    }

def remove_all(db_path: str = "./chroma_langchain_db", collection_name: str = "example_collection"):
    """
    删除 Chroma 数据库中的所有数据
    
    参数:
        db_path: 数据库路径
        collection_name: 集合名称
    """
    # 初始化 Chroma 客户端
    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=db_path
    )
    
    # 获取底层集合对象
    collection = chroma._collection
    
    # 先获取所有 ID
    results = collection.get()
    all_ids = results['ids']
    
    if len(all_ids) > 0:
        # 删除所有记录
        collection.delete(ids=all_ids)
        print(f"\n{'='*60}")
        print(f"成功删除 {len(all_ids)} 条记录")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("数据库为空，无需删除")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    # 调用查询函数
    remove_all()
    result = list_collection()
    
    print(f"\n\n{'='*60}")
    print("查询完成!")
    print(f"共查询到 {result['total_count']} 条记录")
    print(f"{'='*60}")