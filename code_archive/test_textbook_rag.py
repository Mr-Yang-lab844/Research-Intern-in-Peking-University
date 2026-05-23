import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 加载知识库
docs = []
with open("./knowledge/medical_textbook_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        docs.append(Document(page_content=data["text"], metadata={"id": data["id"]}))
print(f"Loaded {len(docs)} documents")

# 构建向量库（首次运行会下载嵌入模型，可能需要几分钟）
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", model_kwargs={'device': 'cuda'})
vector_store = FAISS.from_documents(docs, embedding_model)
print("Vector store built")

# 测试检索
query = "血糖正常范围是多少？"
retrieved = vector_store.similarity_search(query, k=3)
for i, doc in enumerate(retrieved):
    print(f"\n--- Result {i+1} ---\n{doc.page_content[:200]}...")