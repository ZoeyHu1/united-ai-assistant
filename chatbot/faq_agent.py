from pathlib import Path
env_path = Path(".") / ".env"         
env_path.write_text("OPENAI_API_KEY=key\n")

"""faq_rag_runtime.py
Load the pre‑built FAQ Chroma store and expose ask_faq().
Place this file in dispatcher_repo/ alongside the `stores/` directory.
"""
# faq_agent.py
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma  # updated import path
from langchain.chains import RetrievalQA

# ✅ Load .env (ensure OPENAI_API_KEY is set)
load_dotenv(dotenv_path=Path(".") / ".env")


class FAQAgent:
    def __init__(self,
                 store_dir: str = "stores/chroma_faq",
                 collection_name: str = "ua_faq_demo",
                 embedding_model: str = "text-embedding-3-small",
                 chat_model: str = "gpt-4o-mini"):

        self.db = Chroma(
            collection_name=collection_name,
            persist_directory=store_dir,
            embedding_function=OpenAIEmbeddings(model=embedding_model)
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})

        self.chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=chat_model, temperature=0.2),
            retriever=self.retriever,
            chain_type="stuff"
        )

    def ask(self, question: str) -> str:
        """Ask a question to the FAQ agent"""
        return self.chain.run(question)


if __name__ == "__main__":
    faq = FAQAgent()
    q = "Can I bring a full-size carry-on in Basic Economy?"
    print("Q:", q)
    print("A:", faq.ask(q))
