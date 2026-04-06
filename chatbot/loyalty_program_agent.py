from pathlib import Path

env_path = Path(".") / ".env"         
env_path.write_text("OPENAI_API_KEY=''your_key'')

"""mileage_rag_runtime.py
Load the pre‑built MileagePlus PDF Chroma store and expose ask_mileage().
Place this file in dispatcher_repo/ alongside the `stores/` directory.
"""
# loyalty_program_agent.py

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma  # updated for langchain v0.2+
from langchain.chains import RetrievalQA

# Load environment variables (OPENAI_API_KEY)
load_dotenv(dotenv_path=Path(".") / ".env")


class LoyaltyProgramAgent:
    def __init__(self,
                 store_dir: str = "stores/chroma_mileage_pdf",
                 collection_name: str = "ua_mileage_pdf",
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
        """Answer a question about the MileagePlus program"""
        return self.chain.run(question)


if __name__ == "__main__":
    agent = LoyaltyProgramAgent()
    demo_q = "How many PQP to reach Premier Platinum?"
    print("Q:", demo_q)
    print("A:", agent.ask(demo_q))
