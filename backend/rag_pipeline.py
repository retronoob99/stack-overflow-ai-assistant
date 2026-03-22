import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any, Optional
from utils import EmbeddingManager, VectorStore, RAGretriever

load_dotenv()


class AnswerFetcher:

    def __init__(self, answers_path: str):
        self.answers_df = pd.read_parquet(answers_path)
        print(f"Answers loaded: {len(self.answers_df):,} rows")

    def fetch(self, question_ids: List[int], top_n: int = 3) -> Dict[int, List[str]]:
        # fetches top 3 answers for each question_id

        answers_map = {}

        for qid in question_ids:
            # Filter answers for the given question_id
            matches = self.answers_df[self.answers_df['question_id'] == qid].head(top_n)

            if len(matches) == 0:
                answers_map[qid] = []
            else:
                answers_map[qid] = [
                    {
                        "body" : row['body'],
                        "score" : int(row['score']),
                        "is_accepted" : bool(row['is_accepted']),
                        "has_code" : bool(row['has_code']),
                        "answer_rank" : int(row['answer_rank'])
                    }
                    for _, row in matches.iterrows()
                ]
        return answers_map
    

class ContextBuilder:
    # Format retrieved questions and answers into a clean context string for the LLM
    

    def build(self, retrieved_docs: List[Dict], answers_map: Dict[int, List[Dict]]) -> str:
        # Builds context strings from retrieved questions and their answers

        context_blocks = []

        for doc in retrieved_docs:
            question_id = doc['question_id']
            title = doc['title']
            primary_tag = doc['primary_tag']
            score = doc['score']
            answers = answers_map.get(question_id, [])

            # Question block
            block = f"Question: {title}\nPrimary Tag: {primary_tag}\nScore: {score}\n"

            # Add answers if available
            if answers:
                for i, answer in enumerate(answers, start = 1):
                    accepted = f'Accepted ' if answer['is_accepted'] else ''
                    block += f""" Answer{i}: {accepted} {answer['body'][:500]}"""

            else:
                block+= " No answers found."

            context_blocks.append(block)

        final_context = "\n".join(context_blocks)
        return final_context


class LLMcaller:
    # LLM call using groq

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(api_key=self.api_key, model = "llama-3.1-8b-instant", temperature=0.1, max_tokens=1024)
        print("LLM initialized")


    def call_rag(self, query: str, context: str) -> str:
        system_prompt = """" You are a helpful Stack Overflow programming assistant.
Answer the user's question using ONLY the context provided below.
Follow these rules strictly:
- Answer ONLY programming and tech related questions
- Use the context provided to answer — do NOT use your own knowledge
- If context has code examples, include them in your answer
- Format your answer clearly with proper code blocks
- If context is insufficient, say so honestly
- Do NOT make up information
        """ 

        user_prompt = f""" Context from stack overflow

        Context:
        {context}

        Question: {query}

        Answer based on context above:"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content
    
    def call_general(self, query: str) -> str:
        '''Calls LLM without context — used when no relevant question found'''

        system_prompt = """You are a helpful programming and tech assistant.
Follow these rules strictly:
- Answer ONLY programming and tech related questions
- If question is not related to programming or tech, politely decline
- Be concise and accurate
- Include code examples where relevant"""

        human_prompt = f"""Question: {query}

Answer:"""

        
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        return response.content
    

class Guardrail:
    '''Checks if query is programming or tech related before processing'''

    def __init__(self, llm_caller: LLMcaller):
        self.llm = llm_caller.llm

    def is_greeting(self, query: str) -> bool:
        '''Checks if query is a greeting'''
        greetings = [
            "hi", "hello", "hey", "hii", "helo",
            "good morning", "good evening", "good afternoon",
            "howdy", "greetings", "sup", "what's up", "whats up"
        ]
        return query.strip().lower() in greetings

    def get_greeting_response(self) -> str:
        '''Returns a friendly intro message for greetings'''
        return """👋 Hello! I'm your Stack Overflow AI Assistant.

I can help you with:
  • 🐍 Python, JavaScript, Java, C++ and more
  • 🌐 Web development (HTML, CSS, React)
  • 🗄️ Databases and SQL
  • 🔧 Debugging and error fixing
  • 💡 Programming concepts and best practices

Ask me any programming or tech question and I'll find the best answers from Stack Overflow's knowledge base!

**Try asking:**
  • "How do I reverse a list in Python?"
  • "What is async/await in JavaScript?"
  • "How to fix a NullPointerException in Java?" """

    def is_tech_query(self, query: str) -> bool:
        '''Returns True if query is programming/tech related'''

        classification_prompt = f"""You are a strict query classifier for a Stack Overflow programming assistant.

Classify if the following query is related to:
- Programming / coding
- Software development
- Technology / tools
- Databases
- Web development
- DevOps / cloud
- Computer science concepts

Reply with ONLY "YES" or "NO". No explanation, no punctuation, just YES or NO.

Query: {query}"""

        response = self.llm.invoke([
            HumanMessage(content=classification_prompt)
        ])

        answer = response.content.strip().upper()
        return answer == "YES"

    def get_rejection_message(self, query: str) -> str:
        '''Returns a polite rejection message for non-tech queries'''
        return """⛔ I'm a programming and tech assistant — I can only answer coding and technology related questions.

Try asking something like:
  • "How do I reverse a list in Python?"
  • "What is async/await in JavaScript?"
  • "How to fix a NullPointerException in Java?"
  • "What is the difference between SQL and NoSQL?" """


class RAGPipeline:
    # Main pipeline to orchestrate the RAG process

    def __init__(self, vector_store_path: str, answers_path: str, score_threshold: float = 0.5):
        
        self.score_threshold = score_threshold
        self.embedding_manager = EmbeddingManager()

        self.vector_store = VectorStore(index_name = 'questions', persist_directory=vector_store_path)
        self.vector_store.load()

        self.retriever = RAGretriever(self.vector_store, self.embedding_manager)

        self.answer_fetcher = AnswerFetcher(answers_path = answers_path)

        self.context_builder = ContextBuilder()

        self.llm_caller = LLMcaller()

        self.guardrail = Guardrail(llm_caller = self.llm_caller)

        print("RAG Pipeline initialized")

    def run(self, query: str, top_k: int = 5) -> dict:
        # Runs the RAG pipeline for a given query

        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")
        
        # Step 0: Check for greetings
        if self.guardrail.is_greeting(query):
            print("Query is a greeting. Returning greeting response.")
            return {
                "answer" : self.guardrail.get_greeting_response(),
                "is_relevant" : False,
                "is_tech" : False,
                "sources" : [],
                "path" : "greeting"
            }


        # Step 1: Guardrail check
        print("Checking if query is tech related...")

        is_tech = self.guardrail.is_tech_query(query)

        if not is_tech:
            print("Query is not tech related. Returning rejection message.")

            return {
                "answer" : self.guardrail.get_rejection_message(query),
                "is_relevant" : False,
                "is_tech" : False,
                "sources" : [],
                "path" : "rejected"
            }
        
        print("Query is tech related. Proceeding with RAG pipeline.")

        # Step 2: Retrieve relevant questions
        print("Retrieving relevant questions from vector store...")

        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # Step 3: Check relevance score
        print("Checking relevance score")
        max_score = max([doc['score'] for doc in retrieved_docs]) if retrieved_docs else 0

        is_relevant = max_score >= self.score_threshold
        print(f"Max relevance score: {max_score:.4f} (Threshold: {self.score_threshold}) - {'Relevant' if is_relevant else 'Not Relevant'}")

        # RAG - relevant query

        if is_relevant:
            print("Fetching answers for retrieved questions...")
            question_ids = [doc['question_id'] for doc in retrieved_docs]
            answers_map = self.answer_fetcher.fetch(question_ids)

            # Check if answers were found for the retrieved questions
            total_answers = sum(len(v) for v in answers_map.values())

            if total_answers == 0:
                print("No answers found for retrieved questions. Returning response without context.")
                answer = self.llm_caller.call_general(query)
                return {
                    "answer" : answer,
                    "is_relevant" : True,
                    "is_tech" : True,
                    "sources" : retrieved_docs,
                    "path" : "no_answers"
                }
            
            # Buld context and call LLM
            print("Building context and calling LLM...")

            context = self.context_builder.build(retrieved_docs, answers_map)
            answer = self.llm_caller.call_rag(query, context)

            sources = []
            for doc in retrieved_docs[:3]:
                qid = doc['question_id']
                answers = answers_map.get(qid, [])
                sources.append({
                    "rank" : doc["rank"],
                    "question_id" : qid,
                    "primary_tag" : doc['primary_tag'],
                    "title"       : doc["title"],
                    "top_answers" : [a["body"][:300] for a in answers[:3]]
                })
            
            return {
                "answer" : answer,
                "is_relevant" : True,
                "is_tech" : True,
                "sources" : sources,
                "path" : "rag"
            }
        
        else:
            print("General LLM path → no SO context...")
            answer = self.llm_caller.call_general(query)

            return {
                "answer" : answer,
                "is_relevant" : False,
                "is_tech" : True,
                "sources" : [],
                "path" : "general"
            }


        