import os
import getpass
from typing import List

# 1. CORE IMPORTS (Fixed imports for your error)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma  # Using the new langchain-chroma package
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool, create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub

# 2. CONFIGURATION
# Ensure your API key is set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("AIzaSyBGoeYZfGNRahmNDQJwrjYS198sJ-s9TYo")

class PersonalAssistant:
    def __init__(self, db_path="./chroma_db"):
        # Gemini Embedding Model
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Gemini 2.0 Flash with Search Grounding enabled
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            tools=[{"google_search": {}}] # Native Google Search Grounding
        )
        self.db_path = db_path
        self.vectorstore = None

    # 3. DATA INGESTION PHASE (PDF -> Chunks -> Chroma)
    def ingest_pdfs(self, file_paths: List[str]):
        all_docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(all_docs)
        
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(f"✅ Ingested {len(splits)} chunks into {self.db_path}")

    # 4. FUNCTION CALLING (Mock Tools)
    @tool
    def check_calendar(date: str):
        """Checks the user's personal calendar for a given date (YYYY-MM-DD)."""
        # Mock data (In reality, this would call Google Calendar API)
        schedules = {"2026-01-08": "10 AM: RAG Architecture Review, 2 PM: Sync with Team"}
        return schedules.get(date, "No events found for this date.")

    # 5. QUERY & GENERATION (Agent Execution)
    def ask(self, query: str):
        # Load vectorstore if not already in memory
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.db_path, 
                embedding_function=self.embeddings
            )
            
        # Create the Retriever Tool
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever_tool = create_retriever_tool(
            retriever,
            "document_search",
            "Search for information within the uploaded PDF documents."
        )
        
        # Combine all tools (RAG, Calendar, and implicitly Search via LLM config)
        tools = [retriever_tool, self.check_calendar]
        
        # Pull standard prompt for tool-calling agents
        prompt = hub.pull("hwchase17/openai-functions-agent")
        
        # Initialize Agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Run
        result = agent_executor.invoke({"input": query})
        return result["output"]

# --- INITIALIZATION ---
assistant = PersonalAssistant()

# Step 1: Ingest (Uncomment this line the first time you run it)
# assistant.ingest_pdfs(["your_document.pdf"])

# Step 2: Ask a multi-dimensional question
# This will trigger: 
# 1. check_calendar (Function Tool) 
# 2. document_search (RAG Tool)
# 3. Google Search Grounding (Built-in to LLM)
answer = assistant.ask(
    "Check my calendar for 2026-01-08 and explain if I'll be busy "
    "to someone asking about 'Project X' mentioned in my PDFs. "
    "Also, provide the stock price of Google right now."
)

print(f"\n--- ASSISTANT RESPONSE ---\n{answer}")