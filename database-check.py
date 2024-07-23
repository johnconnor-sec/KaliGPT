import os
import subprocess
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Database configuration
DB_PARAMS = {
    "dbname": "langchain",
    "user": "langchain",
    "password": "langchain",
    "host": "localhost",
    "port": "6024"
}
CONNECTION_STRING = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"

# Embedding configuration
EMBEDDING_SIZE = 768
COLLECTION_NAME = "kali_linux_docs"

# Initialize OpenAI API (replace with your key)
llm = ChatOpenAI()

def create_db_if_not_exists():
    conn = psycopg2.connect(dbname="postgres", **{k: v for k, v in DB_PARAMS.items() if k != "dbname"})
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_PARAMS["dbname"],))
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_PARAMS["dbname"])))
    conn.close()

def initialize_db():
    create_db_if_not_exists()
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {COLLECTION_NAME} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({EMBEDDING_SIZE})
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {COLLECTION_NAME}_embedding_idx 
                ON {COLLECTION_NAME} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
        conn.commit()
        print("Database initialized successfully.")
    except psycopg2.Error as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_all_commands():
    result = subprocess.run(['compgen', '-c'], capture_output=True, text=True, shell=True)
    return sorted(set(result.stdout.split()))

def get_command_help(cmd):
    try:
        man_output = subprocess.run(['man', cmd], capture_output=True, text=True, timeout=5)
        if man_output.returncode == 0:
            return man_output.stdout
        
        help_output = subprocess.run([cmd, '--help'], capture_output=True, text=True, timeout=5)
        if help_output.returncode == 0:
            return help_output.stdout
        
        return f"No help available for {cmd}"
    except subprocess.TimeoutExpired:
        return f"Command {cmd} timed out when trying to get help"
    except Exception as e:
        return f"Error getting help for {cmd}: {str(e)}"

def create_vector_store():
    embeddings = OpenAIEmbeddings()
    vector_store = PGVector.from_documents(
        embedding=embeddings,
        documents=[],
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME
    )

    commands = get_all_commands()
    for cmd in tqdm(commands, desc="Processing commands"):
        try:
            help_output = get_command_help(cmd)
            doc = Document(page_content=help_output, metadata={"command": cmd})
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents([doc])
            vector_store.add_documents(split_docs)
        except Exception as e:
            print(f"Error processing {cmd}: {str(e)}")

    return vector_store

def query_vector_store(query: str, vector_store: PGVector, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([f"Command: {doc.metadata['command']}\n{doc.page_content}" for doc in results])

def main():
    initialize_db()
    vector_store = create_vector_store()
    
    print("Kali Linux RAG System initialized. You can now query the system.")
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = query_vector_store(query, vector_store)
        print("\nRelevant Command Information:")
        print(results)

if __name__ == "__main__":
    main()
