import os
import re
import platform
import logging
import subprocess
import psycopg2
import psycopg2.extras
from tqdm import tqdm
from typing import List
from psycopg2 import sql
from dotenv import load_dotenv
from langchain.schema import Document
from transformers import GPT2TokenizerFast
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
EMBEDDING_SIZE = 1536
COLLECTION_NAME = "kali_linux_docs"

def get_all_commands():
    result = subprocess.run(['bash', '-c', 'compgen -c'], capture_output=True, text=True)
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

def improved_strict_text_chunker(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    # Pre-split the text into smaller pieces to avoid tokenizer limitations
    pre_chunks = re.split(r'(\n\n|\n(?=[A-Z]))', text)
    pre_chunks = [chunk for chunk in pre_chunks if chunk.strip()]
    
    final_chunks = []
    current_chunk = ""
    current_tokens = []
    
    for pre_chunk in pre_chunks:
        chunk_tokens = tokenizer.encode(pre_chunk)
        
        if len(current_tokens) + len(chunk_tokens) > max_chunk_size:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            current_chunk = pre_chunk
            current_tokens = chunk_tokens
        else:
            current_chunk += " " + pre_chunk
            current_tokens.extend(chunk_tokens)
        
        # If the current chunk is getting too long, split it
        while len(current_tokens) > max_chunk_size:
            split_point = max_chunk_size - overlap
            final_chunks.append(tokenizer.decode(current_tokens[:split_point]).strip())
            current_tokens = current_tokens[split_point:]
            current_chunk = tokenizer.decode(current_tokens)
    
    if current_chunk:
        final_chunks.append(current_chunk.strip())
    
    return final_chunks

def check_and_update_schema():
    conn = psycopg2.connect(**DB_PARAMS)
    try:
        with conn.cursor() as cur:
            # Check if the table exists
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{COLLECTION_NAME}'
                )
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                print(f"Table {COLLECTION_NAME} does not exist. Creating it now.")
                initialize_db()
            else:
                # Check the embedding vector size
                cur.execute(f"""
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = '{COLLECTION_NAME}'::regclass
                    AND attname = 'embedding'
                """)
                current_size = cur.fetchone()[0] - 4  # Postgres stores it as size+4
                
                if current_size != EMBEDDING_SIZE:
                    print(f"Updating embedding size from {current_size} to {EMBEDDING_SIZE}")
                    cur.execute(f"""
                        ALTER TABLE {COLLECTION_NAME}
                        ALTER COLUMN embedding TYPE vector({EMBEDDING_SIZE})
                    """)
                    conn.commit()
                    print("Embedding size updated successfully.")
                else:
                    print("Schema is up to date.")
    except psycopg2.Error as e:
        print(f"Error checking/updating schema: {e}")
        conn.rollback()
    finally:
        conn.close()

def create_vector_store():
    embeddings = OpenAIEmbeddings()
    vector_store = PGVector.from_documents(
        embedding=embeddings,
        documents=[],
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME
    )

    commands = get_all_commands()
    batch_size = 100
    batch_documents = []

    for cmd in tqdm(commands, desc="Processing commands"):
        try:
            help_output = get_command_help(cmd)
            chunks = improved_strict_text_chunker(help_output, max_chunk_size=500, overlap=50)
            
            logger.info(f"Processing command: {cmd}")
            logger.info(f"Number of chunks: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
                chunk_size = len(tokenizer.encode(chunk))
                logger.info(f"Chunk {i+1} size: {chunk_size}")
                
                doc = Document(page_content=chunk, metadata={"command": cmd, "chunk": i+1})
                batch_documents.append(doc)

                if len(batch_documents) >= batch_size:
                    vector_store.add_documents(batch_documents)
                    batch_documents = []

        except Exception as e:
            logger.error(f"Error processing {cmd}: {str(e)}")

    # Add any remaining documents
    if batch_documents:
        vector_store.add_documents(batch_documents)

    return vector_store

def query_vector_store(query: str, vector_store: PGVector, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return "\n".join([f"Command: {doc.metadata['command']}\n{doc.page_content}" for doc in results])

def get_shell_name():
    return os.path.basename(os.environ.get('SHELL', 'bash'))

def get_os_name():
    return "Kali Linux"

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that generates Kali Linux commands.
Ensure the commands are for {os_name} using {shell_name}.
Here is the conversation so far:
{history}
Additional command information:
{command_info}
Human: {question}
Assistant: Let me help you with that. Here's a command that should work for your request:
""")

# Initialize memory
memory = ConversationBufferMemory(return_messages=True, input_key="question", output_key="answer", memory_key="history")

# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create the chain using LCEL
chain = (
    prompt_template 
    | llm 
    | StrOutputParser()
)

# To use the chain with memory
def invoke_chain_with_memory(question, os_name, shell_name, command_info):
    history = memory.load_memory_variables({})["history"]
    response = chain.invoke({
        "question": question,
        "os_name": os_name,
        "shell_name": shell_name,
        "command_info": command_info,
        "history": history
    })
    memory.save_context({"question": question}, {"answer": response})
    return response

# Main loop
def main():
    print("Welcome to Kali Linux Command Assistant! Type 'exit' to quit.")
    
    # Create or load vector store
    vector_store = create_vector_store()
    
    while True:
        question = input("Ask a Kali Linux command question: ")
        if question.lower() == 'exit':
            break
        
        os_name = get_os_name()
        shell_name = get_shell_name()
        
        # Query vector store for relevant command info
        command_info = query_vector_store(question, vector_store)
        
        # Generate response
        response = invoke_chain_with_memory(
            question=question, 
            os_name=os_name, 
            shell_name=shell_name, 
            command_info=command_info
        )
        
        print("Generated Command(s):")
        print(response)
        
        # Option to execute the command
        if input("Do you want to execute this command? (y/n): ").lower() == 'y':
            try:
                result = subprocess.run(response, shell=True, check=True, text=True, capture_output=True)
                print("Command output:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_and_update_schema()
    main()
