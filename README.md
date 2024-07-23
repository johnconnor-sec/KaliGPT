# Kali Command Assistant

## Overview

Kali Linux Command Assistant is an intelligent, interactive tool designed to help users navigate and utilize Kali Linux commands more effectively. This project leverages natural language processing and a vector database to provide accurate, context-aware command suggestions and explanations for Kali Linux users.

## Features

- Natural language interface for querying Kali Linux commands
- Intelligent command suggestion based on user input
- Detailed explanations and usage examples for Kali Linux commands
- Vector database for efficient storage and retrieval of command information
- Option to execute suggested commands directly from the interface

## Technology Stack

- Python 3.8+
- LangChain for natural language processing
- OpenAI's GPT-4o-mini for language model
- PostgreSQL with pgvector extension for vector storage
- Transformers library for tokenization

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- PostgreSQL 12 or higher with pgvector extension installed
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/kali-linux-command-assistant.git
   cd kali-linux-command-assistant
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Initialize the database:
   
   `docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16`

    You can connect to the database using the psql command-line tool:
    `psql -h localhost -p 6024 -U langchain -d langchain`
    You'll be prompted for the password (which should be "langchain" based on your Docker setup).
    
    Once inside the pgvector database:
    `CREATE EXTENSION vector`
    
    Verify that the Docker container's logs don't show any errors:
    `docker logs pgvector-container`
    
    If all else fails, you might want to recreate the Docker container:
    ```shell
    docker stop pgvector-container
    docker rm pgvector-container
    docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
    ```
    
## Usage

To start the Kali Linux Command Assistant, run:

```
python main.py
```

Follow the on-screen prompts to interact with the assistant. You can ask questions about Kali Linux commands, and the assistant will provide relevant suggestions and explanations.

Example interaction:

```
Welcome to Kali Linux Command Assistant! Type 'exit' to quit.
Ask a Kali Linux command question: How do I scan for open ports on a network?

Generated Command(s):
To scan for open ports on a network using Kali Linux, you can use the nmap command. Here's an example:

nmap -sS -O 192.168.1.0/24

This command will perform a SYN scan (-sS) and try to detect the operating system (-O) of all hosts in the 192.168.1.0/24 network range.

Do you want to execute this command? (y/n): n

Ask a Kali Linux command question: exit

Goodbye!
```



## Contributing

Contributions to the Kali Linux Command Assistant are welcome. Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

## Acknowledgements

- OpenAI for providing the GPT-4 model
- The LangChain community for their excellent framework
- The Kali Linux team for their comprehensive command documentation

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.
