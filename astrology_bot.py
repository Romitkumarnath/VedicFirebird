import os
import pickle
import json
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import tiktoken

class DocumentManager:
    def __init__(self, documents_dir: str):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.documents_dir = documents_dir
        self.cache_dir = os.path.join(script_dir, "cache")
        self.document_cache_file = os.path.join(self.cache_dir, "documents_cache.pkl")
        self.metadata_file = os.path.join(self.cache_dir, "documents_metadata.json")
        self.chroma_dir = os.path.join(self.cache_dir, "chroma_db")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            print(f"Creating cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir)
            
        if not os.path.exists(self.chroma_dir):
            os.makedirs(self.chroma_dir)
    
    def get_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_documents_metadata(self) -> Dict:
        """Get metadata of all PDF files in directory"""
        metadata = {}
        print(f"\nScanning directory: {self.documents_dir}")
        for root, _, files in os.walk(self.documents_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    filepath = os.path.join(root, file)
                    relative_path = os.path.relpath(filepath, self.documents_dir)
                    print(f"Found PDF: {relative_path}")
                    metadata[relative_path] = {
                        'path': filepath,
                        'hash': self.get_file_hash(filepath),
                        'last_modified': os.path.getmtime(filepath)
                    }
        return metadata

    def load_cache_metadata(self) -> Dict:
        """Load cached metadata"""
        if os.path.exists(self.metadata_file):
            print(f"Loading cache metadata from: {self.metadata_file}")
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        print("No cache metadata found")
        return {}

    def save_cache_metadata(self, metadata: Dict):
        """Save metadata to cache"""
        print(f"Saving metadata to: {self.metadata_file}")
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

class TokenCounter:
    def __init__(self, model: str = "cl100k_base"):
        self.model = model
        self.encoder = tiktoken.get_encoding(model)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.encoder.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens])

class ClaudeAstrologyBot:
    def __init__(self, api_key: str):
        """Initialize the Vedic Astrology bot using Claude API"""
        self.documents_dir = r"C:\Users\rnath\Desktop\Learning\Astrology Notes\Vedic Notes"
        self.client = anthropic.Client(api_key=api_key)
        self.doc_manager = DocumentManager(self.documents_dir)
        self.token_counter = TokenCounter()
        
        # Initialize embedding model
        print("Initializing embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        print("Initializing vector database...")
        self.chroma_client = chromadb.PersistentClient(path=self.doc_manager.chroma_dir)
        
        # Initialize or load the collection
        try:
            self.collection = self.chroma_client.get_collection(name="astrology_docs")
            print("Loaded existing vector database collection")
        except:
            print("Creating new vector database collection")
            self.collection = self.chroma_client.create_collection(name="astrology_docs")
        
        # Process documents if needed
        self._load_or_process_documents()
        
        # Load chat history
        self.chat_history_file = os.path.join(self.doc_manager.cache_dir, "chat_history.json")
        self.chat_history = self._load_chat_history()
        
        # Limits
        self.max_context_tokens = 70000  # Claude 3 Sonnet context window
        self.max_response_tokens = 1000
        self.max_history_items = 10  # Number of history items to keep
        
    def _load_document(self, filepath: str) -> Dict[str, Any]:
        """Load a single PDF document"""
        try:
            print(f"Loading document: {filepath}")
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            
            content = "\n\n".join([page.page_content for page in pages])
            source = os.path.basename(filepath)
            
            print(f"Successfully loaded document: {source}")
            return {'content': content, 'source': source}
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return {'content': '', 'source': os.path.basename(filepath)}

    def _load_or_process_documents(self) -> None:
        """Load documents with selective reloading"""
        print("\n=== Starting document loading process ===")
        
        # Get current documents metadata
        current_metadata = self.doc_manager.get_documents_metadata()
        cached_metadata = self.doc_manager.load_cache_metadata()
        
        # Check if metadata matches - if not, reprocess documents
        if self._metadata_matches(current_metadata, cached_metadata) and self.collection.count() > 0:
            print("No document changes detected. Using existing vector database.")
            return
        
        # Process documents and update vector store
        print("Processing documents and updating vector database...")
        self._process_documents(current_metadata)
        
        # Save metadata
        self.doc_manager.save_cache_metadata(current_metadata)
    
    def _metadata_matches(self, current: Dict, cached: Dict) -> bool:
        """Check if current metadata matches cached metadata"""
        if len(current) != len(cached):
            return False
        
        for key, meta in current.items():
            if key not in cached or meta['hash'] != cached[key]['hash']:
                return False
        
        return True
    
    def _process_documents(self, metadata: Dict) -> None:
        """Process documents into chunks and add to vector database"""
        all_docs = []
        
        # Load all documents
        for rel_path, meta in metadata.items():
            doc = self._load_document(meta['path'])
            if doc['content']:
                all_docs.append(doc)
        
        if not all_docs:
            raise ValueError("No PDF documents were successfully loaded!")
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for better retrieval
            chunk_overlap=100,
            length_function=len
        )
        
        # Clear existing collection
        print("Clearing existing vector database...")
        try:
            self.collection.delete(where={})
        except:
            pass
        
        # Process documents
        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_id = 0
        
        print("Processing document chunks...")
        for doc in all_docs:
            doc_chunks = text_splitter.split_text(doc['content'])
            for chunk in doc_chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"source": doc['source']})
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
                
                # Add in batches to avoid memory issues
                if len(all_chunks) >= 100:
                    # Generate embeddings via sentence-transformers
                    print(f"Adding batch of {len(all_chunks)} chunks to vector database...")
                    self.collection.add(
                        documents=all_chunks,
                        metadatas=all_metadatas,
                        ids=all_ids
                    )
                    all_chunks = []
                    all_metadatas = []
                    all_ids = []
        
        # Add any remaining chunks
        if all_chunks:
            print(f"Adding final batch of {len(all_chunks)} chunks to vector database...")
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
        
        print(f"Added {chunk_id} document chunks to vector database")
    
    def _load_chat_history(self) -> List[Dict[str, str]]:
        """Load chat history from file"""
        if os.path.exists(self.chat_history_file):
            try:
                with open(self.chat_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading chat history: {str(e)}")
        return []
    
    def _save_chat_history(self):
        """Save chat history to file"""
        with open(self.chat_history_file, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
    
    def get_relevant_context(self, question: str, max_tokens: int = 7000) -> Tuple[str, List[str]]:
        """Get relevant context using vector similarity search"""
        # Convert question to query embedding using the model
        query_embedding = self.embedder.encode(question).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"]
        )
        
        # Extract documents and sources
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        sources = list(set(meta.get('source', 'Unknown') for meta in metadatas))
        
        # Combine document content
        context = "\n\n---\n\n".join(documents)
        
        # Truncate context if too long
        if self.token_counter.count_tokens(context) > max_tokens:
            context = self.token_counter.truncate_text(context, max_tokens)
            
        return context, sources
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question about Vedic astrology using the loaded documents
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The answer
        """
        # Get relevant context
        context, sources = self.get_relevant_context(question)
        print(f"Retrieved context from {len(sources)} sources: {', '.join(sources)}")
        
        # Prepare the system message with context
        system_message = f"""You are an expert in Vedic astrology. Use the following reference material to answer questions accurately. 
        If you're unsure about something, say so rather than making assumptions.
        
        Reference Material:
        {context}"""
        
        # Prepare conversation history - only use last N items
        formatted_history = []
        recent_history = self.chat_history[-self.max_history_items:] if self.chat_history else []
        
        # Count tokens in history
        history_tokens = 0
        for item in recent_history:
            item_tokens = self.token_counter.count_tokens(item["question"]) + self.token_counter.count_tokens(item["answer"])
            if history_tokens + item_tokens > 10000:  # Limit history tokens
                break
            history_tokens += item_tokens
            formatted_history.append({"role": "user", "content": item["question"]})
            formatted_history.append({"role": "assistant", "content": item["answer"]})
        
        # Add current question
        formatted_history.append({"role": "user", "content": question})
        
        # Get response from Claude
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=self.max_response_tokens,
            system=system_message,
            messages=formatted_history
        )
        
        answer = response.content[0].text
        
        # Update chat history
        self.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim chat history if needed
        if len(self.chat_history) > 100:  # Keep last 100 for long-term storage
            self.chat_history = self.chat_history[-100:]
        
        # Save chat history
        self._save_chat_history()
        
        return answer
    
    def show_document_status(self):
        """Show status of documents in the system"""
        metadata = self.doc_manager.get_documents_metadata()
        
        print("\n=== Document Status ===")
        print(f"Total documents: {len(metadata)}")
        print(f"Total chunks in vector database: {self.collection.count()}")
        
        for i, (rel_path, meta) in enumerate(metadata.items(), 1):
            print(f"{i}. {rel_path}")
            print(f"   - Last modified: {datetime.fromtimestamp(meta['last_modified']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   - File hash: {meta['hash'][:8]}...")
        
        print("\n")
    
    def reload_documents(self):
        """Force reload all documents"""
        print("\n=== Reloading all documents ===")
        
        # Get current documents metadata
        metadata = self.doc_manager.get_documents_metadata()
        
        # Process and update vector store
        self._process_documents(metadata)
        
        # Save updated metadata
        self.doc_manager.save_cache_metadata(metadata)
        
        print("Documents reloaded successfully!")

def main():
    print("\n=== Vedic Astrology Bot Starting ===")
    print("This bot uses Claude AI and your PDF documents to answer questions about Vedic astrology.")
    
    try:
        api_key = input("Please enter your Anthropic API key: ")
        print("Initializing bot with the provided API key...")
        
        print("Creating ClaudeAstrologyBot instance...")
        bot = ClaudeAstrologyBot(api_key)
        print("ClaudeAstrologyBot initialized successfully!")
        
        while True:
            print("\nOptions:")
            print("1. Ask a question")
            print("2. Show document status")
            print("3. Reload all documents")
            print("4. Clear chat history")
            print("5. Quit")
            
            choice = input("\nEnter your choice (1-5): ")
            print(f"Selected option: {choice}")
            
            if choice == "1":
                question = input("\nEnter your question: ")
                print("\nThinking...")
                answer = bot.ask_question(question)
                print(f"\nAnswer: {answer}\n")
            
            elif choice == "2":
                bot.show_document_status()
            
            elif choice == "3":
                bot.reload_documents()
            
            elif choice == "4":
                bot.chat_history = []
                bot._save_chat_history()
                print("Chat history cleared!")
            
            elif choice == "5":
                break
            
            else:
                print("Invalid choice! Please try again.")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Print the full stack trace for debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()