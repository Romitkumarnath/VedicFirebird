import os
import pickle
from typing import List, Tuple, Dict
import anthropic
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime
import hashlib

class DocumentManager:
    def __init__(self, documents_dir: str):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.documents_dir = documents_dir
        self.cache_dir = os.path.join(script_dir, "cache")
        self.document_cache_file = os.path.join(self.cache_dir, "documents_cache.pkl")
        self.metadata_file = os.path.join(self.cache_dir, "documents_metadata.json")
        
        print(f"\nCache directory: {self.cache_dir}")
        print(f"Cache file: {self.document_cache_file}")
        print(f"Metadata file: {self.metadata_file}")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            print(f"Creating cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir)
    
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
                    print(f"Found PDF: {file}")
                    metadata[file] = {
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

class ClaudeAstrologyBot:
    def __init__(self, api_key: str):
        """Initialize the Vedic Astrology bot using Claude API"""
        self.documents_dir = r"C:\Users\rnath\Desktop\Learning\Astrology Notes\Vedic Notes"
        self.client = anthropic.Client(api_key=api_key)
        self.doc_manager = DocumentManager(self.documents_dir)
        self.context = self._load_or_process_documents()

    def _load_document(self, filepath: str) -> str:
        """Load a single PDF document"""
        try:
            print(f"Loading document: {filepath}")
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            text = '\n'.join([page.page_content for page in pages])
            print(f"Successfully loaded document: {os.path.basename(filepath)}")
            return text
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return ""

    def _load_or_process_documents(self) -> str:
        """Load documents with selective reloading"""
        print("\n=== Starting document loading process ===")
        
        # Get current documents metadata
        current_metadata = self.doc_manager.get_documents_metadata()
        cached_metadata = self.doc_manager.load_cache_metadata()
        
        documents_text = {}
        changes_made = False
        
        # Check if cache exists
        if os.path.exists(self.doc_manager.document_cache_file):
            print(f"Found existing cache: {self.doc_manager.document_cache_file}")
            try:
                with open(self.doc_manager.document_cache_file, 'rb') as f:
                    documents_text = pickle.load(f)
                print("Successfully loaded cache")
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
                documents_text = {}
        else:
            print("No cache file found")
        
        # Process each document
        for filename, meta in current_metadata.items():
            reload_needed = True
            if filename in cached_metadata:
                if meta['hash'] == cached_metadata[filename]['hash']:
                    if filename in documents_text:
                        print(f"Using cached version of: {filename}")
                        reload_needed = False
            
            if reload_needed:
                print(f"Loading and processing: {filename}")
                documents_text[filename] = self._load_document(meta['path'])
                changes_made = True
        
        # If changes were made, update cache
        if changes_made:
            print("\nUpdating cache...")
            try:
                with open(self.doc_manager.document_cache_file, 'wb') as f:
                    pickle.dump(documents_text, f)
                print("Cache updated successfully")
                self.doc_manager.save_cache_metadata(current_metadata)
            except Exception as e:
                print(f"Error updating cache: {str(e)}")
        else:
            print("\nNo changes detected, using cached documents")
        
        # Combine all documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
        
        all_splits = []
        for doc_text in documents_text.values():
            splits = text_splitter.split_text(doc_text)
            all_splits.extend(splits)
        
        return "\n\n---\n\n".join(all_splits)

    # ... [rest of the code remains the same] ...

def main():
    print("\n=== Vedic Astrology Bot Starting ===")
    api_key = input("Please enter your Anthropic API key: ")
    
    try:
        bot = ClaudeAstrologyBot(api_key)
        chat_history = []
        
        while True:
            print("\nOptions:")
            print("1. Ask a question")
            print("2. Show document status")
            print("3. Reload specific documents")
            print("4. Quit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == "1":
                question = input("\nEnter your question: ")
                answer, chat_history = bot.ask_question(question, chat_history)
                print(f"\nAnswer: {answer}\n")
            
            elif choice == "2":
                bot.show_document_status()
            
            elif choice == "3":
                bot.show_document_status()
                files = input("\nEnter filenames to reload (comma-separated): ").split(",")
                files = [f.strip() for f in files if f.strip()]
                if files:
                    bot.reload_specific_documents(files)
            
            elif choice == "4":
                break
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()