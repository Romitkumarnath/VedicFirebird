import os
from typing import List, Tuple
import anthropic
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ClaudeAstrologyBot:
    def __init__(self, api_key: str):
        """
        Initialize the Vedic Astrology bot using Claude API
        
        Args:
            api_key (str): Anthropic API key
        """
        self.documents_dir = r"C:\Users\rnath\Desktop\Learning\Astrology Notes\Vedic Notes"
        self.client = anthropic.Client(api_key=api_key)
        self.context = self._prepare_documents()
        
    def _load_documents(self) -> List[str]:
        """Load all PDF documents from the specified directory"""
        all_texts = []
        
        # Walk through directory and process each PDF
        for root, _, files in os.walk(self.documents_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    try:
                        loader = PyPDFLoader(pdf_path)
                        pages = loader.load()
                        text = '\n'.join([page.page_content for page in pages])
                        all_texts.append(text)
                        print(f"Successfully loaded: {file}")
                    except Exception as e:
                        print(f"Error loading {file}: {str(e)}")
        
        return all_texts
    
    def _prepare_documents(self) -> str:
        """Process documents and prepare them for context"""
        documents = self._load_documents()
        
        if not documents:
            raise ValueError("No PDF documents were successfully loaded!")
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # Smaller chunks for better context management
            chunk_overlap=200,
            length_function=len
        )
        
        all_splits = []
        for doc in documents:
            splits = text_splitter.split_text(doc)
            all_splits.extend(splits)
            
        # Combine splits into a single context string
        return "\n\n---\n\n".join(all_splits)
    
    def ask_question(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Ask a question about Vedic astrology using the loaded documents
        
        Args:
            question (str): The question to ask
            chat_history (list): Optional list of previous (question, answer) tuples
            
        Returns:
            tuple: (answer, updated chat history)
        """
        if chat_history is None:
            chat_history = []
            
        # Prepare the system message with context
        system_message = f"""You are an expert in Vedic astrology. Use the following reference material to answer questions accurately. 
        If you're unsure about something, say so rather than making assumptions.
        
        Reference Material:
        {self.context}"""
        
        # Prepare conversation history
        messages = [
            {
                "role": "system",
                "content": system_message
            }
        ]
        
        # Add chat history
        for prev_q, prev_a in chat_history:
            messages.extend([
                {"role": "user", "content": prev_q},
                {"role": "assistant", "content": prev_a}
            ])
            
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Get response from Claude
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=messages
        )
        
        answer = response.content[0].text
        
        # Update chat history
        chat_history.append((question, answer))
        
        return answer, chat_history

def main():
    # Get API key from user
    api_key = input("Please enter your Anthropic API key: ")
    
    try:
        # Initialize bot
        print("Initializing bot and loading documents...")
        bot = ClaudeAstrologyBot(api_key)
        print("Bot initialized successfully!")
        
        # Interactive loop
        chat_history = []
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            answer, chat_history = bot.ask_question(question, chat_history)
            print(f"\nAnswer: {answer}\n")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()