import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import anthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from enum import Enum
import pandas as pd
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

class AstrologySystem(Enum):
    PARASHARI = "parashari"

class DocumentCategory(Enum):
    PLANETS = "planets"
    HOUSES = "houses"
    DASHAS = "dashas"
    YOGAS = "yogas"
    COMBINATIONS = "combinations"
    PREDICTIONS = "predictions"
    REMEDIES = "remedies"

class PredictionRecord:
    def __init__(self, question: str, prediction: str, confidence: float, 
                 sources: List[str], date_made: datetime,
                 feedback: str = None, accuracy: float = None, notes: str = None):
        self.question = question
        self.prediction = prediction
        self.confidence = confidence
        self.sources = sources
        self.date_made = date_made if isinstance(date_made, datetime) else datetime.fromisoformat(date_made)
        self.feedback = feedback
        self.accuracy = accuracy
        self.notes = notes

    def to_dict(self):
        return {
            "question": self.question,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "sources": self.sources,
            "date_made": self.date_made.isoformat(),
            "feedback": self.feedback,
            "accuracy": self.accuracy,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data):
        # Convert old format records if they exist
        if 'systems_used' in data:
            del data['systems_used']
        return cls(**data)

class ProfessionalAstrologyBot:
    def __init__(self, api_key: str):
        self.setup_directories()
        self.client = anthropic.Client(api_key=api_key)
        self.embeddings = self.initialize_embeddings()
        self.document_registry = self.load_document_registry()
        self.vectorstores = {}
        self.vectorstores = self.initialize_vectorstores()
        self.prediction_history = self.load_prediction_history()

    def setup_directories(self):
        """Setup directory structure for documents and caches"""
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.docs_dir = os.path.join(self.base_dir, "astrology_docs")
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.predictions_file = os.path.join(self.base_dir, "prediction_history.json")
        self.registry_file = os.path.join(self.cache_dir, "document_registry.json")
        self.vedic_notes_dir = r"C:\Users\rnath\Desktop\Learning\Astrology Notes\Vedic Notes"
        
        # Create category subdirectories for Parashari only
        for category in DocumentCategory:
            path = os.path.join(self.docs_dir, category.value, AstrologySystem.PARASHARI.value)
            os.makedirs(path, exist_ok=True)
                
        os.makedirs(self.cache_dir, exist_ok=True)

    def initialize_embeddings(self):
        """Initialize the embedding model"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

    def load_document_registry(self) -> Dict:
        """Load registry of processed documents"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}

    def save_document_registry(self):
        """Save registry of processed documents"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.document_registry, f, indent=2)

    def get_file_hash(self, filepath: str) -> str:
        """Calculate hash of file to detect changes"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _check_for_updates(self) -> bool:
        """Check if any documents need processing"""
        pdf_files = [f for f in os.listdir(self.vedic_notes_dir) if f.endswith('.pdf')]
        
        for file in pdf_files:
            filepath = os.path.join(self.vedic_notes_dir, file)
            current_hash = self.get_file_hash(filepath)
            
            if (file not in self.document_registry or 
                self.document_registry[file]['hash'] != current_hash):
                return True
        
        for file in list(self.document_registry.keys()):
            if not os.path.exists(os.path.join(self.vedic_notes_dir, file)):
                return True
        
        return False

    def initialize_vectorstores(self) -> Dict[str, FAISS]:
        """Initialize vector stores"""
        vectorstores = {}
        cache_path = os.path.join(self.cache_dir, "faiss_parashari")
        
        if os.path.exists(cache_path):
            try:
                vectorstores['parashari'] = FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)
                print("Loaded existing vector database")
                
                needs_update = self._check_for_updates()
                if not needs_update:
                    return vectorstores
                
            except Exception as e:
                print(f"Error loading vector database: {e}")
                print("Will rebuild the database...")
        
        return self.process_documents()

    def process_documents(self) -> Dict[str, FAISS]:
        """Process all documents and create vector store"""
        print("\nProcessing documents...")
        vectorstores = {}
        texts = []
        batch_size = 100
        
        pdf_files = [f for f in os.listdir(self.vedic_notes_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found!")
            return vectorstores
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        processed_files = set()
        
        for file in pdf_files:
            filepath = os.path.join(self.vedic_notes_dir, file)
            current_hash = self.get_file_hash(filepath)
            
            if (file in self.document_registry and 
                self.document_registry[file]['hash'] == current_hash):
                processed_files.add(file)
                continue
                
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                for page in pages:
                    page.metadata['source'] = file
                texts.extend(pages)
                print(f"Processed: {file}")
                
                processed_files.add(file)
                self.document_registry[file] = {
                    'hash': current_hash,
                    'last_processed': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        for file in list(self.document_registry.keys()):
            if file not in processed_files:
                del self.document_registry[file]
        
        if texts:
            print(f"\nSplitting documents into chunks...")
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", "!", "?", ";"],
                    length_function=len
                )
                
                chunks = splitter.split_documents(texts)
                total_chunks = len(chunks)
                print(f"Created {total_chunks} chunks from documents")
                
                print("\nGenerating embeddings and creating vector store...")
                processed_chunks = 0
                vectorstore = None
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, self.embeddings)
                    else:
                        batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                        vectorstore.merge_from(batch_vectorstore)
                    
                    processed_chunks += len(batch)
                    progress = (processed_chunks / total_chunks) * 100
                    print(f"Progress: {processed_chunks}/{total_chunks} chunks ({progress:.1f}%)")
                
                vectorstores['parashari'] = vectorstore
                
                print("\nSaving vector database...")
                cache_path = os.path.join(self.cache_dir, "faiss_parashari")
                vectorstore.save_local(cache_path)
                print("Vector database saved successfully!")
                
            except Exception as e:
                print(f"Error during document processing: {e}")
                return vectorstores
        
        self.save_document_registry()
        return vectorstores

    def refresh_documents(self, force_refresh: bool = False) -> Dict[str, FAISS]:
        """Process only new or modified documents, or force refresh all documents"""
        try:
            print("\nChecking for document changes...")
            if force_refresh:
                print("Force refresh requested - processing all documents...")
                # Clear the document registry to force reprocessing
                self.document_registry = {}
                result = self.process_documents()
                if result:
                    print("All documents have been reprocessed successfully!")
                    return result
            elif self._check_for_updates():
                result = self.process_documents()
                if result:
                    print("Documents refreshed successfully!")
                    return result
            else:
                print("No document changes detected. Using existing database.")
                return self.vectorstores or {}
        except Exception as e:
            print(f"Error refreshing documents: {e}")
            return self.vectorstores or {}

    def get_relevant_context(self, question: str) -> Tuple[str, List[str], float]:
        """Get relevant context from the vector store"""
        if 'parashari' not in self.vectorstores:
            return "", [], 0.5
            
        docs = self.vectorstores['parashari'].max_marginal_relevance_search(
            question,
            k=5,
            fetch_k=20,
            lambda_mult=0.7
        )
        
        sources = list(set(doc.metadata['source'] for doc in docs))
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        
        confidence = min(len(docs) / 5, 1.0)
        
        return context, sources, confidence

    def load_prediction_history(self) -> List[PredictionRecord]:
        """Load prediction history from file"""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, 'r') as f:
                data = json.load(f)
                return [PredictionRecord.from_dict(record) for record in data]
        return []

    def save_prediction_history(self):
        """Save prediction history to file"""
        with open(self.predictions_file, 'w') as f:
            json.dump([record.to_dict() for record in self.prediction_history], f, indent=2)

    def make_prediction(self, question: str) -> PredictionRecord:
        """Make an astrological prediction based on the question"""
        context, sources, confidence = self.get_relevant_context(question)
        
        system_prompt = f"""You are an expert Vedic astrologer specializing in Parashari astrology. 
        Use the following reference material to make a detailed prediction based on Parashari principles. 
        If the reference material doesn't contain enough information for a confident prediction, 
        say so and explain what additional information would be needed.

        Reference Material:
        {context}"""

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": question}]
        )

        prediction = response.content[0].text
        
        record = PredictionRecord(
            question=question,
            prediction=prediction,
            confidence=confidence,
            sources=sources,
            date_made=datetime.now()
        )
        
        self.prediction_history.append(record)
        self.save_prediction_history()
        
        return record

    def update_prediction_feedback(self, prediction_index: int, feedback: str, 
                                 accuracy: float = None, notes: str = None):
        """Update feedback and accuracy for a prediction"""
        if 0 <= prediction_index < len(self.prediction_history):
            record = self.prediction_history[prediction_index]
            record.feedback = feedback
            record.accuracy = accuracy
            record.notes = notes
            self.save_prediction_history()

    def analyze_prediction_accuracy(self) -> pd.DataFrame:
        """Analyze prediction accuracy and patterns"""
        records = [record.to_dict() for record in self.prediction_history if record.accuracy is not None]
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        analysis = {
            'overall_accuracy': df['accuracy'].mean(),
            'confidence_correlation': df['confidence'].corr(df['accuracy']),
            'accuracy_over_time': df.groupby(pd.to_datetime(df['date_made']).dt.month)['accuracy'].mean()
        }
        
        return pd.DataFrame(analysis)

def main():
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    try:
        bot = ProfessionalAstrologyBot(api_key)
        print("\nHello! I'm your Professional Astrology Bot. I can help you with astrological predictions based on Vedic principles.")
        print("You can chat with me naturally. Type 'help' to see available commands, or 'quit' to exit.")
        
        while True:
            user_input = input("\nYou: ").strip().lower()
            
            if user_input == 'quit':
                print("\nThank you for consulting with me. Goodbye!")
                break
                
            elif user_input == 'help':
                print("\nAvailable commands:")
                print("- Ask any astrological question")
                print("- 'history': View past predictions")
                print("- 'feedback': Update feedback for a prediction")
                print("- 'analysis': View prediction accuracy analysis")
                print("- 'refresh': Update document database")
                print("- 'force refresh': Rebuild entire document database")
                print("- 'help': Show this help message")
                print("- 'quit': Exit the program")
                
            elif user_input == 'history':
                if not bot.prediction_history:
                    print("\nNo predictions found in history.")
                else:
                    for i, record in enumerate(bot.prediction_history):
                        print(f"\nPrediction {i}:")
                        print(f"Question: {record.question}")
                        print(f"Prediction: {record.prediction[:200]}..." if len(record.prediction) > 200 else record.prediction)
                        print(f"Date: {record.date_made}")
                        print(f"Accuracy: {record.accuracy if record.accuracy is not None else 'Not rated'}")
                        if record.feedback:
                            print(f"Feedback: {record.feedback}")
                        if record.notes:
                            print(f"Notes: {record.notes}")
                            
            elif user_input == 'feedback':
                if not bot.prediction_history:
                    print("\nNo predictions to update.")
                    continue
                
                try:
                    idx = int(input("Which prediction would you like to update? (Enter number): "))
                    if idx < 0 or idx >= len(bot.prediction_history):
                        print("Invalid prediction number!")
                        continue
                    
                    feedback = input("What's your feedback? ")
                    
                    accuracy_input = input("How accurate was the prediction? (0-1, or press Enter to skip): ")
                    accuracy = None
                    if accuracy_input.strip():
                        try:
                            accuracy = float(accuracy_input)
                            if accuracy < 0 or accuracy > 1:
                                print("Accuracy must be between 0 and 1!")
                                continue
                        except ValueError:
                            print("Invalid accuracy value!")
                            continue
                    
                    notes = input("Any additional notes? ")
                    
                    bot.update_prediction_feedback(idx, feedback, accuracy, notes)
                    print("Thank you for your feedback!")
                    
                except ValueError:
                    print("Please enter a valid prediction number!")
                except Exception as e:
                    print(f"Error updating feedback: {str(e)}")
                
            elif user_input == 'analysis':
                try:
                    analysis = bot.analyze_prediction_accuracy()
                    if analysis.empty:
                        print("\nNo rated predictions available for analysis yet.")
                    else:
                        print("\nPrediction Analysis:")
                        print(f"Overall Accuracy: {analysis['overall_accuracy'].iloc[0]:.2%}")
                        print(f"Confidence-Accuracy Correlation: {analysis['confidence_correlation'].iloc[0]:.3f}")
                        print("\nAccuracy Over Time:")
                        print(analysis['accuracy_over_time'].apply(lambda x: f"{x:.2%}"))
                except Exception as e:
                    print(f"Error analyzing predictions: {str(e)}")
                
            elif user_input == 'refresh':
                try:
                    old_vectorstores = bot.vectorstores.copy()
                    bot.vectorstores = bot.refresh_documents(force_refresh=False)
                    if bot.vectorstores != old_vectorstores:
                        print("Documents have been updated successfully!")
                except Exception as e:
                    print(f"Error refreshing documents: {str(e)}")
                
            elif user_input == 'force refresh':
                try:
                    confirm = input("This will reprocess all documents. Continue? (y/n): ").lower()
                    if confirm == 'y':
                        old_vectorstores = bot.vectorstores.copy()
                        bot.vectorstores = bot.refresh_documents(force_refresh=True)
                        if bot.vectorstores != old_vectorstores:
                            print("All documents have been reprocessed successfully!")
                except Exception as e:
                    print(f"Error refreshing documents: {str(e)}")
                
            else:
                try:
                    record = bot.make_prediction(user_input)
                    print(f"\nPrediction:")
                    print(f"{record.prediction}")
                    print(f"\nConfidence: {record.confidence:.2f}")
                    print(f"Sources consulted: {', '.join(record.sources)}")
                except Exception as e:
                    print(f"I apologize, but I encountered an error: {str(e)}")
                    print("Please try rephrasing your question or type 'help' for available commands.")
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    finally:
        try:
            if 'bot' in locals() and hasattr(bot, 'save_prediction_history'):
                bot.save_prediction_history()
            if 'bot' in locals() and hasattr(bot, 'save_document_registry'):
                bot.save_document_registry()
        except Exception as e:
            print(f"Error saving data: {str(e)}")

if __name__ == "__main__":
    main()