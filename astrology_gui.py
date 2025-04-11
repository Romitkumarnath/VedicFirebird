import os
import gradio as gr
from astrology_bot import ClaudeAstrologyBot, TokenCounter
from dotenv import load_dotenv
import time
import traceback

# Load environment variables (for API key)
load_dotenv()

class AstrologyGUI:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.bot = None
        self.is_initialized = False
        
    def initialize_bot(self, api_key, documents_dir, progress=gr.Progress()):
        """Initialize the bot with the provided API key and documents directory"""
        try:
            progress(0.1, desc="Initializing...")
            self.api_key = api_key
            
            # Update progress
            progress(0.3, desc="Loading embedding model...")
            progress(0.5, desc="Setting up vector database...")
            
            # Initialize the bot
            self.bot = ClaudeAstrologyBot(api_key)
            
            # Set the documents directory if it's different from default
            if documents_dir and documents_dir != self.bot.documents_dir:
                self.bot.documents_dir = documents_dir
                self.bot.doc_manager.documents_dir = documents_dir
                
                # Reload documents with new directory
                progress(0.7, desc="Processing documents...")
                self.bot.reload_documents()
            
            progress(0.9, desc="Finalizing setup...")
            time.sleep(1)  # Small delay to show progress
            self.is_initialized = True
            
            # Get document statistics
            doc_count = len(self.bot.doc_manager.get_documents_metadata())
            chunk_count = self.bot.collection.count()
            
            progress(1.0, desc="Ready!")
            return f"✅ Bot initialized successfully!\n\nLoaded {doc_count} documents with {chunk_count} searchable chunks."
        except Exception as e:
            traceback.print_exc()
            return f"❌ Error initializing bot: {str(e)}"
    
    def ask_question(self, question, chat_history, progress=gr.Progress()):
        """Ask a question to the bot"""
        if not self.is_initialized:
            return chat_history + [[question, "⚠️ Bot not initialized. Please provide your API key and initialize first."]]
        
        progress(0.3, desc="Finding relevant context...")
        
        try:
            # Get answer from bot
            progress(0.7, desc="Generating response...")
            answer = self.bot.ask_question(question)
            
            # Add to chat history
            chat_history.append([question, answer])
            
            progress(1.0, desc="Complete!")
            return chat_history
        except Exception as e:
            traceback.print_exc()
            error_message = f"❌ Error: {str(e)}"
            return chat_history + [[question, error_message]]
    
    def show_document_status(self):
        """Show the status of loaded documents"""
        if not self.is_initialized:
            return "⚠️ Bot not initialized. Please provide your API key and initialize first."
        
        try:
            metadata = self.bot.doc_manager.get_documents_metadata()
            chunk_count = self.bot.collection.count()
            
            status = f"=== Document Status ===\n"
            status += f"Total documents: {len(metadata)}\n"
            status += f"Total chunks in vector database: {chunk_count}\n\n"
            
            for i, (rel_path, meta) in enumerate(metadata.items(), 1):
                last_modified = meta['last_modified']
                hash_prefix = meta['hash'][:8]
                status += f"{i}. {rel_path}\n"
                status += f"   - Last modified: {time.ctime(last_modified)}\n"
                status += f"   - File hash: {hash_prefix}...\n"
            
            return status
        except Exception as e:
            traceback.print_exc()
            return f"❌ Error getting document status: {str(e)}"
    
    def reload_documents(self, progress=gr.Progress()):
        """Reload all documents"""
        if not self.is_initialized:
            return "⚠️ Bot not initialized. Please provide your API key and initialize first."
        
        try:
            progress(0.2, desc="Starting document reload...")
            self.bot.reload_documents()
            
            # Get updated stats
            metadata = self.bot.doc_manager.get_documents_metadata()
            chunk_count = self.bot.collection.count()
            
            progress(1.0, desc="Complete!")
            return f"✅ Documents reloaded successfully!\n\nLoaded {len(metadata)} documents with {chunk_count} searchable chunks."
        except Exception as e:
            traceback.print_exc()
            return f"❌ Error reloading documents: {str(e)}"
    
    def clear_chat_history(self):
        """Clear the chat history"""
        if not self.is_initialized:
            return [], "⚠️ Bot not initialized. Please provide your API key and initialize first."
        
        try:
            self.bot.chat_history = []
            self.bot._save_chat_history()
            return [], "✅ Chat history cleared!"
        except Exception as e:
            traceback.print_exc()
            return [], f"❌ Error clearing chat history: {str(e)}"

# Create the Gradio interface
def create_interface():
    astrology_gui = AstrologyGUI()
    
    with gr.Blocks(title="Vedic Astrology Bot") as interface:
        gr.Markdown("# 🔮 Vedic Astrology Bot")
        gr.Markdown("An AI-powered bot that can answer questions about Vedic astrology using documents.")
        
        with gr.Tab("Setup"):
            with gr.Row():
                with gr.Column():
                    api_key_input = gr.Textbox(
                        label="Anthropic API Key", 
                        placeholder="Enter your Claude API key here...",
                        type="password",
                        value=astrology_gui.api_key
                    )
                    documents_dir = gr.Textbox(
                        label="Documents Directory", 
                        placeholder="Path to your PDF documents (leave empty for default)",
                        value=""
                    )
                    init_button = gr.Button("Initialize Bot", variant="primary")
                with gr.Column():
                    init_status = gr.Textbox(label="Status", lines=8, interactive=False)
            
            init_button.click(
                fn=astrology_gui.initialize_bot,
                inputs=[api_key_input, documents_dir],
                outputs=init_status
            )
        
        with gr.Tab("Chat"):
            chat_interface = gr.Chatbot(height=500)
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask something about Vedic astrology...",
                    scale=9
                )
                ask_button = gr.Button("Ask", variant="primary", scale=1)
            
            clear_button = gr.Button("Clear Chat History")
            
            ask_button.click(
                fn=astrology_gui.ask_question,
                inputs=[question_input, chat_interface],
                outputs=chat_interface
            )
            
            question_input.submit(
                fn=astrology_gui.ask_question,
                inputs=[question_input, chat_interface],
                outputs=chat_interface
            )
            
            clear_button.click(
                fn=astrology_gui.clear_chat_history,
                inputs=[],
                outputs=[chat_interface, question_input]
            )
        
        with gr.Tab("Document Management"):
            with gr.Row():
                doc_status = gr.Textbox(label="Document Status", lines=15, interactive=False)
                
            with gr.Row():
                status_button = gr.Button("Show Document Status")
                reload_button = gr.Button("Reload All Documents", variant="secondary")
            
            status_button.click(
                fn=astrology_gui.show_document_status,
                inputs=[],
                outputs=doc_status
            )
            
            reload_button.click(
                fn=astrology_gui.reload_documents,
                inputs=[],
                outputs=doc_status
            )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False) 