#!/usr/bin/env python3
"""
Fixed Vedic Astrology Bot with ChromaDB Vector Database
----------------------------------------------------------
A Flask web interface with ChromaDB for vector storage with fixed embeddings
"""

import os
import json
import glob
import requests
import pickle
import time
import shutil
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import hashlib
from datetime import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Path constants
NOTES_DIR = r"C:\Users\rnath\Desktop\Learning\Astrology Notes"
CACHE_DIR = "fixed_cache"  # Use a different cache directory
CHROMA_DIR = os.path.join(CACHE_DIR, "chroma_data")
DOCUMENTS_FILE = os.path.join(CACHE_DIR, "documents.pkl")
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "embeddings.pkl")
VECTORIZER_FILE = os.path.join(CACHE_DIR, "vectorizer.pkl")

# Chunking and embedding configurations
CHUNK_SIZE = 550        # Smaller chunks for more precise matching (500-600 recommended)
CHUNK_OVERLAP = 225     # Increased overlap for better context preservation (200-250 recommended)
MIN_CHUNK_LENGTH = 100  # Keeping minimum chunk length to filter fragments
MAX_FEATURES = 4000     # Reduced to a more efficient size (3000-4000 recommended)
TOP_K_RESULTS = 12      # Increased results per query (10-15 recommended)

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR)

# Check if the notes directory exists
if not os.path.exists(NOTES_DIR):
    print(f"WARNING: Directory not found: {NOTES_DIR}")
    print("Please make sure the directory path is correct and accessible")
else:
    print(f"Using document directory: {NOTES_DIR}")
    # Check if there are any PDF files
    pdf_files = glob.glob(os.path.join(NOTES_DIR, "**/*.pdf"), recursive=True)
    print(f"Found {len(pdf_files)} PDF files in the document directory")

# Functions for document processing
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, min_chunk_length=MIN_CHUNK_LENGTH):
    """Split text into overlapping chunks with improved handling"""
    chunks = []
    if not text or len(text) < min_chunk_length:
        return chunks
    
    # Split by paragraphs first to try to keep logical sections together
    paragraphs = text.split('\n\n')
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        paragraph_words = paragraph.split()
        paragraph_size = len(paragraph_words)
        
        # If paragraph itself exceeds the chunk size, split it
        if paragraph_size > chunk_size:
            # Add current chunk if it's not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap from previous chunk
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] if overlap_start > 0 else []
                current_size = len(current_chunk)
            
            # Process the large paragraph
            for i in range(0, paragraph_size, chunk_size - overlap):
                chunk = paragraph_words[i:i + chunk_size]
                if len(chunk) >= min_chunk_length:
                    chunks.append(' '.join(chunk))
                    
                    # Last piece becomes part of the next chunk for overlap
                    if i + chunk_size < paragraph_size:
                        current_chunk = chunk[-overlap:]
                        current_size = len(current_chunk)
        else:
            # Check if adding this paragraph would exceed the chunk size
            if current_size + paragraph_size > chunk_size:
                # Save current chunk and start a new one with overlap
                chunks.append(' '.join(current_chunk))
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] if overlap_start > 0 else []
                current_size = len(current_chunk)
            
            # Add paragraph to current chunk
            current_chunk.extend(paragraph_words)
            current_size += paragraph_size
    
    # Add the last chunk if not empty and meets minimum size
    if current_chunk and len(current_chunk) >= min_chunk_length:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_file_hash(filepath):
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_metadata():
    """Get metadata for all PDF files with hash for better detection of changes"""
    pdf_files = glob.glob(os.path.join(NOTES_DIR, "**/*.pdf"), recursive=True)
    metadata = {}
    
    print(f"Scanning directory: {NOTES_DIR}")
    for pdf_file in pdf_files:
        try:
            metadata[pdf_file] = {
                'mtime': os.stat(pdf_file).st_mtime,
                'size': os.stat(pdf_file).st_size,
                'hash': get_file_hash(pdf_file)
            }
            print(f"Processed metadata for: {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"Error getting metadata for {pdf_file}: {str(e)}")
    
    # Save metadata to a file for future comparison
    metadata_file = os.path.join(CACHE_DIR, "file_metadata.json")
    with open(metadata_file, 'w') as f:
        # Convert file paths to relative paths for better portability
        serializable_metadata = {}
        for path, data in metadata.items():
            serializable_metadata[os.path.basename(path)] = data
        json.dump(serializable_metadata, f, indent=2)
    
    return metadata

def load_documents():
    """Process all documents and create embeddings with selective reloading"""
    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created cache directory: {CACHE_DIR}")

    documents = []
    vectorizer = None
    embeddings = None
    
    # Check if we already have processed documents
    cached_data_exists = (
        os.path.exists(DOCUMENTS_FILE) and 
        os.path.exists(EMBEDDINGS_FILE) and 
        os.path.exists(VECTORIZER_FILE) and
        os.path.exists(os.path.join(CACHE_DIR, "file_metadata.json"))
    )
    
    if cached_data_exists:
        try:
            # Load cached metadata
            with open(os.path.join(CACHE_DIR, "file_metadata.json"), 'r') as f:
                cached_metadata = json.load(f)
            
            # Get current metadata
            current_metadata = get_file_metadata()
            
            # Check if any files have changed
            files_changed = False
            current_files = set(os.path.basename(f) for f in current_metadata.keys())
            cached_files = set(cached_metadata.keys())
            
            # Check for new or deleted files
            if current_files != cached_files:
                print("File set has changed (new files added or files deleted)")
                files_changed = True
            else:
                # Check for modified files
                for filepath, data in current_metadata.items():
                    filename = os.path.basename(filepath)
                    if filename in cached_metadata:
                        if data['hash'] != cached_metadata[filename]['hash']:
                            print(f"File has changed: {filename}")
                            files_changed = True
                            break
            
            if not files_changed:
                # Load cached data if nothing has changed
                print("No file changes detected, loading from cache...")
                with open(DOCUMENTS_FILE, 'rb') as f:
                    documents = pickle.load(f)
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    embeddings = pickle.load(f)
                with open(VECTORIZER_FILE, 'rb') as f:
                    vectorizer = pickle.load(f)
                    
                # Print debug information about vectorizer and embeddings
                print(f"Loaded vectorizer with {len(vectorizer.get_feature_names_out())} features")
                if hasattr(embeddings, 'shape'):
                    print(f"Loaded embeddings matrix shape: {embeddings.shape}")
                else:
                    print(f"Warning: Embeddings object has no shape attribute")
                    
                print(f"Loaded {len(documents)} documents and embeddings from cache")
                return documents, embeddings, vectorizer
            else:
                print("Changes detected in files, reprocessing documents...")
                # Delete the cache files to ensure clean rebuild
                if os.path.exists(DOCUMENTS_FILE):
                    os.remove(DOCUMENTS_FILE)
                if os.path.exists(EMBEDDINGS_FILE):
                    os.remove(EMBEDDINGS_FILE)
                if os.path.exists(VECTORIZER_FILE):
                    os.remove(VECTORIZER_FILE)
        except Exception as e:
            print(f"Error checking cached data: {e}")
            print("Proceeding with full document processing...")
            # Delete potentially corrupted cache files
            if os.path.exists(DOCUMENTS_FILE):
                os.remove(DOCUMENTS_FILE)
            if os.path.exists(EMBEDDINGS_FILE):
                os.remove(EMBEDDINGS_FILE)
            if os.path.exists(VECTORIZER_FILE):
                os.remove(VECTORIZER_FILE)
    
    # Process all files if no cache is available or files have changed
    print("Processing documents and creating new embeddings...")
    all_chunks = []
    documents = []
    
    file_metadata = get_file_metadata()
    for pdf_file in file_metadata:
        print(f"Processing {os.path.basename(pdf_file)}...")
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            documents.append({
                "id": f"{os.path.basename(pdf_file)}_{i}",
                "text": chunk,
                "source": os.path.basename(pdf_file),
                "file_path": pdf_file
            })
    
    if all_chunks:
        # Create vectorizer with optimal parameters
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,  # Limit features to most important ones
            min_df=1,                   # Include terms that appear at least once
            max_df=0.9,                 # Ignore terms that appear in more than 90% of docs
            ngram_range=(1, 3),         # Include 1, 2, and 3-grams for phrases
            stop_words='english',       # Remove English stop words
            use_idf=True,               # Use inverse document frequency
            norm='l2',                  # Normalize with L2 norm
            analyzer='word'             # Analyze by word
        )
        
        # Create a list of all document texts
        texts = [doc["text"] for doc in documents]
        
        # Fit vectorizer and transform all documents
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = vectorizer.fit_transform(texts)
        
        # Print debug information about new vectorizer and embeddings
        print(f"Created vectorizer with {len(vectorizer.get_feature_names_out())} features")
        print(f"Created embeddings matrix shape: {embeddings.shape}")
        
        # Save everything to cache
        with open(DOCUMENTS_FILE, 'wb') as f:
            pickle.dump(documents, f)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(VECTORIZER_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"Processed and saved {len(documents)} document chunks")
    
    return documents, embeddings, vectorizer

def preprocess_query(query):
    """Enhanced query preprocessing with better expansion logic"""
    # Convert to lowercase
    query = query.lower()
    
    # Remove special characters but keep spaces
    query = re.sub(r'[^\w\s]', ' ', query)
    
    # Extract key terms and expand them
    expanded_query = query
    
    # Identify subject matter in the query
    subjects = {
        'saturn': ['saturn', 'shani', 'sani', 'kronos'],
        'jupiter': ['jupiter', 'guru', 'brihaspati'],
        'mars': ['mars', 'mangal', 'kuja', 'angaraka'],
        'venus': ['venus', 'shukra', 'sukra'],
        'sun': ['sun', 'surya', 'ravi', 'aditya'],
        'moon': ['moon', 'chandra', 'soma'],
        'mercury': ['mercury', 'budha', 'buddha'],
        'rahu': ['rahu', 'dragon\'s head', 'north node'],
        'ketu': ['ketu', 'dragon\'s tail', 'south node']
    }
    
    # Identify if the query is asking about a planet
    for planet, terms in subjects.items():
        if any(term in query for term in terms):
            # Add relevant terms to query
            expanded_query = f"{expanded_query} {planet} "
            expanded_query += " ".join(terms)
            
            # Add related concepts for better matching
            expanded_query += " planet graha characteristics traits effects influence impact"
    
    # If query is just looking for information (tell me about, what is, etc.)
    information_patterns = ['tell me about', 'what is', 'describe', 'explain', 'information on']
    if any(pattern in query for pattern in information_patterns):
        expanded_query += " description characteristics qualities attributes significance meaning importance"
    
    # Add house-related terms if query mentions houses
    house_patterns = ['house', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th']
    if any(pattern in query for pattern in house_patterns):
        expanded_query += " house bhava placement position effects results impact significance"
    
    print(f"Original query: {query}")
    print(f"Expanded query: {expanded_query}")
    return expanded_query

def search_documents(query, documents, embeddings, vectorizer, top_k=TOP_K_RESULTS):
    """Search for documents relevant to the query using cosine similarity"""
    try:
        print("\n" + "="*80)
        print(f"SEARCH QUERY: '{query}'")
        print("="*80)
        
        # Preprocess the query to improve matching
        processed_query = preprocess_query(query)
        print(f"Processing query using TF-IDF vectorization...")
        
        # Create the query embedding - IMPORTANT: Make sure we're using the same vectorizer as the embeddings
        try:
            # Debug info
            print(f"Vectorizer has {len(vectorizer.get_feature_names_out())} features")
            print(f"Embeddings matrix shape: {embeddings.shape}")
            
            query_embedding = vectorizer.transform([processed_query])
            print(f"Created query embedding with shape {query_embedding.shape}")
            
            # Safety check for dimension match
            if query_embedding.shape[1] != embeddings.shape[1]:
                print(f"⚠️ WARNING: Query embedding dimension {query_embedding.shape[1]} does not match document embeddings dimension {embeddings.shape[1]}")
                print("This would cause a dimension mismatch error. Attempting recovery...")
                
                # Re-fit the vectorizer on the texts from documents to ensure matching vocabulary
                print("Re-fitting vectorizer on document texts...")
                all_texts = [doc["text"] for doc in documents]
                vectorizer.fit(all_texts)
                print(f"Re-fitted vectorizer now has {len(vectorizer.get_feature_names_out())} features")
                
                # Transform the query with the updated vectorizer
                query_embedding = vectorizer.transform([processed_query])
                print(f"Updated query embedding shape: {query_embedding.shape}")
        except Exception as e:
            print(f"❌ Error creating query embedding: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Calculate similarity scores using dot product
        similarity_scores = (query_embedding @ embeddings.T).toarray()[0]
        
        # Print top similarity scores for debugging
        top_scores = sorted(similarity_scores, reverse=True)[:5]
        print(f"Top 5 similarity scores: {[round(score, 4) for score in top_scores]}")
        
        # Get indices of top k results
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        # Prepare results
        search_results = []
        print("\nDOCUMENT MATCHES:")
        print("-"*80)
        
        for idx in top_indices:
            # Use a much lower similarity threshold to ensure we get results
            if similarity_scores[idx] > 0.01:
                search_results.append({
                    "source": documents[idx]["source"],
                    "content": documents[idx]["text"],
                    "similarity": float(similarity_scores[idx])
                })
                print(f"✓ Match: {documents[idx]['source']} (score: {similarity_scores[idx]:.4f})")
                # Print excerpt of the matching content (first 100 chars)
                text_preview = documents[idx]["text"][:100].replace('\n', ' ').strip() + "..."
                print(f"  Preview: {text_preview}")
                print("-"*80)
        
        if not search_results:
            print("✗ No documents passed the similarity threshold (0.01)")
            print("Returning top 3 results anyway:")
            print("-"*80)
            # Return top 3 results regardless of similarity score
            for idx in top_indices[:3]:
                search_results.append({
                    "source": documents[idx]["source"],
                    "content": documents[idx]["text"],
                    "similarity": float(similarity_scores[idx])
                })
                print(f"⚠ Low score match: {documents[idx]['source']} (score: {similarity_scores[idx]:.4f})")
                # Print excerpt of the matching content (first 100 chars)
                text_preview = documents[idx]["text"][:100].replace('\n', ' ').strip() + "..."
                print(f"  Preview: {text_preview}")
                print("-"*80)
                
        print(f"Vector search found {len(search_results)} results\n")
        return search_results
    except Exception as e:
        print(f"❌ Error searching documents: {e}")
        import traceback
        traceback.print_exc()
        return []

def hybrid_search(query, documents, embeddings, vectorizer, top_k=TOP_K_RESULTS):
    """Combine TF-IDF similarity with keyword matching for better results"""
    print("\n" + "="*80)
    print(f"HYBRID SEARCH QUERY: '{query}'")
    print("="*80)
    
    # First, try standard vector search
    print("STEP 1: Running vector-based search...")
    vector_results = search_documents(query, documents, embeddings, vectorizer, top_k)
    
    # If we got good results, return them
    if len(vector_results) >= 3 and vector_results[0]['similarity'] > 0.1:
        print("✓ Vector search found good results, using those.")
        return vector_results
    
    # Otherwise, fall back to keyword-based search
    print("\nSTEP 2: Vector search results insufficient, falling back to keyword search...")
    
    # Extract key terms from query
    query_lower = query.lower()
    key_terms = []
    
    # Check for planets
    planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'rahu', 'ketu']
    for planet in planets:
        if planet in query_lower:
            key_terms.append(planet)
            print(f"✓ Found planet: {planet}")
    
    # Check for houses
    houses = ['1st house', '2nd house', '3rd house', '4th house', '5th house', 
              '6th house', '7th house', '8th house', '9th house', '10th house', 
              '11th house', '12th house', 'first house', 'second house', 
              'third house', 'fourth house', 'fifth house', 'sixth house',
              'seventh house', 'eighth house', 'ninth house', 'tenth house',
              'eleventh house', 'twelfth house']
    for house in houses:
        if house in query_lower:
            key_terms.append(house)
            print(f"✓ Found house: {house}")
    
    # If no planets or houses, look for other key astrological terms
    if not key_terms:
        astro_terms = ['house', 'sign', 'nakshatra', 'yoga', 'dasha', 'transit', 'conjunction', 'aspect', 'lagna', 'ascendant']
        for term in astro_terms:
            if term in query_lower:
                key_terms.append(term)
                print(f"✓ Found astrological term: {term}")
    
    # If still no terms, use important words from query (excluding stop words)
    if not key_terms:
        stop_words = ['tell', 'me', 'about', 'what', 'is', 'the', 'a', 'an', 'in', 'on', 'of', 'for', 'by', 'to', 'and', 'or', 'as']
        query_words = query_lower.split()
        for word in query_words:
            if word not in stop_words:
                key_terms.append(word)
                print(f"✓ Found non-stop word: {word}")
    
    print(f"Using search terms: {', '.join(key_terms)}")
    
    # Search for documents containing these terms
    keyword_results = []
    documents_checked = 0
    documents_matched = 0
    
    print("\nKEYWORD MATCHES:")
    print("-"*80)
    
    for idx, doc in enumerate(documents):
        doc_text = doc['text'].lower()
        documents_checked += 1
        
        # Count matches of key terms
        matches = []
        for term in key_terms:
            if term in doc_text:
                matches.append(term)
        
        match_count = len(matches)
        
        if match_count > 0:
            documents_matched += 1
            # Calculate a simple relevance score
            relevance = match_count / len(key_terms) if key_terms else 0
            
            keyword_results.append({
                "source": doc["source"],
                "content": doc["text"],
                "similarity": float(relevance)  # Use key term matches as similarity
            })
            
            if documents_matched <= 10:  # Log just the first 10 matches to avoid console spam
                print(f"✓ Match: {doc['source']} (score: {relevance:.4f})")
                print(f"  Matched terms: {', '.join(matches)}")
                # Print excerpt of the matching content (first 100 chars)
                text_preview = doc["text"][:100].replace('\n', ' ').strip() + "..."
                print(f"  Preview: {text_preview}")
                print("-"*80)
    
    # Sort by relevance and return top results
    keyword_results.sort(key=lambda x: x['similarity'], reverse=True)
    result_count = min(top_k, len(keyword_results))
    
    print(f"\nKeyword search checked {documents_checked} documents and found {documents_matched} matches")
    print(f"Returning top {result_count} keyword search results\n")
    
    return keyword_results[:top_k]

def ask_claude(question, context=None):
    """Ask a question to Claude with optional context using the Messages API"""
    if not API_KEY:
        return "No API key found in .env file. Please add your Anthropic API key."
        
    try:
        headers = {
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Create a more concise context by truncating document content
        context_text = ""
        if context:
            context_text = "Relevant information from astrology documents:\n"
            for doc in context:
                # Truncate content to ~1000 characters to prevent excessive token usage
                content = doc['content']
                if len(content) > 1000:
                    content = content[:1000] + "..."
                context_text += f"From {doc['source']} (similarity: {doc['similarity']:.2f}):\n{content}\n\n"
        
        # System message
        system_message = "You are a Vedic Astrology expert. Give detailed, educational responses based on traditional principles. Don't provide personal readings."
        
        # Use the Messages API format 
        messages = []
        
        # Add context and question as a single user message
        if context:
            messages.append({
                "role": "user",
                "content": f"{context_text}\n\nQuestion: {question}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Question: {question}"
            })
        
        # Try using a smaller model if available
        models = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
        model = models[0]  # Start with the smallest model
        
        data = {
            "model": model,
            "messages": messages,
            "system": system_message,  # System is a top-level parameter
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        # Calculate approximate token count
        token_estimate = len(str(messages)) / 4  # Rough estimate, 4 chars per token
        print(f"Sending API request to Anthropic with estimated {int(token_estimate)} tokens in prompt")
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("content", [{}])[0].get("text", "No answer received.")
        else:
            error_message = f"Error: API request failed with status code {response.status_code}"
            try:
                error_details = response.json()
                error_message += f"\nDetails: {json.dumps(error_details, indent=2)}"
                
                # Check for common errors
                if response.status_code == 400:
                    if "error" in error_details:
                        error_type = error_details.get("error", {}).get("type", "")
                        if error_type == "invalid_api_key":
                            error_message += "\n\nYour API key appears to be invalid. Please check your .env file and ensure you have a valid Anthropic API key."
                        elif error_type == "context_length_exceeded":
                            error_message += "\n\nThe prompt is too long. Try asking a more specific question or using fewer documents as context."
                        elif "model" in error_type.lower():
                            error_message += f"\n\nThe specified model '{model}' may not be available. Please check for the correct model name."
                
                # Check token count issues
                if "token" in str(error_details).lower() or "length" in str(error_details).lower():
                    error_message += f"\n\nEstimated token count: {int(token_estimate)}. The API may have limits on token count."
                    
                    # Suggest solutions for token limits
                    if token_estimate > 100000:
                        error_message += "\n\nThe prompt is too long. Try using fewer documents as context or ask a more specific question."
            except Exception as e:
                error_message += f"\nCould not parse error details: {str(e)}"
            
            print(f"Claude API Error: {error_message}")
            return error_message
    
    except Exception as e:
        error = f"Error communicating with Claude API: {str(e)}"
        print(error)
        import traceback
        traceback.print_exc()
        return error

# Load documents and create embeddings
documents, embeddings, vectorizer = load_documents()
print(f"Loaded {len(documents)} documents with embeddings.")

# Create Flask app
app = Flask(__name__)

# Define HTML template for the main page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vedic Astrology Bot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .input-area {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background-color: #4285f4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #3367d6;
        }
        .answer {
            margin-top: 20px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
            white-space: pre-wrap;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .examples {
            margin-top: 20px;
        }
        .example-question {
            margin: 5px 0;
            padding: 8px 15px;
            background-color: #f1f1f1;
            border-radius: 5px;
            cursor: pointer;
            display: inline-block;
        }
        .example-question:hover {
            background-color: #e1e1e1;
        }
        .sources {
            margin-top: 20px;
            font-size: 14px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .menu {
            display: flex;
            justify-content: flex-end;
            margin-top: 20px;
        }
        .menu-button {
            background-color: #f1f1f1;
            color: #333;
            border: none;
            padding: 8px 15px;
            margin-left: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .menu-button:hover {
            background-color: #e1e1e1;
        }
        .config-info {
            font-size: 12px;
            color: #666;
            margin-top: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vedic Astrology Bot</h1>
        <p>Ask questions about Vedic Astrology concepts, principles, and interpretations.</p>
        
        <div class="input-area">
            <textarea id="question" rows="3" placeholder="Ask your question about Vedic Astrology..."></textarea>
            <div style="margin-top: 10px; display: flex; justify-content: space-between;">
                <button onclick="askQuestion()">Ask Question</button>
                <div>
                    <a href="/documents" class="menu-button" style="text-decoration: none; display: inline-block;">View Documents</a>
                    <a href="/manage-documents" class="menu-button" style="text-decoration: none; display: inline-block;">Manage Documents</a>
                    <button class="menu-button" onclick="reloadDocuments()">Reload Documents</button>
                    <a href="/debug/terms" class="menu-button" style="text-decoration: none; display: inline-block;">Debug Terms</a>
                    <a href="/clear-cache" class="menu-button" style="text-decoration: none; display: inline-block; background-color: #f44336;">Clear Cache</a>
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <p>Consulting the stars...</p>
        </div>
        
        <div class="answer" id="answer"></div>
        
        <div class="sources" id="sources"></div>
        
        <div class="examples">
            <h3>Example Questions:</h3>
            <div class="example-question" onclick="useExample('What are nakshatras in Vedic astrology?')">What are nakshatras?</div>
            <div class="example-question" onclick="useExample('Explain the significance of Jupiter in the 5th house.')">Jupiter in 5th house?</div>
            <div class="example-question" onclick="useExample('What are the remedies for malefic Saturn?')">Saturn remedies?</div>
            <div class="example-question" onclick="useExample('How does Rahu affect career prospects?')">Rahu and career?</div>
        </div>
        
        <div class="config-info">
            <h4>System Configuration:</h4>
            <ul>
                <li>Vector Database: Custom TF-IDF with NumPy</li>
                <li>Embedding: TF-IDF with ngrams(1,3)</li>
                <li>Chunk Size: """ + str(CHUNK_SIZE) + """ words</li>
                <li>Chunk Overlap: """ + str(CHUNK_OVERLAP) + """ words</li>
                <li>Vector Features: """ + str(MAX_FEATURES) + """ </li>
                <li>Top Results: """ + str(TOP_K_RESULTS) + """ per query</li>
                <li>Documents: """ + str(len(documents)) + """ chunks</li>
                <li>Distance Metric: Cosine similarity</li>
                <li>Model: Claude-3-Opus</li>
            </ul>
        </div>
    </div>
    
    <script>
        function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('answer').innerHTML = '';
            document.getElementById('sources').innerHTML = '';
            
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('answer').innerHTML = data.answer;
                
                if (data.sources && data.sources.length > 0) {
                    let sourcesHtml = '<h3>Sources:</h3><ul>';
                    for (const source of data.sources) {
                        sourcesHtml += `<li>${source.source} (similarity: ${source.similarity.toFixed(2)})</li>`;
                    }
                    sourcesHtml += '</ul>';
                    document.getElementById('sources').innerHTML = sourcesHtml;
                } else {
                    document.getElementById('sources').innerHTML = '<p>No relevant sources found in the document collection.</p>';
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('answer').innerHTML = 'Error: ' + error;
            });
        }
        
        function useExample(example) {
            document.getElementById('question').value = example;
        }
        
        function reloadDocuments() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('answer').innerHTML = 'Reloading documents...';
            document.getElementById('sources').innerHTML = '';
            
            fetch('/reload', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('answer').innerHTML = data.message;
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('answer').innerHTML = 'Error reloading documents: ' + error;
            });
        }
        
        // Enable pressing Enter to submit
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the home page"""
    return HTML_TEMPLATE

@app.route('/ask', methods=['POST'])
def process_question():
    """Process a question and return the answer"""
    try:
        # Get current timestamp for logging
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\n\n{'*'*100}")
        print(f"NEW QUERY AT {timestamp}")
        print(f"{'*'*100}")
        
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            print("Error: Empty question received")
            return jsonify({"answer": "Please enter a question.", "sources": []})
        
        print(f"QUESTION: {question}")
        client_ip = request.remote_addr
        print(f"Client IP: {client_ip}")
        
        # Use hybrid search instead of just vector search for better results
        start_time = time.time()
        search_results = hybrid_search(question, documents, embeddings, vectorizer)
        search_time = time.time() - start_time
        print(f"Search completed in {search_time:.2f} seconds, found {len(search_results)} relevant documents")
        
        # Get answer from Claude
        print(f"Sending query to Claude API...")
        start_time = time.time()
        if search_results:
            # Limit the number of documents to prevent token limit issues
            MAX_DOCUMENTS = 5  # Limit to top 5 most relevant documents
            
            if len(search_results) > MAX_DOCUMENTS:
                print(f"Limiting context to top {MAX_DOCUMENTS} documents (out of {len(search_results)}) to prevent token limit issues")
                limited_results = search_results[:MAX_DOCUMENTS]
            else:
                limited_results = search_results
                
            print(f"Using {len(limited_results)} documents as context")
            answer = ask_claude(question, limited_results)
        else:
            # Fallback for when no documents are found - use Claude's knowledge
            print("No relevant documents found. Using Claude's general knowledge.")
            answer = ask_claude(question, None) + "\n\n(Note: This answer is based on Claude's general knowledge as no specific documents were found in our Vedic Astrology collection on this topic.)"
        
        claude_time = time.time() - start_time
        print(f"Claude response received in {claude_time:.2f} seconds")
        
        # Format sources for display
        sources = []
        for result in search_results:
            sources.append({
                "source": result["source"],
                "similarity": result["similarity"]
            })
        
        # Log summary
        total_time = search_time + claude_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Query completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"{'*'*100}\n")
        
        return jsonify({"answer": answer, "sources": sources})
    
    except Exception as e:
        print(f"ERROR PROCESSING QUESTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"Error: {str(e)}", "sources": []})

@app.route('/reload', methods=['POST'])
def reload_documents():
    """Reload documents from the source directory"""
    try:
        global documents, embeddings, vectorizer
        
        # Delete cache files to force reprocessing
        if os.path.exists(DOCUMENTS_FILE):
            os.remove(DOCUMENTS_FILE)
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)
        if os.path.exists(VECTORIZER_FILE):
            os.remove(VECTORIZER_FILE)
        
        # Reload everything
        start_time = time.time()
        documents, embeddings, vectorizer = load_documents()
        elapsed_time = time.time() - start_time
        
        return jsonify({
            "message": f"Successfully reloaded {len(documents)} document chunks. (Completed in {elapsed_time:.2f} seconds)"
        })
    except Exception as e:
        print(f"Error reloading documents: {e}")
        return jsonify({
            "message": f"Error reloading documents: {str(e)}"
        })

@app.route('/documents')
def list_documents():
    """List all documents in the collection"""
    try:
        # Extract unique source files
        sources = {}
        for doc in documents:
            source = doc.get('source')
            if source:
                if source in sources:
                    sources[source] += 1
                else:
                    sources[source] = 1
        
        # Create an HTML page to display the document list
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document List</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .document-list { margin-top: 20px; }
                .document-item { 
                    margin-bottom: 10px; 
                    padding: 10px 15px;
                    background-color: #f1f1f1;
                    border-radius: 5px;
                }
                .document-item a { 
                    color: #4285f4;
                    text-decoration: none;
                    font-weight: bold;
                }
                .document-item:hover { background-color: #e1e1e1; }
                .back-button { 
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 15px;
                    background-color: #4285f4;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>Document List</h1>
            
            <div class="document-list">
        """
        
        for source, count in sources.items():
            html += f"""
                <div class="document-item">
                    <a href="/documents/{source}">{source}</a> ({count} chunks)
                </div>
            """
        
        html += """
            </div>
            
            <a href="/" class="back-button">Back to Main Page</a>
        </body>
        </html>
        """
        
        return html
    
    except Exception as e:
        return f"Error retrieving document list: {str(e)}"

@app.route('/documents/<filename>')
def view_document(filename):
    """View chunks from a specific document"""
    try:
        # Filter for documents with the matching source file
        document_chunks = []
        for i, doc in enumerate(documents):
            if doc.get('source') == filename:
                document_chunks.append({
                    "id": doc.get('id'),
                    "content": doc.get('text')
                })
        
        # Create an HTML page to display the chunks
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document: {filename}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .chunk {{ 
                    margin-bottom: 20px; 
                    padding: 15px; 
                    background-color: #f9f9f9; 
                    border-radius: 5px;
                    border-left: 4px solid #4285f4;
                }}
                .chunk-id {{ color: #999; font-size: 12px; margin-bottom: 5px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                .back-button {{ 
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 15px;
                    background-color: #4285f4;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Document: {filename}</h1>
            <p>Found {len(document_chunks)} chunks from this document</p>
            
            <div id="chunks">
        """
        
        for i, chunk in enumerate(document_chunks):
            html += f"""
                <div class="chunk">
                    <div class="chunk-id">Chunk {i+1} (ID: {chunk['id']})</div>
                    <pre>{chunk['content']}</pre>
                </div>
            """
        
        html += """
            </div>
            
            <a href="/" class="back-button">Back to Main Page</a>
        </body>
        </html>
        """
        
        return html
    
    except Exception as e:
        return f"Error retrieving document chunks: {str(e)}"

@app.route('/debug/terms')
def debug_terms():
    """Display the most common terms in the document collection"""
    try:
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # Get document-term matrix
        X = embeddings.toarray()
        
        # Calculate term frequencies across all documents
        term_freqs = np.sum(X, axis=0)
        
        # Sort terms by frequency
        sorted_indices = np.argsort(term_freqs)[::-1]
        top_terms = [(feature_names[idx], term_freqs[idx]) for idx in sorted_indices[:200]]
        
        # Create HTML output
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Terms Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #4285f4; color: white; }
                .search-box { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
                .search-result { margin-top: 15px; padding: 10px; border-radius: 5px; }
                .found { background-color: #dff0d8; }
                .not-found { background-color: #f2dede; }
                .back-button { 
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 15px;
                    background-color: #4285f4;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>Document Terms Analysis</h1>
            <p>This shows the most common terms in your document collection.</p>
            
            <div class="search-box">
                <h3>Search for Term</h3>
                <input type="text" id="searchTerm" placeholder="Enter term to search">
                <button onclick="searchTerm()">Search</button>
                <div id="searchResult" class="search-result"></div>
            </div>
            
            <h2>Top 200 Terms</h2>
            <table>
                <tr>
                    <th>Term</th>
                    <th>Frequency</th>
                </tr>
        """
        
        for term, freq in top_terms:
            html += f"<tr><td>{term}</td><td>{int(freq)}</td></tr>"
        
        html += """
            </table>
            
            <script>
                function searchTerm() {
                    const term = document.getElementById('searchTerm').value.toLowerCase();
                    const rows = document.querySelectorAll('table tr');
                    let found = false;
                    
                    for (let i = 1; i < rows.length; i++) {
                        const rowTerm = rows[i].cells[0].innerText.toLowerCase();
                        if (rowTerm === term) {
                            const freq = rows[i].cells[1].innerText;
                            document.getElementById('searchResult').innerHTML = 
                                `<div class="found">Term "${term}" found with frequency ${freq}</div>`;
                            found = true;
                            break;
                        }
                    }
                    
                    if (!found) {
                        document.getElementById('searchResult').innerHTML = 
                            `<div class="not-found">Term "${term}" not found in the top 200 terms</div>`;
                    }
                }
            </script>
            
            <a href="/" class="back-button">Back to Main Page</a>
        </body>
        </html>
        """
        
        return html
    
    except Exception as e:
        return f"Error analyzing terms: {str(e)}"

@app.route('/manage-documents')
def manage_documents():
    """Display document management interface"""
    try:
        # Get metadata file if it exists
        metadata_file = os.path.join(CACHE_DIR, "file_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Create HTML output
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Management</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #4285f4; color: white; }
                .action-bar { margin: 20px 0; }
                .checkbox-column { width: 50px; text-align: center; }
                .back-button, .action-button { 
                    display: inline-block;
                    margin-right: 10px;
                    padding: 10px 15px;
                    background-color: #4285f4;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    border: none;
                    cursor: pointer;
                    font-size: 14px;
                }
                .action-button.warning {
                    background-color: #f44336;
                }
                .action-button:hover, .back-button:hover {
                    opacity: 0.9;
                }
                .status {
                    margin-top: 20px;
                    padding: 10px;
                    border-radius: 5px;
                    display: none;
                }
                .success { background-color: #dff0d8; }
                .error { background-color: #f2dede; }
            </style>
        </head>
        <body>
            <h1>Document Management</h1>
            <p>View and manage your Vedic Astrology document collection.</p>
            
            <div class="action-bar">
                <button class="action-button" onclick="selectAll()">Select All</button>
                <button class="action-button" onclick="deselectAll()">Deselect All</button>
                <button class="action-button" onclick="reloadSelected()">Reload Selected</button>
                <button class="action-button warning" onclick="clearCache()">Clear All Cache</button>
            </div>
            
            <div id="status" class="status"></div>
            
            <table id="documents-table">
                <tr>
                    <th class="checkbox-column"><input type="checkbox" id="select-all" onchange="toggleSelectAll()"></th>
                    <th>Document</th>
                    <th>Last Modified</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
        """
        
        for filename, data in metadata.items():
            mod_time = datetime.fromtimestamp(data['mtime']).strftime('%Y-%m-%d %H:%M:%S')
            html += f"""
                <tr>
                    <td class="checkbox-column"><input type="checkbox" name="doc" value="{filename}"></td>
                    <td>{filename}</td>
                    <td>{mod_time}</td>
                    <td>Cached</td>
                    <td>
                        <a href="/documents/{filename}" class="action-button">View</a>
                        <button class="action-button" onclick="reloadDocument('{filename}')">Reload</button>
                    </td>
                </tr>
            """
        
        html += """
            </table>
            
            <div class="action-bar">
                <a href="/" class="back-button">Back to Main Page</a>
            </div>
            
            <script>
                function toggleSelectAll() {
                    const selectAll = document.getElementById('select-all').checked;
                    const checkboxes = document.querySelectorAll('input[name="doc"]');
                    checkboxes.forEach(checkbox => checkbox.checked = selectAll);
                }
                
                function selectAll() {
                    document.getElementById('select-all').checked = true;
                    toggleSelectAll();
                }
                
                function deselectAll() {
                    document.getElementById('select-all').checked = false;
                    toggleSelectAll();
                }
                
                function showStatus(message, isError = false) {
                    const status = document.getElementById('status');
                    status.textContent = message;
                    status.className = isError ? 'status error' : 'status success';
                    status.style.display = 'block';
                    
                    // Hide after 5 seconds
                    setTimeout(() => {
                        status.style.display = 'none';
                    }, 5000);
                }
                
                function reloadDocument(filename) {
                    showStatus(`Reloading document: ${filename}...`);
                    
                    fetch('/reload-document', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ documents: [filename] }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showStatus(data.message);
                        } else {
                            showStatus(data.message, true);
                        }
                    })
                    .catch(error => {
                        showStatus('Error reloading document: ' + error, true);
                    });
                }
                
                function reloadSelected() {
                    const checkboxes = document.querySelectorAll('input[name="doc"]:checked');
                    const documents = Array.from(checkboxes).map(checkbox => checkbox.value);
                    
                    if (documents.length === 0) {
                        showStatus('Please select at least one document to reload.', true);
                        return;
                    }
                    
                    showStatus(`Reloading ${documents.length} document(s)...`);
                    
                    fetch('/reload-document', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ documents }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            showStatus(data.message);
                        } else {
                            showStatus(data.message, true);
                        }
                    })
                    .catch(error => {
                        showStatus('Error reloading documents: ' + error, true);
                    });
                }
                
                function clearCache() {
                    if (confirm('Are you sure you want to clear all cache and reload all documents? This may take some time.')) {
                        showStatus('Clearing cache and reloading all documents...');
                        
                        fetch('/reload', {
                            method: 'POST',
                        })
                        .then(response => response.json())
                        .then(data => {
                            showStatus(data.message);
                            // Reload the page after successful cache clear
                            setTimeout(() => {
                                window.location.reload();
                            }, 2000);
                        })
                        .catch(error => {
                            showStatus('Error clearing cache: ' + error, true);
                        });
                    }
                }
            </script>
        </body>
        </html>
        """
        
        return html
    
    except Exception as e:
        return f"Error loading document management interface: {str(e)}"

@app.route('/reload-document', methods=['POST'])
def reload_specific_documents():
    """Reload specific documents"""
    try:
        global documents, embeddings, vectorizer
        
        # Get the documents to reload
        data = request.get_json()
        documents_to_reload = data.get('documents', [])
        
        if not documents_to_reload:
            return jsonify({
                "success": False,
                "message": "No documents specified for reloading."
            })
        
        # Find the full paths for the documents
        metadata_file = os.path.join(CACHE_DIR, "file_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            return jsonify({
                "success": False,
                "message": "Metadata file not found. Try clearing all cache instead."
            })
        
        # Get all PDF files to find the full paths
        all_pdf_files = glob.glob(os.path.join(NOTES_DIR, "**/*.pdf"), recursive=True)
        found_documents = []
        
        for pdf_file in all_pdf_files:
            filename = os.path.basename(pdf_file)
            if filename in documents_to_reload:
                found_documents.append(pdf_file)
        
        if not found_documents:
            return jsonify({
                "success": False,
                "message": "None of the specified documents were found in the source directory."
            })
        
        # Load current documents from cache
        if os.path.exists(DOCUMENTS_FILE) and os.path.exists(EMBEDDINGS_FILE) and os.path.exists(VECTORIZER_FILE):
            with open(DOCUMENTS_FILE, 'rb') as f:
                documents = pickle.load(f)
            with open(EMBEDDINGS_FILE, 'rb') as f:
                embeddings = pickle.load(f)
            with open(VECTORIZER_FILE, 'rb') as f:
                vectorizer = pickle.load(f)
        else:
            return jsonify({
                "success": False,
                "message": "Cache files not found. Try clearing all cache instead."
            })
        
        # Process each document to reload
        # First, remove the existing chunks for these documents
        document_ids_to_keep = []
        documents_to_keep = []
        
        for i, doc in enumerate(documents):
            source = doc.get('source')
            if source not in documents_to_reload:
                document_ids_to_keep.append(i)
                documents_to_keep.append(doc)
        
        # Create new chunks for the documents to reload
        new_chunks = []
        for pdf_file in found_documents:
            filename = os.path.basename(pdf_file)
            print(f"Reprocessing {filename}...")
            
            text = extract_text_from_pdf(pdf_file)
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                new_chunks.append({
                    "id": f"{filename}_{i}",
                    "text": chunk,
                    "source": filename,
                    "file_path": pdf_file
                })
                
        # Combine kept documents and new chunks
        updated_documents = documents_to_keep + new_chunks
        
        # Update vectorizer and embeddings
        texts = [doc["text"] for doc in updated_documents]
        updated_embeddings = vectorizer.transform(texts)
        
        # Save updated data to cache
        with open(DOCUMENTS_FILE, 'wb') as f:
            pickle.dump(updated_documents, f)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(updated_embeddings, f)
        
        # Update global variables
        documents = updated_documents
        embeddings = updated_embeddings
        
        # Update metadata
        for pdf_file in found_documents:
            filename = os.path.basename(pdf_file)
            metadata[filename] = {
                'mtime': os.stat(pdf_file).st_mtime,
                'size': os.stat(pdf_file).st_size,
                'hash': get_file_hash(pdf_file)
            }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": f"Successfully reloaded {len(found_documents)} document(s). Document collection now contains {len(updated_documents)} chunks."
        })
        
    except Exception as e:
        print(f"Error reloading documents: {e}")
        return jsonify({
            "success": False,
            "message": f"Error reloading documents: {str(e)}"
        })

@app.route('/clear-cache', methods=['GET', 'POST'])
def clear_cache():
    """Clear the cache and force document reprocessing"""
    try:
        # Delete all cache files
        cache_files = [
            DOCUMENTS_FILE,
            EMBEDDINGS_FILE,
            VECTORIZER_FILE,
            os.path.join(CACHE_DIR, "file_metadata.json")
        ]
        
        deleted_count = 0
        for file_path in cache_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted cache file: {file_path}")
        
        message = f"Successfully cleared {deleted_count} cache files. Please restart the application to reload documents."
        
        # If this was a GET request, show a confirmation page
        if request.method == 'GET':
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cache Cleared</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .message {{ 
                        padding: 15px; 
                        background-color: #dff0d8; 
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .back-button {{ 
                        display: inline-block;
                        padding: 10px 15px;
                        background-color: #4285f4;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <h1>Cache Cleared</h1>
                <div class="message">{message}</div>
                <p>You'll need to restart the server to apply these changes.</p>
                <a href="/" class="back-button">Back to Main Page</a>
            </body>
            </html>
            """
            return html
        else:
            # For POST requests, return a JSON response
            return jsonify({
                "success": True,
                "message": message
            })
    except Exception as e:
        error_message = f"Error clearing cache: {str(e)}"
        print(error_message)
        
        if request.method == 'GET':
            return f"<h1>Error</h1><p>{error_message}</p><a href='/'>Back to Main Page</a>"
        else:
            return jsonify({
                "success": False,
                "message": error_message
            })

if __name__ == "__main__":
    # Start the Flask app
    port = 5000
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True) 