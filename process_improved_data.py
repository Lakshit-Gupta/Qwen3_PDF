#!/usr/bin/env python3
"""
Process improved data for EduPlan AI system.
This script loads the improved JSON data, generates embeddings using NV-Embed,
and stores them in the Qdrant vector database for efficient retrieval.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import required modules
from src.models.embedding_model import NVEmbedPipeline
from src.database.qdrant_connector import QdrantConnector
from src.core.config import QDRANT_COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, QDRANT_VECTOR_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_improved_data(data_dir: str = "../../data/processed_improved") -> List[Dict[str, Any]]:
    """
    Load all improved data files from the specified directory.
    
    Args:
        data_dir: Directory containing improved JSON data files
    
    Returns:
        List of dictionaries containing the loaded data
    """
    # Resolve data path relative to this script file
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / data_dir
    logger.info(f"Loading improved data from {data_path}")
    
    all_data = []
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return all_data
    
    # Find all JSON files with the new naming pattern (Chapter_X_Y.json)
    json_files = sorted(data_path.glob("Chapter_*.json"))
    
    if not json_files:
        logger.warning(f"No Chapter_*.json files found in {data_path}")
        return all_data
    
    logger.info(f"Found {len(json_files)} JSON files to load")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded {json_file.name}: {len(data) if isinstance(data, list) else '1'} items")
                all_data.append({
                    "file": json_file.name,
                    "data": data
                })
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    return all_data

def extract_chapter_info(filename: str) -> Tuple[str, str]:
    """
    Extract chapter number and chunk info from filename.
    
    Args:
        filename: Filename like "Chapter_1_1.json"
    
    Returns:
        Tuple of (chapter_number, chunk_number)
    """
    try:
        # Parse filename: Chapter_X_Y.json -> chapter=X, chunk=Y
        name_parts = filename.replace('.json', '').split('_')
        if len(name_parts) >= 3:
            chapter_num = name_parts[1]
            chunk_num = name_parts[2]
            return chapter_num, chunk_num
        else:
            # Fallback for unexpected format
            return name_parts[1] if len(name_parts) > 1 else "unknown", "1"
    except Exception as e:
        logger.warning(f"Could not parse filename {filename}: {e}")
        return "unknown", "1"

def extract_content_from_block(content_block: Dict[str, Any]) -> str:
    """
    Extract text content from a content block based on its type.
    
    Args:
        content_block: Dictionary containing content block data
    
    Returns:
        Extracted text content
    """
    block_type = content_block.get("type", "unknown")
    content_parts = []
    
    if block_type == "text":
        content = content_block.get("content", "")
        if content:
            content_parts.append(content)
    
    elif block_type == "summary":
        title = content_block.get("title", "")
        if title:
            content_parts.append(f"Summary: {title}")
        
        summary_points = content_block.get("summary_points", [])
        if summary_points:
            content_parts.append("Key Points:")
            for point in summary_points:
                content_parts.append(f"• {point}")
    
    elif block_type == "activity":
        activity_num = content_block.get("activity_number", "")
        title = content_block.get("title", "")
        description = content_block.get("description", "")
        questions = content_block.get("questions", [])
        
        # Build activity content
        activity_parts = []
        if activity_num:
            activity_parts.append(f"Activity {activity_num}")
        if title:
            activity_parts.append(f"Title: {title}")
        if description:
            activity_parts.append(f"Description: {description}")
        if questions:
            activity_parts.append("Questions:")
            for i, question in enumerate(questions, 1):
                activity_parts.append(f"{i}. {question}")
        
        if activity_parts:
            content_parts.append("Activity:\n" + "\n".join(activity_parts))
    
    elif block_type == "questions":
        title = content_block.get("title", "")
        questions = content_block.get("questions", [])
        
        if title:
            content_parts.append(f"Questions Section: {title}")
        
        if questions:
            content_parts.append("Questions:")
            for i, question in enumerate(questions, 1):
                content_parts.append(f"{i}. {question}")
    
    return "\n".join(content_parts)

def prepare_documents(improved_data: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extract text and metadata from improved data with the new chunked format.
    
    Args:
        improved_data: List of dictionaries containing improved data
    
    Returns:
        Tuple containing (text_chunks, metadata)
    """
    texts = []
    metadata = []
    
    for file_data in improved_data:
        filename = file_data["file"]
        data = file_data["data"]
        
        # Extract chapter and chunk info from new filename format
        chapter_num, chunk_num = extract_chapter_info(filename)
        
        logger.info(f"Processing {filename} - Chapter {chapter_num}, Chunk {chunk_num}")
        
        # Handle the new format: data is a list with chapter objects
        if isinstance(data, list) and len(data) > 0:
            for chapter_idx, chapter_data in enumerate(data):
                if not isinstance(chapter_data, dict):
                    continue
                
                chapter_number = chapter_data.get("chapter_number", chapter_num)
                chapter_title = chapter_data.get("chapter_title", f"Chapter {chapter_num}")
                
                # Process all sections in this chapter
                sections = chapter_data.get("sections", [])
                
                for section_idx, section in enumerate(sections):
                    section_number = section.get("section_number", "")
                    section_title = section.get("section_title", "")
                    
                    # Process content blocks in main section
                    content_blocks = section.get("content_blocks", [])
                    
                    for block_idx, content_block in enumerate(content_blocks):
                        if not isinstance(content_block, dict):
                            continue
                        
                        # Extract text content from this block
                        text_content = extract_content_from_block(content_block)
                        
                        # Only add if we have substantial content
                        if text_content and len(text_content.strip()) > 10:
                            texts.append(text_content.strip())
                            
                            # Create metadata for this content block
                            meta = {
                                "id": f"ch{chapter_num}_chunk{chunk_num}_s{section_idx}_b{block_idx}",
                                "chapter_number": chapter_number,
                                "chapter_title": chapter_title,
                                "section_number": section_number or f"Section {section_idx + 1}",
                                "section_title": section_title,
                                "content_type": content_block.get("type", "unknown"),
                                "source_file": filename,
                                "chunk_number": chunk_num,
                                "original_chapter": f"Chapter {chapter_num}",
                                "block_index": block_idx
                            }
                            metadata.append(meta)
                    
                    # Process sub-sections
                    sub_sections = section.get("sub_sections", [])
                    
                    for sub_idx, sub_section in enumerate(sub_sections):
                        sub_section_number = sub_section.get("section_number", "")
                        sub_section_title = sub_section.get("section_title", "")
                        
                        # Process content blocks in sub-sections
                        sub_content_blocks = sub_section.get("content_blocks", [])
                        
                        for sub_block_idx, content_block in enumerate(sub_content_blocks):
                            if not isinstance(content_block, dict):
                                continue
                            
                            # Extract text content from this block
                            text_content = extract_content_from_block(content_block)
                            
                            # Only add if we have substantial content
                            if text_content and len(text_content.strip()) > 10:
                                texts.append(text_content.strip())
                                
                                # Create metadata for this sub-section content block
                                meta = {
                                    "id": f"ch{chapter_num}_chunk{chunk_num}_s{section_idx}_sub{sub_idx}_b{sub_block_idx}",
                                    "chapter_number": chapter_number,
                                    "chapter_title": chapter_title,
                                    "section_number": section_number or f"Section {section_idx + 1}",
                                    "section_title": section_title,
                                    "sub_section_number": sub_section_number,
                                    "sub_section_title": sub_section_title,
                                    "content_type": content_block.get("type", "unknown"),
                                    "source_file": filename,
                                    "chunk_number": chunk_num,
                                    "original_chapter": f"Chapter {chapter_num}",
                                    "block_index": sub_block_idx,
                                    "is_sub_section": True
                                }
                                metadata.append(meta)
        
        else:
            logger.warning(f"Unexpected data format for file: {filename}")
    
    logger.info(f"Prepared {len(texts)} documents with metadata")
    return texts, metadata

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for text chunks using NV-Embed.
    
    Args:
        texts: List of text chunks to embed
    
    Returns:
        List of embedding vectors
    """
    logger.info("Initializing NV-Embed model...")
    embedding_model = NVEmbedPipeline()
    
    logger.info(f"Generating embeddings for {len(texts)} documents...")
    start_time = time.time()
    embeddings = embedding_model.embed_texts(texts)
    elapsed = time.time() - start_time
    
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f} seconds")
    
    # Debug: Check the format of embeddings
    if embeddings and len(embeddings) > 0:
        first_emb = embeddings[0]
        logger.info(f"First embedding type: {type(first_emb)}")
        logger.info(f"First embedding length: {len(first_emb)}")
        if hasattr(first_emb, 'tolist') and callable(getattr(first_emb, 'tolist')):
            logger.info("Converting embeddings from numpy/tensor to list format")
            embeddings = [emb.tolist() for emb in embeddings]
    
    return embeddings

def store_in_database(texts, embeddings, metadata, collection_name="science_9_collection"):
    """
    Store documents and embeddings in Qdrant.
    
    Args:
        texts: List of text chunks
        embeddings: List of embedding vectors
        metadata: List of metadata dictionaries
        collection_name: Name of the Qdrant collection
    """
    logger.info(f"Storing data in collection: {collection_name}")

    # Always use 2560 for QWEN-4B size
    vector_size = 2560
    logger.info(f"Using vector size: {vector_size}")
    
    # Create Qdrant connector
    qdrant = QdrantConnector(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=collection_name,
        vector_size=vector_size
    )
    
    # Create collection if it doesn't exist
    qdrant.recreate_collection()
    
    # Prepare documents for insertion
    documents = []
    for i, (text, meta) in enumerate(zip(texts, metadata)):
        # Use numeric ID (required by Qdrant) but save original ID in metadata
        original_id = meta.get("id", f"doc_{i}")
        meta["original_id"] = original_id  # Keep the original ID in metadata
        
        doc = {
            "id": i,  # Use simple numeric ID for Qdrant
            "text": text,
            "metadata": meta
        }
        documents.append(doc)
    
    # Insert documents
    success = qdrant.insert_documents(documents, embeddings)
    
    if success:
        logger.info(f"Successfully stored {len(documents)} documents in database")
    else:
        logger.error("Failed to store documents in database")
        
    return success

def main():
    """Main processing function"""
    logger.info("Starting improved data processing with NV-Embed")
    
    # Load improved data
    improved_data = load_improved_data()
    if not improved_data:
        logger.error("No improved data found. Exiting.")
        return
    
    # Prepare documents
    texts, metadata = prepare_documents(improved_data)
    if not texts:
        logger.error("No text chunks extracted. Exiting.")
        return
    
    # Generate embeddings
    embeddings = generate_embeddings(texts)
    if not embeddings or len(embeddings) != len(texts):
        logger.error(f"Embedding generation failed. Got {len(embeddings)} embeddings for {len(texts)} texts.")
        return
    
    # Store in database
    success = store_in_database(texts, embeddings, metadata)
    
    if success:
        logger.info(f"✅ Processing completed successfully!")
        logger.info(f"   Processed {len(texts)} documents across {len(improved_data)} files")
        logger.info(f"   Files processed: {[data['file'] for data in improved_data]}")
    else:
        logger.error("❌ Processing failed.")

if __name__ == "__main__":
    main()