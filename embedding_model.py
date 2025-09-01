"""
Embedding model for EduPlan AI.
This module provides the NVEmbedPipeline class for generating embeddings
using the NVIDIA NV-Embed model.
"""

import logging
import time
import torch
from typing import List, Union, Dict, Any
from transformers import AutoModel, AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NVEmbedPipeline:
    """Pipeline for generating embeddings using NVIDIA NV-Embed."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = None):
        """Initialize the NVEmbedPipeline."""
        self.model_name = model_name
        
        # Use CUDA if available, otherwise fall back to CPU
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use main discrete GPU
        else:
            self.device = device
            
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load the NV-Embed model and tokenizer."""
        try:
            # GPU memory management (ADD THIS)
            if self.device.startswith("cuda"):
                torch.cuda.set_per_process_memory_fraction(1.0)  # Use only 70% of VRAM
                torch.cuda.empty_cache()

            # Print loading message
            print(f"🔄 Loading Qwen/Qwen3-Embedding-4B: {self.model_name}")
            print(f"🎯 Using device: {self.device}")
            
            # Use half precision for GPU to save memory
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

            # dtype = torch.float16 
            # Load tokenizer with trust_remote_code=True for NVIDIA models
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load model with memory optimization (ADD low_cpu_mem_usage=True)
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True  # ← ADD THIS
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Save embedding dimension from config
            self.embedding_dim = 2560  # Hard-coded for NV-Embed-v2
            self.vector_size = self.embedding_dim  # Add this for compatibility
            
            # Print success message
            print(f"✅ Qwen/Qwen3-Embedding-4B loaded successfully!")
            print(f"   📊 Vector size: {self.embedding_dim}")
            print(f"   🎯 Device: {self.device}")
            print(f"   📏 Model dtype: {dtype}")
            
        except Exception as e:
            logger.error(f"Error loading NV-Embed model: {e}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 1) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Cast float tensors to model dtype
            for key in inputs:
                if torch.is_floating_point(inputs[key]):
                    inputs[key] = inputs[key].to(self.model.dtype)
            
            try:
                # Generate embeddings with no gradient tracking
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get embeddings - simpler approach with less memory usage
                if isinstance(outputs, dict):
                    if "sentence_embeddings" in outputs:
                        token_embeddings = outputs["sentence_embeddings"]
                        
                        # Check shape - don't print to reduce log spam
                        if len(token_embeddings.shape) == 3:
                            # Memory-efficient pooling: process one sequence at a time
                            batch_embeddings = []
                            for seq_idx in range(token_embeddings.shape[0]):
                                # Get single sequence tokens and its mask
                                seq_tokens = token_embeddings[seq_idx]
                                seq_mask = inputs["attention_mask"][seq_idx]
                                
                                # Apply mask and mean only for this sequence
                                masked_tokens = seq_tokens * seq_mask.unsqueeze(-1)
                                # Sum and divide by non-zero mask elements
                                sum_tokens = torch.sum(masked_tokens, dim=0)
                                token_count = torch.sum(seq_mask).item()
                                if token_count > 0:
                                    mean_embedding = (sum_tokens / token_count).cpu().numpy()
                                else:
                                    # Fallback if no tokens (shouldn't happen)
                                    mean_embedding = torch.zeros(self.embedding_dim).cpu().numpy()
                                    
                                batch_embeddings.append(mean_embedding)
                            
                            # Convert to numpy array
                            batch_embeddings = np.array(batch_embeddings)
                        else:
                            # Already sentence-level embeddings
                            batch_embeddings = token_embeddings.cpu().numpy()
                    else:
                        # Alternative keys
                        if "last_hidden_state" in outputs:
                            # Similar memory-efficient approach for last_hidden_state
                            token_embeddings = outputs["last_hidden_state"]
                            batch_embeddings = []
                            for seq_idx in range(token_embeddings.shape[0]):
                                seq_tokens = token_embeddings[seq_idx]
                                seq_mask = inputs["attention_mask"][seq_idx]
                                masked_tokens = seq_tokens * seq_mask.unsqueeze(-1)
                                sum_tokens = torch.sum(masked_tokens, dim=0)
                                token_count = torch.sum(seq_mask).item()
                                if token_count > 0:
                                    mean_embedding = (sum_tokens / token_count).cpu().numpy()
                                else:
                                    mean_embedding = torch.zeros(self.embedding_dim).cpu().numpy()
                                batch_embeddings.append(mean_embedding)
                            
                            batch_embeddings = np.array(batch_embeddings)
                        else:
                            # Try to find any usable tensor
                            usable_key = None
                            for key, value in outputs.items():
                                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                                    usable_key = key
                                    break
                            
                            if usable_key:
                                logger.info(f"Using fallback key: {usable_key}")
                                batch_embeddings = outputs[usable_key].cpu().numpy()
                            else:
                                raise ValueError("Cannot find usable embeddings in model output")
                else:
                    # Direct tensor - use memory-efficient approach
                    token_embeddings = outputs
                    batch_embeddings = []
                    for seq_idx in range(token_embeddings.shape[0]):
                        seq_tokens = token_embeddings[seq_idx]
                        seq_mask = inputs["attention_mask"][seq_idx]
                        masked_tokens = seq_tokens * seq_mask.unsqueeze(-1)
                        sum_tokens = torch.sum(masked_tokens, dim=0)
                        token_count = torch.sum(seq_mask).item()
                        if token_count > 0:
                            mean_embedding = (sum_tokens / token_count).cpu().numpy()
                        else:
                            mean_embedding = torch.zeros(self.embedding_dim).cpu().numpy()
                        batch_embeddings.append(mean_embedding)
                    
                    batch_embeddings = np.array(batch_embeddings)
                
                # Add to results
                embeddings.extend(batch_embeddings)
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # Out of memory, clean up and try with smaller batch
                    torch.cuda.empty_cache()
                    logger.warning(f"GPU out of memory, reducing batch size and retrying...")
                    
                    if batch_size > 1:
                        # Try with batch_size of 1
                        for text in batch_texts:
                            try:
                                # Process one text at a time
                                single_embedding = self.embed_query(text)
                                embeddings.append(single_embedding)
                            except Exception as inner_e:
                                logger.error(f"Error processing single text: {inner_e}")
                                # Add zeros as fallback
                                embeddings.append([0.0] * self.embedding_dim)
                    else:
                        # Even batch_size=1 failed, add zeros as fallback
                        logger.error(f"Cannot process even with batch_size=1: {e}")
                        for _ in batch_texts:
                            embeddings.append([0.0] * self.embedding_dim)
                else:
                    # Other error
                    raise
                
            # Log progress
            print(f"   📊 Processed {min(i+batch_size, len(texts))}/{len(texts)} texts ({self.device.upper()})")
            
            # Free GPU memory
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                
        # Convert to lists for consistent output
        result = []
        for emb in embeddings:
            # Convert to list and ensure correct dimension
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = list(emb)
            
            # Check dimensions
            if len(emb_list) != self.vector_size:
                logger.warning(f"Fixing dimension: {len(emb_list)} → {self.vector_size}")
                if len(emb_list) < self.vector_size:
                    # Pad with zeros
                    emb_list = emb_list + [0.0] * (self.vector_size - len(emb_list))
                else:
                    # Truncate
                    emb_list = emb_list[:self.vector_size]
            
            result.append(emb_list)
        
        print(f"✅ Generated {len(result)} embeddings with dimension {self.vector_size}")
        return result
            
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        try:
            print(torch.cuda.device_count())
            print(torch.cuda.get_device_name(0))
            print(torch.cuda.get_device_name(1))
        except Exception as e:
            print(f"Error occurred while accessing GPU: {e}")
        result = self.embed_texts([text], batch_size=1)
        return result[0] if result else []
