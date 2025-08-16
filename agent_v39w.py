import asyncio
import json
import re
import time
import base64
import uuid
import urllib3
import threading
import hashlib
import math
import random
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
from urllib.parse import urlparse, urljoin
import trafilatura
from datetime import datetime
from io import BytesIO
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from bs4 import BeautifulSoup, NavigableString, Tag
import html2text
import markdown

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Image handling
try:
    import PIL.Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not installed. Screenshot display will not be available.")

# DuckDuckGo search
from ddgs import DDGS

# Browser components
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page, Error as PlaywrightError
    from playwright.sync_api import Locator
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- Enhanced Agent Roles (from X-Master paper) ---

class AgentRole(Enum):
    """Agent roles for scattered-and-stacked workflow."""
    SOLVER = "solver"
    CRITIC = "critic"
    REWRITER = "rewriter"
    SELECTOR = "selector"
    MAIN = "main"


@dataclass
class WorkflowStage:
    """Represents a stage in the scattered-and-stacked workflow."""
    stage_name: str
    role: AgentRole
    input_data: Any
    output_data: Any = None
    success: bool = False
    error: Optional[str] = None
    iteration_count: int = 0
    tool_calls: List[Any] = field(default_factory=list)


# --- Sculptor Memory Management Implementation with GigaChat Embeddings ---

@dataclass
class ContextFragment:
    """Represents a fragment of conversation context."""
    id: str
    content: str
    start_marker: str = ""
    end_marker: str = ""
    is_folded: bool = False
    is_summarized: bool = False
    summary: str = ""
    original_content: str = ""
    char_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result in context."""
    fragment_id: str
    content: str
    score: float
    start_pos: int
    end_pos: int
    context_before: str = ""
    context_after: str = ""


class SculptorMemoryManager:
    """
    Sculptor framework implementation for Active Context Management (ACM).
    Based on the research paper: "Sculptor: Empowering LLMs with Cognitive Agency via Active Context Management"
    
    Uses GigaChat API for embeddings instead of OpenAI.
    """
    
    def __init__(self, gigachat_client):
        """Initialize Sculptor memory manager with GigaChat client."""
        self.fragments = {}  # Dict[str, ContextFragment]
        self.conversation_history = []
        self.gigachat_client = gigachat_client
        self.embedding_cache = {}
        self.embedding_model = "EmbeddingsGigaR"  # Use advanced model by default
        
    def _generate_fragment_id(self) -> str:
        """Generate unique 6-character fragment ID."""
        chars = string.ascii_lowercase + string.digits
        return ''.join(random.choice(chars) for _ in range(6))
    
    def _get_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Get text embedding using GigaChat API.
        
        Args:
            text: Text to embed
            is_query: If True, adds instruction for better retrieval quality
        """
        # For EmbeddingsGigaR, add instruction for queries to improve retrieval quality
        if is_query and self.embedding_model == "EmbeddingsGigaR":
            text = f"Дан вопрос, необходимо найти абзац текста с ответом\nвопрос: {text}"
        
        # Check cache
        cache_key = f"{self.embedding_model}:{text}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Use GigaChat embeddings API
            response = self.gigachat_client.get_embeddings(
                texts=[text],
                model=self.embedding_model
            )
            
            if response and 'data' in response and len(response['data']) > 0:
                embedding = np.array(response['data'][0]['embedding'])
                self.embedding_cache[cache_key] = embedding
                return embedding
            else:
                logger.warning("No embedding data received from GigaChat")
                
        except Exception as e:
            logger.warning(f"GigaChat embedding failed: {e}")
        
        # Fallback to TF-IDF
        logger.info("Using TF-IDF fallback for embeddings")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='russian')
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            embedding = tfidf_matrix.toarray()[0]
            self.embedding_cache[cache_key] = embedding
            return embedding
        except:
            return np.zeros(1000)
    
    def fragment_context(self, start_marker: str, end_marker: str, 
                        num_fragments: int = 5, content: str = None) -> Dict[str, Any]:
        """
        Segment conversation into manageable fragments.
        
        Args:
            start_marker: Start boundary for fragmentation
            end_marker: End boundary for fragmentation  
            num_fragments: Number of fragments to create
            content: Content to fragment (uses conversation history if None)
        
        Returns:
            Dict with operation result and fragment IDs
        """
        try:
            if content is None:
                # Use current conversation history
                content = "\n".join([
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in self.conversation_history
                ])
            
            # Find content boundaries
            start_pos = content.find(start_marker) if start_marker else 0
            end_pos = content.find(end_marker, start_pos) if end_marker else len(content)
            
            if start_pos == -1:
                start_pos = 0
            if end_pos == -1:
                end_pos = len(content)
                
            target_content = content[start_pos:end_pos + len(end_marker)]
            
            # Calculate fragment size
            fragment_size = max(1, len(target_content) // num_fragments)
            fragment_ids = []
            
            for i in range(num_fragments):
                fragment_start = i * fragment_size
                fragment_end = min((i + 1) * fragment_size, len(target_content))
                
                if fragment_start >= len(target_content):
                    break
                    
                fragment_content = target_content[fragment_start:fragment_end]
                fragment_id = self._generate_fragment_id()
                
                fragment = ContextFragment(
                    id=fragment_id,
                    content=fragment_content,
                    start_marker=start_marker,
                    end_marker=end_marker,
                    char_count=len(fragment_content),
                    original_content=fragment_content
                )
                
                self.fragments[fragment_id] = fragment
                fragment_ids.append(fragment_id)
            
            logger.info(f"Created {len(fragment_ids)} fragments: {fragment_ids}")
            
            return {
                "success": True,
                "fragment_ids": fragment_ids,
                "total_fragments": len(fragment_ids),
                "content_length": len(target_content)
            }
            
        except Exception as e:
            logger.error(f"Fragment context failed: {e}")
            return {"success": False, "error": str(e)}
    
    def summary_fragment(self, fragment_id: str, model: str = None) -> Dict[str, Any]:
        """
        Generate AI-powered summary of a specific fragment.
        
        Args:
            fragment_id: ID of fragment to summarize
            model: Model to use for summarization (uses GigaChat by default)
        
        Returns:
            Dict with operation result
        """
        try:
            if fragment_id not in self.fragments:
                return {"success": False, "error": f"Fragment {fragment_id} not found"}
            
            fragment = self.fragments[fragment_id]
            
            if fragment.is_summarized:
                return {
                    "success": True,
                    "summary": fragment.summary,
                    "already_summarized": True
                }
            
            # Store original content
            fragment.original_content = fragment.content
            
            # Generate summary using GigaChat
            try:
                summary_prompt = f"Создай краткое резюме следующего текста, сохранив ключевую информацию:\n\n{fragment.content}"
                
                response = self.gigachat_client.chat(
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                
                if response and 'choices' in response and len(response['choices']) > 0:
                    summary = response['choices'][0]['message']['content']
                    summary += f"\n[AI-резюме GigaChat из {len(fragment.content)} символов]"
                else:
                    raise Exception("No response from GigaChat")
                    
            except Exception as e:
                logger.warning(f"GigaChat summarization failed: {e}, using simple summary")
                # Fallback to simple summary
                content_sentences = sent_tokenize(fragment.content)
                if len(content_sentences) <= 2:
                    summary = fragment.content
                else:
                    summary = f"{content_sentences[0]} ... {content_sentences[-1]}"
                    summary += f"\n[Простое резюме из {len(content_sentences)} предложений]"
            
            fragment.summary = summary
            fragment.content = summary
            fragment.is_summarized = True
            
            return {
                "success": True,
                "summary": summary,
                "original_length": len(fragment.original_content),
                "summary_length": len(summary),
                "compression_ratio": 1 - len(summary) / len(fragment.original_content)
            }
            
        except Exception as e:
            logger.error(f"Summary fragment failed: {e}")
            return {"success": False, "error": str(e)}
    
    def revert_summary(self, fragment_id: str) -> Dict[str, Any]:
        """
        Revert fragment from summary to original content.
        
        Args:
            fragment_id: ID of fragment to revert
        
        Returns:
            Dict with operation result
        """
        try:
            if fragment_id not in self.fragments:
                return {"success": False, "error": f"Fragment {fragment_id} not found"}
            
            fragment = self.fragments[fragment_id]
            
            if not fragment.is_summarized:
                return {
                    "success": True,
                    "message": "Fragment is not summarized",
                    "already_original": True
                }
            
            # Restore original content
            fragment.content = fragment.original_content
            fragment.is_summarized = False
            fragment.summary = ""
            
            return {
                "success": True,
                "message": "Fragment reverted to original content",
                "restored_length": len(fragment.content)
            }
            
        except Exception as e:
            logger.error(f"Revert summary failed: {e}")
            return {"success": False, "error": str(e)}
    
    def fold_fragment(self, fragment_id: str) -> Dict[str, Any]:
        """
        Fold fragment to hide from active context.
        
        Args:
            fragment_id: ID of fragment to fold
        
        Returns:
            Dict with operation result
        """
        try:
            if fragment_id not in self.fragments:
                return {"success": False, "error": f"Fragment {fragment_id} not found"}
            
            fragment = self.fragments[fragment_id]
            fragment.is_folded = True
            
            return {
                "success": True,
                "message": f"Fragment {fragment_id} folded",
                "fragment_id": fragment_id
            }
            
        except Exception as e:
            logger.error(f"Fold fragment failed: {e}")
            return {"success": False, "error": str(e)}
    
    def expand_fragment(self, fragment_id: str) -> Dict[str, Any]:
        """
        Expand folded fragment to make it active again.
        
        Args:
            fragment_id: ID of fragment to expand
        
        Returns:
            Dict with operation result
        """
        try:
            if fragment_id not in self.fragments:
                return {"success": False, "error": f"Fragment {fragment_id} not found"}
            
            fragment = self.fragments[fragment_id]
            fragment.is_folded = False
            
            return {
                "success": True,
                "message": f"Fragment {fragment_id} expanded",
                "fragment_id": fragment_id,
                "content_length": len(fragment.content)
            }
            
        except Exception as e:
            logger.error(f"Expand fragment failed: {e}")
            return {"success": False, "error": str(e)}
    
    def restore_context(self, fragment_ids: List[str] = None) -> Dict[str, Any]:
        """
        Restore fragments to original state.
        
        Args:
            fragment_ids: List of fragment IDs to restore (all if None)
        
        Returns:
            Dict with operation result
        """
        try:
            target_fragments = fragment_ids or list(self.fragments.keys())
            restored_count = 0
            
            for fragment_id in target_fragments:
                if fragment_id in self.fragments:
                    fragment = self.fragments[fragment_id]
                    fragment.is_folded = False
                    
                    if fragment.is_summarized and fragment.original_content:
                        fragment.content = fragment.original_content
                        fragment.is_summarized = False
                        fragment.summary = ""
                    
                    restored_count += 1
            
            return {
                "success": True,
                "message": f"Restored {restored_count} fragments",
                "restored_count": restored_count,
                "total_fragments": len(self.fragments)
            }
            
        except Exception as e:
            logger.error(f"Restore context failed: {e}")
            return {"success": False, "error": str(e)}
    
    def search_context(self, query: str, mode: str = "semantic", 
                      target: str = "all", similarity_threshold: float = 0.3,
                      max_results: int = 5) -> Dict[str, Any]:
        """
        Search across conversation context using GigaChat embeddings.
        
        Args:
            query: Search query
            mode: "exact" or "semantic"
            target: "user", "assistant", or "all"
            similarity_threshold: Minimum similarity for semantic search
            max_results: Maximum number of results
            
        Returns:
            Dict with search results
        """
        try:
            results = []
            
            # Search in fragments
            for fragment_id, fragment in self.fragments.items():
                if fragment.is_folded:
                    continue  # Skip folded fragments
                    
                content = fragment.content
                
                if mode == "exact":
                    # Exact string matching
                    if query.lower() in content.lower():
                        # Find all occurrences
                        start = 0
                        while True:
                            pos = content.lower().find(query.lower(), start)
                            if pos == -1:
                                break
                            results.append(SearchResult(
                                fragment_id=fragment_id,
                                content=content[max(0, pos-50):pos+len(query)+50],
                                score=1.0,
                                start_pos=pos,
                                end_pos=pos + len(query)
                            ))
                            start = pos + 1
                            
                elif mode == "semantic":
                    # Semantic similarity search using GigaChat embeddings
                    query_embedding = self._get_embedding(query, is_query=True)
                    content_embedding = self._get_embedding(content, is_query=False)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, content_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding) + 1e-8
                    )
                    
                    if similarity >= similarity_threshold:
                        results.append(SearchResult(
                            fragment_id=fragment_id,
                            content=content[:200] + "..." if len(content) > 200 else content,
                            score=float(similarity),
                            start_pos=0,
                            end_pos=len(content)
                        ))
            
            # Sort by score and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:max_results]
            
            return {
                "success": True,
                "query": query,
                "mode": mode,
                "embedding_model": self.embedding_model,
                "results": [
                    {
                        "fragment_id": r.fragment_id,
                        "content": r.content,
                        "score": r.score,
                        "start_pos": r.start_pos,
                        "end_pos": r.end_pos
                    }
                    for r in results
                ],
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Search context failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_search_detail(self, fragment_id: str, start_pos: int = 0, end_pos: int = None) -> Dict[str, Any]:
        """
        Get detailed content from a specific fragment.
        
        Args:
            fragment_id: ID of fragment
            start_pos: Start position in content
            end_pos: End position in content
        
        Returns:
            Dict with detailed content
        """
        try:
            if fragment_id not in self.fragments:
                return {"success": False, "error": f"Fragment {fragment_id} not found"}
            
            fragment = self.fragments[fragment_id]
            content = fragment.content
            
            if end_pos is None:
                end_pos = len(content)
            
            detailed_content = content[start_pos:end_pos]
            
            return {
                "success": True,
                "fragment_id": fragment_id,
                "content": detailed_content,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "full_length": len(content),
                "is_summarized": fragment.is_summarized,
                "is_folded": fragment.is_folded
            }
            
        except Exception as e:
            logger.error(f"Get search detail failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_context_state(self) -> Dict[str, Any]:
        """
        Get current state of context management.
        
        Returns:
            Dict with context state information
        """
        try:
            total_fragments = len(self.fragments)
            active_fragments = sum(1 for f in self.fragments.values() if not f.is_folded)
            summarized_fragments = sum(1 for f in self.fragments.values() if f.is_summarized)
            total_content_length = sum(len(f.content) for f in self.fragments.values())
            
            fragment_details = []
            for fragment_id, fragment in self.fragments.items():
                fragment_details.append({
                    "id": fragment_id,
                    "length": len(fragment.content),
                    "is_folded": fragment.is_folded,
                    "is_summarized": fragment.is_summarized,
                    "created_at": fragment.created_at.isoformat(),
                    "start_marker": fragment.start_marker,
                    "end_marker": fragment.end_marker
                })
            
            return {
                "success": True,
                "total_fragments": total_fragments,
                "active_fragments": active_fragments,
                "summarized_fragments": summarized_fragments,
                "total_content_length": total_content_length,
                "embedding_model": self.embedding_model,
                "embedding_cache_size": len(self.embedding_cache),
                "conversation_history_length": len(self.conversation_history),
                "fragments": fragment_details
            }
            
        except Exception as e:
            logger.error(f"Get context state failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_optimized_context(self) -> str:
        """
        Get optimized context by combining active fragments.
        
        Returns:
            Optimized context string
        """
        try:
            active_fragments = [
                f for f in self.fragments.values() 
                if not f.is_folded
            ]
            
            # Sort by creation time
            active_fragments.sort(key=lambda x: x.created_at)
            
            # Combine content
            context_parts = []
            for fragment in active_fragments:
                context_parts.append(fragment.content)
            
            optimized_context = "\n\n".join(context_parts)
            
            return optimized_context
            
        except Exception as e:
            logger.error(f"Get optimized context failed: {e}")
            return ""


# --- D2Snap Implementation ---

class D2SnapProcessor:
    """D2Snap: DOM Downsampling processor based on research paper."""
    
    # Element classifications and ratings based on UI semantics
    ELEMENT_CLASSIFICATIONS = {
        # Container elements (hierarchy and layout)
        'article': ('container', 0.95),
        'aside': ('container', 0.85),
        'body': ('container', 0.90),
        'div': ('container', 0.30),
        'footer': ('container', 0.70),
        'header': ('container', 0.75),
        'main': ('container', 0.85),
        'nav': ('container', 0.80),
        'section': ('container', 0.90),
        
        # Interactive elements (user actions)
        'a': ('interactive', 0.85),
        'button': ('interactive', 0.80),
        'details': ('interactive', 0.60),
        'form': ('interactive', 0.75),
        'input': ('interactive', 0.70),
        'label': ('interactive', 0.50),
        'select': ('interactive', 0.65),
        'summary': ('interactive', 0.55),
        'textarea': ('interactive', 0.65),
        'option': ('interactive', 0.60),
        
        # Content elements (text formatting)
        'address': ('content', 0.60),
        'b': ('content', 0.40),
        'blockquote': ('content', 0.65),
        'code': ('content', 0.60),
        'em': ('content', 0.50),
        'figure': ('content', 0.50),
        'figcaption': ('content', 0.45),
        'h1': ('content', 1.00),
        'h2': ('content', 0.95),
        'h3': ('content', 0.90),
        'h4': ('content', 0.85),
        'h5': ('content', 0.80),
        'h6': ('content', 0.75),
        'hr': ('content', 0.20),
        'img': ('content', 0.60),
        'li': ('content', 0.60),
        'ol': ('content', 0.55),
        'p': ('content', 0.60),
        'pre': ('content', 0.55),
        'small': ('content', 0.30),
        'span': ('content', 0.20),
        'strong': ('content', 0.50),
        'sub': ('content', 0.25),
        'sup': ('content', 0.25),
        'table': ('content', 0.70),
        'tbody': ('content', 0.65),
        'td': ('content', 0.50),
        'th': ('content', 0.65),
        'tr': ('content', 0.50),
        'ul': ('content', 0.55),
        'thead': ('content', 0.65),
        'tfoot': ('content', 0.60),
        
        # Other elements (low UI importance)
        'base': ('other', 0.10),
        'br': ('other', 0.05),
        'canvas': ('other', 0.20),
        'head': ('other', 0.10),
        'html': ('other', 0.10),
        'link': ('other', 0.05),
        'meta': ('other', 0.00),
        'noscript': ('other', 0.05),
        'script': ('other', 0.00),
        'source': ('other', 0.05),
        'style': ('other', 0.00),
        'template': ('other', 0.00),
        'title': ('other', 0.40),
        'track': ('other', 0.05),
        'video': ('other', 0.50),
        'audio': ('other', 0.40),
        'iframe': ('other', 0.30),
        'embed': ('other', 0.30),
        'object': ('other', 0.30)
    }
    
    # Attribute ratings based on UI importance
    ATTRIBUTE_RATINGS = {
        'alt': 0.9, 'href': 0.9, 'src': 0.8, 'id': 0.8, 'class': 0.7,
        'title': 0.6, 'lang': 0.6, 'role': 0.6, 'placeholder': 0.5,
        'label': 0.5, 'for': 0.5, 'value': 0.5, 'checked': 0.5,
        'disabled': 0.5, 'readonly': 0.5, 'required': 0.5, 'maxlength': 0.5,
        'minlength': 0.5, 'pattern': 0.5, 'step': 0.5, 'min': 0.5, 'max': 0.5,
        'accept': 0.4, 'action': 0.4, 'method': 0.4, 'target': 0.4, 'rel': 0.4,
        'type': 0.3, 'name': 0.3, 'hidden': 0.1, 'style': 0.1
    }
    
    def __init__(self):
        """Initialize D2Snap processor with HTML to Markdown converter."""
        self.h2m = html2text.HTML2Text()
        self.h2m.ignore_links = False
        self.h2m.ignore_images = False
        self.h2m.body_width = 0  # No line wrapping
        self.h2m.unicode_snob = True
        
    def process(self, dom_html: str, k: float = 0.3, l: float = 0.3, m: float = 0.3) -> str:
        """Process DOM with D2Snap algorithm."""
        try:
            soup = BeautifulSoup(dom_html, 'html.parser')
            self._process_node(soup, k, l, m)
            return str(soup)
        except Exception as e:
            logger.error(f"D2Snap processing failed: {e}")
            return dom_html  # Return original on error
    
    def adaptive_process(self, dom_html: str, max_tokens: int = 8192, 
                        max_iterations: int = 10) -> str:
        """Adaptively downsample DOM to fit within token limit."""
        estimated_tokens = self._estimate_tokens(dom_html)
        
        if estimated_tokens <= max_tokens:
            return dom_html
        
        # Halton sequence for low-discrepancy parameter exploration
        def halton_sequence(index: int, base: int) -> float:
            result = 0
            f = 1 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f = f / base
            return result
        
        # Progressive downsampling
        magnitude = estimated_tokens / 1000000  # Based on 1MB soft limit
        
        for i in range(1, max_iterations + 1):
            k = min(magnitude * halton_sequence(i, 2), 1.0)
            l = min(magnitude * halton_sequence(i, 3), 1.0)
            m = min(magnitude * halton_sequence(i, 5), 1.0)
            
            downsampled = self.process(dom_html, k, l, m)
            
            current_tokens = self._estimate_tokens(downsampled)
            if current_tokens <= max_tokens:
                logger.info(f"D2Snap achieved target size in {i} iterations")
                return downsampled
            
            magnitude *= 1.125
        
        logger.warning("D2Snap max iterations reached, applying aggressive downsampling")
        return self.process(dom_html, 0.9, 0.9, 0.9)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        return len(text) // 4
    
    def _process_node(self, node, k: float, l: float, m: float):
        """Recursively process DOM node with D2Snap algorithm."""
        if isinstance(node, NavigableString):
            if node.parent and node.strip():
                downsampled_text = self._downsample_text(str(node), l)
                node.replace_with(downsampled_text)
            return
        
        if not isinstance(node, Tag):
            return
        
        # Process children first (post-order)
        for child in list(node.children):
            self._process_node(child, k, l, m)
        
        # Process element node
        tag_name = node.name.lower()
        classification, rating = self.ELEMENT_CLASSIFICATIONS.get(tag_name, ('other', 0.0))
        
        if classification == 'container':
            self._process_container(node, k, rating)
        elif classification == 'content':
            self._process_content(node)
        elif classification == 'interactive':
            self._filter_attributes(node, m)
        elif classification == 'other':
            if rating < 0.2:
                node.decompose()
            else:
                self._filter_attributes(node, m)
        
        # Filter attributes for remaining elements
        if node.parent:  # Check if node wasn't decomposed
            self._filter_attributes(node, m)
    
    def _downsample_text(self, text: str, l: float) -> str:
        """Downsample text using TextRank algorithm."""
        if not text or l == 0:
            return text
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= 1:
                return text
            
            # TextRank implementation
            scores = self._textrank_sentences(sentences)
            
            # Select top sentences
            num_keep = max(1, int(len(sentences) * (1 - l)))
            ranked_sentences = sorted(
                zip(sentences, scores),
                key=lambda x: x[1],
                reverse=True
            )[:num_keep]
            
            # Maintain original order
            selected = [s for s in sentences if any(s == rs[0] for rs in ranked_sentences)]
            return ' '.join(selected)
            
        except Exception as e:
            logger.debug(f"Text downsampling failed: {e}")
            return text
    
    def _textrank_sentences(self, sentences: List[str]) -> List[float]:
        """Calculate TextRank scores for sentences."""
        if len(sentences) <= 1:
            return [1.0] * len(sentences)
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            scores = np.ones(len(sentences))
            damping = 0.85
            
            # Power iteration for PageRank
            for _ in range(10):
                new_scores = np.zeros(len(sentences))
                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        if i != j and similarity_matrix[j][i] > 0:
                            new_scores[i] += (
                                similarity_matrix[j][i] * scores[j] /
                                similarity_matrix[j].sum()
                            )
                scores = (1 - damping) + damping * new_scores
            
            return scores.tolist()
            
        except Exception as e:
            logger.debug(f"TextRank calculation failed: {e}")
            return [1.0] * len(sentences)
    
    def _filter_attributes(self, node: Tag, m: float):
        """Filter attributes based on importance threshold."""
        if not hasattr(node, 'attrs'):
            return
        
        attrs_to_remove = []
        for attr_name in node.attrs:
            if attr_name.startswith('aria-'):
                rating = 0.6
            elif attr_name.startswith('data-'):
                rating = 0.1
            else:
                rating = self.ATTRIBUTE_RATINGS.get(attr_name, 0.0)
            
            if rating < m:
                attrs_to_remove.append(attr_name)
        
        for attr in attrs_to_remove:
            del node.attrs[attr]
    
    def _process_container(self, node: Tag, k: float, rating: float):
        """Process container element with hierarchical merging."""
        depth = self._get_depth(node)
        max_depth = self._get_max_depth(node)
        
        if max_depth > 0:
            merge_threshold = k * max_depth
            
            if depth % max(1, int(merge_threshold)) == 0:
                return
            
            parent = node.parent
            if parent and isinstance(parent, Tag):
                parent_class = self.ELEMENT_CLASSIFICATIONS.get(
                    parent.name.lower(), ('other', 0.0)
                )[0]
                
                if parent_class == 'container':
                    self._merge_nodes(node, parent)
    
    def _process_content(self, node: Tag):
        """Convert content element to Markdown representation."""
        try:
            if node.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(node.name[1])
                text = node.get_text(strip=True)
                markdown_text = '#' * level + ' ' + text
                node.replace_with(NavigableString(markdown_text))
            elif node.name in ['table', 'ul', 'ol', 'blockquote', 'pre', 'code']:
                html_content = str(node)
                markdown_text = self.h2m.handle(html_content)
                node.replace_with(NavigableString(markdown_text))
            elif node.name in ['b', 'strong']:
                text = node.get_text(strip=True)
                node.replace_with(NavigableString(f"**{text}**"))
            elif node.name in ['i', 'em']:
                text = node.get_text(strip=True)
                node.replace_with(NavigableString(f"*{text}*"))
            elif node.name == 'p':
                text = node.get_text(strip=True)
                if text:
                    node.replace_with(NavigableString(text + '\n'))
        except Exception as e:
            logger.debug(f"Content processing failed for {node.name}: {e}")
    
    def _merge_nodes(self, source: Tag, target: Tag):
        """Merge source node into target node."""
        for child in list(source.children):
            target.append(child)
        
        for attr, value in source.attrs.items():
            if attr not in target.attrs:
                target.attrs[attr] = value
        
        source.decompose()
    
    def _get_depth(self, node: Tag) -> int:
        """Get depth of node in DOM tree."""
        depth = 0
        current = node.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    def _get_max_depth(self, node: Tag) -> int:
        """Get maximum depth of subtree rooted at node."""
        if not list(node.children):
            return 0
        
        max_child_depth = 0
        for child in node.children:
            if isinstance(child, Tag):
                max_child_depth = max(max_child_depth, self._get_max_depth(child))
        
        return max_child_depth + 1


# --- Data Classes ---

@dataclass
class FunctionCall:
    """Represents a function call made by the agent."""
    name: str
    arguments: Dict[str, Any]
    iteration: int = 0
    role: AgentRole = AgentRole.MAIN


@dataclass
class FunctionResult:
    """Represents the result of a function execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    iteration: int = 0
    role: AgentRole = AgentRole.MAIN


@dataclass
class BrowserState:
    """Represents the current state of the browser."""
    url: str
    title: str
    cookies: List[Dict]
    local_storage: Dict[str, str]
    session_storage: Dict[str, str]
    viewport: Dict[str, int]
    screenshot: Optional[bytes] = None
    html: Optional[str] = None
    text: Optional[str] = None
    dom_snapshot: Optional[str] = None  # D2Snap processed DOM


@dataclass
class AgentResponse:
    """Represents a complete response from an agent."""
    agent_name: str
    role: str
    content: str
    thinking_process: List[Dict] = field(default_factory=list)
    function_calls: List[FunctionCall] = field(default_factory=list)
    function_results: List[FunctionResult] = field(default_factory=list)
    confidence: float = 0.0
    error: Optional[str] = None
    final_answer: Optional[str] = None
    functions_state_id: Optional[str] = None
    sculptor_state: Optional[Dict[str, Any]] = None
    workflow_stages: List[WorkflowStage] = field(default_factory=list)
    total_iterations: int = 0
    scattered_solutions: List[str] = field(default_factory=list)
    selected_solution: Optional[str] = None


# --- Enhanced GigaChat API Client with Embeddings Support ---

class GigaChatClient:
    """A client for interacting with the GigaChat API with function calling and embeddings support."""

    def __init__(self, client_id: str, client_secret: str, verify_ssl: bool = False, model: str = "GigaChat-2-Max"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.verify_ssl = verify_ssl
        self.model = model
        self.access_token = None
        self.token_expires_at = 0
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self._lock = threading.Lock()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Creates a requests session with a retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_token(self):
        """Obtains an access token for the GigaChat API."""
        with self._lock:
            if self.access_token and time.time() < self.token_expires_at - 60:
                return self.access_token

            credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers = {
                'Authorization': f'Basic {credentials}',
                'RqUID': str(uuid.uuid4()),
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = {'scope': 'GIGACHAT_API_CORP'}

            try:
                response = self.session.post(self.auth_url, headers=headers, data=data, verify=self.verify_ssl, timeout=20)
                response.raise_for_status()
                token_data = response.json()
                self.access_token = token_data['access_token']
                if 'expires_at' in token_data:
                    self.token_expires_at = int(token_data['expires_at']) // 1000
                else:
                    self.token_expires_at = int(time.time() + int(token_data.get('expires_in', 1800)))
                logger.info("Successfully obtained a new GigaChat access token.")
                return self.access_token
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get GigaChat token: {e}")
                raise

    def chat(self, messages: List[Dict], functions: Optional[List[Dict]] = None, temperature: float = 0.3,
             max_tokens: int = 4096, function_call: Union[str, Dict] = "auto"):
        """Sends a chat request to the GigaChat API with proper function calling support."""
        token = self._get_token()
        url = f"{self.base_url}/chat/completions"
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Add function calling parameters
        if functions:
            data["functions"] = functions
            data["function_call"] = function_call

        try:
            response = self.session.post(url, headers=headers, json=data, verify=self.verify_ssl, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GigaChat chat request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise

    def get_embeddings(self, texts: List[str], model: str = "EmbeddingsGigaR") -> Dict[str, Any]:
        """
        Get embeddings for texts using GigaChat API.
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use ("Embeddings" or "EmbeddingsGigaR")
            
        Returns:
            Response dictionary with embeddings
        """
        token = self._get_token()
        url = f"{self.base_url}/embeddings"
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        
        data = {
            "model": model,
            "input": texts
        }

        try:
            response = self.session.post(url, headers=headers, json=data, verify=self.verify_ssl, timeout=60)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Successfully got embeddings for {len(texts)} texts using {model}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"GigaChat embeddings request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise


# --- Enhanced Browser with D2Snap Integration ---

class D2SnapBrowser:
    """Enhanced browser with D2Snap DOM optimization."""
    
    def __init__(self):
        """Initialize Enhanced Browser with D2Snap."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not installed. Run: pip install playwright && playwright install chromium")
        
        self.browser = None
        self.context = None
        self.page = None
        self.current_url = None
        self.history = []
        self.page_state = None
        self.d2snap = D2SnapProcessor()
        
        # D2Snap configuration
        self.d2snap_config = {
            'enabled': True,
            'adaptive': True,
            'max_tokens': 8192,
            'default_k': 0.3,
            'default_l': 0.3,
            'default_m': 0.3
        }
        
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        ]
        
    def configure_d2snap(self, enabled: bool = True, adaptive: bool = True, 
                        max_tokens: int = 8192, k: float = 0.3, 
                        l: float = 0.3, m: float = 0.3):
        """Configure D2Snap settings."""
        self.d2snap_config = {
            'enabled': enabled,
            'adaptive': adaptive,
            'max_tokens': max_tokens,
            'default_k': k,
            'default_l': l,
            'default_m': m
        }
        logger.info(f"D2Snap configured: {self.d2snap_config}")
    
    def start_session(self, headless: bool = True, viewport: Dict = None, **kwargs) -> FunctionResult:
        """Start a new browser session with enhanced capabilities."""
        try:
            if self.browser:
                self.close_session()
            
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-extensions',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--start-maximized',
                    '--disable-setuid-sandbox',
                    '--disable-accelerated-2d-canvas',
                    '--disable-infobars',
                    '--window-position=0,0',
                    '--ignore-certificate-errors',
                    '--ignore-certificate-errors-skip-list'
                ]
            )
            
            viewport = viewport or {'width': 1920, 'height': 1080}
            self.context = self.browser.new_context(
                user_agent=random.choice(self.user_agents),
                viewport=viewport,
                device_scale_factor=1,
                is_mobile=False,
                has_touch=False,
                locale='ru-RU',
                timezone_id='Europe/Moscow',
                accept_downloads=True,
                ignore_https_errors=True,
                bypass_csp=True,
                java_script_enabled=True,
                permissions=['geolocation', 'notifications'],
                color_scheme='light',
                reduced_motion='reduce'
            )
            
            self.page = self.context.new_page()
            
            # Enhanced stealth mode
            self.page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        {0: {type: "application/x-google-chrome-pdf", suffixes: "pdf"}},
                        {0: {type: "application/pdf", suffixes: "pdf"}}
                    ]
                });
                Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU', 'ru', 'en-US', 'en']});
                window.chrome = {runtime: {}, loadTimes: function() {}, csi: function() {}};
            """)
            
            self.page.set_default_timeout(30000)
            self.page.set_default_navigation_timeout(30000)
            
            logger.info("D2Snap-enhanced browser session started successfully")
            return FunctionResult(
                success=True, 
                data="Browser session started with D2Snap optimization",
                metadata={"viewport": viewport, "headless": headless, "d2snap": self.d2snap_config}
            )
            
        except Exception as e:
            error_msg = f"Failed to start browser session: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)
    
    def get_dom_snapshot(self, use_d2snap: bool = None) -> FunctionResult:
        """Get DOM snapshot with optional D2Snap optimization."""
        try:
            if not self.page:
                return FunctionResult(success=False, data=None, error="Browser session not started")
            
            raw_dom = self.page.content()
            
            if use_d2snap is None:
                use_d2snap = self.d2snap_config['enabled']
            
            if not use_d2snap:
                return FunctionResult(
                    success=True,
                    data=raw_dom,
                    metadata={'type': 'raw_dom', 'size': len(raw_dom)}
                )
            
            if self.d2snap_config['adaptive']:
                processed_dom = self.d2snap.adaptive_process(
                    raw_dom,
                    max_tokens=self.d2snap_config['max_tokens']
                )
            else:
                processed_dom = self.d2snap.process(
                    raw_dom,
                    k=self.d2snap_config['default_k'],
                    l=self.d2snap_config['default_l'],
                    m=self.d2snap_config['default_m']
                )
            
            compression_ratio = (1 - len(processed_dom) / len(raw_dom)) * 100
            
            logger.info(f"D2Snap compression: {compression_ratio:.1f}% reduction")
            
            return FunctionResult(
                success=True,
                data=processed_dom,
                metadata={
                    'type': 'd2snap_dom',
                    'original_size': len(raw_dom),
                    'compressed_size': len(processed_dom),
                    'compression_ratio': compression_ratio,
                    'estimated_tokens': self.d2snap._estimate_tokens(processed_dom)
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to get DOM snapshot: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)
    
    def navigate(self, url: str = None, **kwargs) -> FunctionResult:
        """Navigate to URL with D2Snap-optimized DOM capture."""
        try:
            if not self.page:
                return FunctionResult(success=False, data=None, error="Browser session not started")
            
            logger.info(f"Navigating to: {url}")
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            self.page.wait_for_timeout(1000)
            
            dom_result = self.get_dom_snapshot()
            
            self.current_url = self.page.url
            title = self.page.title() or "No title"
            
            if dom_result.success:
                self.page_state = BrowserState(
                    url=self.current_url,
                    title=title,
                    cookies=self.context.cookies() if self.context else [],
                    local_storage={},
                    session_storage={},
                    viewport=self.page.viewport_size,
                    dom_snapshot=dom_result.data
                )
            
            return FunctionResult(
                success=True,
                data=f"Successfully navigated to {url}",
                metadata={
                    "url": self.current_url,
                    "title": title,
                    "status": response.status if response else None,
                    "dom_snapshot": dom_result.metadata if dom_result.success else None
                }
            )
            
        except Exception as e:
            error_msg = f"Navigation failed: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)
    
    def smart_extract(self, target: Optional[str] = None, use_d2snap: bool = True, **kwargs) -> FunctionResult:
        """Extract content with D2Snap optimization."""
        try:
            if not self.page:
                return FunctionResult(success=False, data=None, error="Browser session not started")
            
            logger.info(f"Smart extracting with D2Snap: {target if target else 'all content'}")
            
            dom_result = self.get_dom_snapshot(use_d2snap=use_d2snap)
            
            if not dom_result.success:
                return dom_result
            
            if target:
                soup = BeautifulSoup(dom_result.data, 'html.parser')
                
                target_element = None
                
                # By ID
                if not target_element:
                    target_element = soup.find(id=target)
                
                # By class
                if not target_element:
                    target_element = soup.find(class_=target)
                
                # By text content
                if not target_element:
                    target_element = soup.find(text=re.compile(target, re.I))
                    if target_element:
                        target_element = target_element.parent
                
                if target_element:
                    extracted_content = str(target_element)
                else:
                    extracted_content = f"Target '{target}' not found in optimized DOM"
            else:
                extracted_content = dom_result.data
            
            return FunctionResult(
                success=True,
                data=extracted_content,
                metadata={
                    'extraction_target': target,
                    'd2snap_applied': use_d2snap,
                    'content_length': len(extracted_content),
                    'dom_metadata': dom_result.metadata
                }
            )
            
        except Exception as e:
            error_msg = f"Smart extract failed: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)
    
    def smart_click(self, identifier: str, **kwargs) -> FunctionResult:
        """Smart click functionality (placeholder)."""
        try:
            if not self.page:
                return FunctionResult(success=False, data=None, error="Browser session not started")
            
            # This is a placeholder implementation
            # In a full implementation, you would add smart element detection and clicking
            logger.info(f"Smart click on: {identifier}")
            
            return FunctionResult(
                success=True,
                data=f"Smart click executed on {identifier}",
                metadata={'identifier': identifier}
            )
            
        except Exception as e:
            error_msg = f"Smart click failed: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)
    
    def analyze_structure(self, **kwargs) -> FunctionResult:
        """Analyze page structure (placeholder)."""
        try:
            if not self.page:
                return FunctionResult(success=False, data=None, error="Browser session not started")
            
            # This is a placeholder implementation
            logger.info("Analyzing page structure")
            
            return FunctionResult(
                success=True,
                data="Page structure analysis completed",
                metadata={'analysis': 'structure_data'}
            )
            
        except Exception as e:
            error_msg = f"Structure analysis failed: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)
    
    def close_session(self, **kwargs) -> FunctionResult:
        """Close the browser session."""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if hasattr(self, 'playwright'):
                self.playwright.stop()
            
            self.page = None
            self.context = None
            self.browser = None
            self.current_url = None
            self.history = []
            self.page_state = None
            
            logger.info("Browser session closed")
            return FunctionResult(
                success=True,
                data="Browser session closed",
                metadata={}
            )
            
        except Exception as e:
            error_msg = f"Failed to close browser session: {str(e)}"
            logger.error(error_msg)
            return FunctionResult(success=False, data=None, error=error_msg)


# --- Enhanced Function Executor with X-Master capabilities ---

class XMasterFunctionExecutor:
    """Enhanced function executor with X-Master tool-augmented reasoning capabilities."""

    def __init__(self, gigachat_client):
        """Initialize with GigaChat client for embeddings and summaries."""
        self.browser = D2SnapBrowser() if PLAYWRIGHT_AVAILABLE else None
        self.sculptor = SculptorMemoryManager(gigachat_client)
        self.gigachat_client = gigachat_client
        self.final_answer = None
        self.browser_started = False
        
        # X-Master enhancements
        self.execution_history = []
        self.tool_usage_stats = defaultdict(int)
        self.iteration_count = 0
        self.max_iterations = 50  # Increased for long chains
        
        # Code execution environment
        self.code_globals = {
            '__builtins__': __builtins__,
            'json': json,
            'math': math,
            'random': random,
            're': re,
            'time': time,
            'datetime': datetime,
            'logging': logging,
            'np': np if 'np' in globals() else None,
            'web_search': self.web_search,
            'web_parse': self.web_parse,
        }

    def execute_code(self, code: str, iteration: int = 0, role: AgentRole = AgentRole.MAIN) -> FunctionResult:
        """Execute Python code in a sandbox environment (X-Master style)."""
        start_time = time.time()
        
        try:
            # Create isolated namespace
            local_namespace = self.code_globals.copy()
            
            # Execute code
            exec(code, local_namespace)
            
            # Capture any printed output
            # Note: In a full implementation, you'd redirect stdout/stderr
            result_data = "Code executed successfully"
            
            execution_time = time.time() - start_time
            
            return FunctionResult(
                success=True,
                data=result_data,
                metadata={
                    'execution_time': execution_time,
                    'code_length': len(code),
                    'iteration': iteration,
                    'role': role.value
                },
                execution_time=execution_time,
                iteration=iteration,
                role=role
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Code execution failed: {str(e)}"
            logger.error(error_msg)
            
            return FunctionResult(
                success=False,
                data=None,
                error=error_msg,
                metadata={
                    'execution_time': execution_time,
                    'code_length': len(code),
                    'iteration': iteration,
                    'role': role.value
                },
                execution_time=execution_time,
                iteration=iteration,
                role=role
            )

    def web_search(
        self,
        query: str = None,
        max_results: int = 7,
        region: str = 'wt-wt',
        timelimit: str = 'y',
        retries: int = 3,
        **kwargs
    ) -> FunctionResult:
        """Enhanced web search with X-Master capabilities.

        Args:
            query: Search query string.
            max_results: Maximum number of results to fetch.
            region: Regional setting for the search.
            timelimit: Time limit filter for search results.
            retries: Number of retry attempts for the search request.
        """
        if not query:
            return FunctionResult(
                success=False,
                data=None,
                error="No search query provided.",
            )

        start_time = time.time()
        self.tool_usage_stats['web_search'] += 1

        logger.info(f"Performing X-Master web search for: '{query}'")

        results = []
        last_exception = None
        attempt = 0
        for attempt in range(1, retries + 1):
            try:
                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(
                            query,
                            region=region,
                            safesearch='off',
                            timelimit=timelimit,
                            max_results=max_results,
                        )
                    )
                break
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"DDGS search attempt {attempt} failed: {e}"
                )
                if attempt < retries:
                    time.sleep(1 * attempt)

        if last_exception and not results:
            execution_time = time.time() - start_time
            error_msg = (
                f"X-Master web search failed after {attempt} attempts: {last_exception}"
            )
            logger.error(error_msg)
            return FunctionResult(
                success=False,
                data=None,
                error=error_msg,
                metadata={
                    'query': query,
                    'attempts': attempt,
                    'retries': retries,
                    'max_results': max_results,
                    'region': region,
                    'timelimit': timelimit,
                    'tool': 'web_search',
                    'execution_time': execution_time,
                },
                execution_time=execution_time,
            )

        if not results:
            execution_time = time.time() - start_time
            return FunctionResult(
                success=True,
                data="No search results found.",
                metadata={
                    'query': query,
                    'results_count': 0,
                    'attempts': attempt,
                    'retries': retries,
                    'max_results': max_results,
                    'region': region,
                    'timelimit': timelimit,
                    'tool': 'web_search',
                    'execution_time': execution_time,
                },
                execution_time=execution_time,
            )

        # Enhanced result formatting for X-Master
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = {
                "rank": i,
                "title": result.get("title", ""),
                "link": result.get("href", ""),
                "snippet": result.get("body", ""),
                "relevance_score": 1.0 - (i * 0.1)  # Simple relevance scoring
            }
            formatted_results.append(formatted_result)

        execution_time = time.time() - start_time

        return FunctionResult(
            success=True,
            data=json.dumps(formatted_results, indent=2, ensure_ascii=False),
            metadata={
                'query': query,
                'results_count': len(formatted_results),
                'execution_time': execution_time,
                'attempts': attempt,
                'retries': retries,
                'max_results': max_results,
                'region': region,
                'timelimit': timelimit,
                'tool': 'web_search',
            },
            execution_time=execution_time
        )

    def web_parse(self, url: str, query: str = None, **kwargs) -> FunctionResult:
        """Enhanced web parsing with X-Master capabilities."""
        if not url:
            return FunctionResult(
                success=False,
                data=None,
                error="No URL provided for parsing."
            )
        
        start_time = time.time()
        self.tool_usage_stats['web_parse'] += 1
        
        logger.info(f"X-Master web parse: {url}")
        
        try:
            # Use trafilatura for content extraction
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return FunctionResult(
                    success=False,
                    data=None,
                    error=f"Failed to download content from {url}"
                )
            
            # Extract main content
            content = trafilatura.extract(downloaded, include_links=True, include_tables=True)
            
            if not content:
                return FunctionResult(
                    success=False,
                    data=None,
                    error=f"No content extracted from {url}"
                )
            
            # If query provided, try to find relevant sections
            if query:
                sentences = sent_tokenize(content)
                relevant_sentences = []
                query_lower = query.lower()
                
                for sentence in sentences:
                    if any(word in sentence.lower() for word in query_lower.split()):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    content = "\n".join(relevant_sentences[:5])  # Top 5 relevant sentences
            
            execution_time = time.time() - start_time
            
            return FunctionResult(
                success=True,
                data=content,
                metadata={
                    'url': url,
                    'query': query,
                    'content_length': len(content),
                    'execution_time': execution_time,
                    'tool': 'web_parse'
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"X-Master web parse failed: {e}"
            logger.error(error_msg)
            return FunctionResult(
                success=False, 
                data=None, 
                error=error_msg,
                execution_time=execution_time
            )

    def browser_action(self, action: str, **kwargs) -> FunctionResult:
        """Execute browser action with D2Snap optimization."""
        if not self.browser:
            return FunctionResult(success=False, data=None, error="Browser not available")
        
        start_time = time.time()
        self.tool_usage_stats['browser'] += 1
        
        # Auto-start browser if needed
        if action != "start" and not self.browser_started:
            logger.info("Auto-starting D2Snap browser session")
            start_result = self.browser.start_session()
            if not start_result.success:
                return start_result
            self.browser_started = True
        
        # Map actions to methods
        action_map = {
            "start": lambda: self._handle_start(**kwargs),
            "navigate": lambda: self.browser.navigate(**kwargs),
            "click": lambda: self.browser.smart_click(**kwargs),
            "extract": lambda: self.browser.smart_extract(**kwargs),
            "structure": lambda: self.browser.analyze_structure(**kwargs),
            "dom_snapshot": lambda: self.browser.get_dom_snapshot(**kwargs),
            "configure_d2snap": lambda: self._configure_d2snap(**kwargs),
            "close": lambda: self._handle_close(**kwargs)
        }
        
        if action in action_map:
            try:
                result = action_map[action]()
                result.execution_time = time.time() - start_time
                return result
            except Exception as e:
                logger.error(f"Error in {action}: {e}")
                return FunctionResult(
                    success=False,
                    data=None,
                    error=f"Error executing {action}: {str(e)}",
                    execution_time=time.time() - start_time
                )
        else:
            return FunctionResult(
                success=False, 
                data=None, 
                error=f"Unknown browser action: {action}",
                execution_time=time.time() - start_time
            )
    
    def sculptor_action(self, action: str, **kwargs) -> FunctionResult:
        """Execute Sculptor Active Context Management actions."""
        start_time = time.time()
        self.tool_usage_stats['sculptor'] += 1
        
        logger.info(f"Executing Sculptor action: {action}")
        
        action_map = {
            "fragment_context": lambda: self.sculptor.fragment_context(**kwargs),
            "summary_fragment": lambda: self.sculptor.summary_fragment(**kwargs),
            "revert_summary": lambda: self.sculptor.revert_summary(**kwargs),
            "fold_fragment": lambda: self.sculptor.fold_fragment(**kwargs),
            "expand_fragment": lambda: self.sculptor.expand_fragment(**kwargs),
            "restore_context": lambda: self.sculptor.restore_context(**kwargs),
            "search_context": lambda: self.sculptor.search_context(**kwargs),
            "get_search_detail": lambda: self.sculptor.get_search_detail(**kwargs),
            "get_context_state": lambda: self.sculptor.get_context_state(**kwargs),
            "get_optimized_context": lambda: {"success": True, "context": self.sculptor.get_optimized_context()}
        }
        
        if action in action_map:
            try:
                result = action_map[action]()
                execution_time = time.time() - start_time
                
                return FunctionResult(
                    success=result.get("success", True),
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    metadata={
                        "sculptor_action": action, 
                        "result": result,
                        "execution_time": execution_time
                    },
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Error in Sculptor {action}: {e}")
                return FunctionResult(
                    success=False,
                    data=None,
                    error=f"Error executing Sculptor {action}: {str(e)}",
                    execution_time=execution_time
                )
        else:
            return FunctionResult(
                success=False, 
                data=None, 
                error=f"Unknown Sculptor action: {action}",
                execution_time=time.time() - start_time
            )
    
    def _handle_start(self, **kwargs):
        """Handle browser start action."""
        result = self.browser.start_session(**kwargs)
        if result.success:
            self.browser_started = True
        return result
    
    def _handle_close(self, **kwargs):
        """Handle browser close action."""
        result = self.browser.close_session(**kwargs)
        if result.success:
            self.browser_started = False
        return result
    
    def _configure_d2snap(self, **kwargs):
        """Configure D2Snap settings."""
        self.browser.configure_d2snap(**kwargs)
        return FunctionResult(
            success=True,
            data="D2Snap configuration updated",
            metadata=self.browser.d2snap_config
        )

    def finish(self, answer: str = None, **kwargs) -> FunctionResult:
        """Marks the task as complete with a final answer."""
        if not answer:
            return FunctionResult(
                success=False,
                data=None,
                error="No answer provided."
            )
        
        self.final_answer = answer
        logger.info(f"Task completed with X-Master (iterations: {self.iteration_count})")
        
        # Close browser if still open
        if self.browser_started and self.browser:
            try:
                self.browser.close_session()
                self.browser_started = False
            except Exception as e:
                logger.warning(f"Error closing browser on finish: {e}")
        
        return FunctionResult(
            success=True, 
            data="Task completed successfully.",
            metadata={
                'total_iterations': self.iteration_count,
                'tool_usage_stats': dict(self.tool_usage_stats),
                'execution_history_length': len(self.execution_history)
            }
        )

    def dispatch(self, function_call: FunctionCall) -> FunctionResult:
        """Dispatches function calls to appropriate executors with X-Master enhancements."""
        name = function_call.name
        args = function_call.arguments or {}
        
        self.iteration_count += 1
        function_call.iteration = self.iteration_count
        
        logger.info(f"X-Master dispatching function '{name}' (iteration {self.iteration_count})")

        # Enhanced dispatch with code execution support
        if name == "code":
            result = self.execute_code(
                args.get("code", ""),
                iteration=self.iteration_count,
                role=function_call.role
            )
        else:
            dispatch_map = {
                "web_search": lambda: self.web_search(**args),
                "web_parse": lambda: self.web_parse(**args),
                "browser": lambda: self.browser_action(
                    args.get("action", "navigate"),
                    **{k: v for k, v in args.items() if k != 'action'}
                ),
                "sculptor": lambda: self.sculptor_action(
                    args.get("action", "get_context_state"),
                    **{k: v for k, v in args.items() if k != 'action'}
                ),
                "finish": lambda: self.finish(**args)
            }

            if name in dispatch_map:
                result = dispatch_map[name]()
            else:
                logger.error(f"Unknown function: {name}")
                result = FunctionResult(
                    success=False, 
                    data=None, 
                    error=f"Unknown function: {name}",
                    iteration=self.iteration_count,
                    role=function_call.role
                )

        # Store execution history
        self.execution_history.append({
            'function_call': function_call,
            'result': result,
            'timestamp': datetime.now()
        })
        
        return result


# --- X-Master Universal Agent with Scattered-and-Stacked Workflow ---

class XMasterUniversalAgent:
    """
    X-Master Universal Agent with scattered-and-stacked workflow, D2Snap DOM optimization, 
    and Sculptor Active Context Management using GigaChat embeddings.
    
    Based on the research paper: "SciMaster: Towards General-Purpose Scientific AI Agents"
    """

    def __init__(self, name: str, client: GigaChatClient):
        self.name = name
        self.client = client
        self.function_executor = XMasterFunctionExecutor(client)
        self.max_iterations = 50  # X-Master supports long chains
        self.enable_d2snap = True
        self.enable_sculptor = True
        self.enable_scattered_stacked = True
        self.functions_state_id = None
        
        # X-Master workflow settings
        self.num_scattered_solutions = 5
        self.temperature_scattered = 0.6
        self.temperature_stacked = 0.3

    def get_initial_reasoning_guidance(self, role: AgentRole = AgentRole.MAIN) -> str:
        """
        Get initial reasoning guidance based on X-Master methodology.
        This helps guide non-agentic models towards agentic behavior.
        """
        base_guidance = """Я могу эффективно ответить на этот запрос, используя доступ к внешним средам и инструментам.
Каждый раз, когда мне нужно взаимодействовать с внешними инструментами, я буду генерировать Python код, заключенный между тегами <code> и </code>.
Я умею работать с любыми Python библиотеками и кастомными инструментами для решения сложных задач.
Мой подход к решению проблем включает итеративное взаимодействие между внутренним рассуждением и использованием внешних инструментов."""

        role_specific_guidance = {
            AgentRole.SOLVER: """Как Solver, я генерирую начальные решения, исследуя различные подходы к проблеме.
Я использую инструменты для поиска информации и проверки фактов, создавая разнообразные варианты решений.""",
            
            AgentRole.CRITIC: """Как Critic, я анализирую предложенные решения, выявляю потенциальные недостатки и предлагаю улучшения.
Я критически оцениваю логическую согласованность и фактическую точность решений.""",
            
            AgentRole.REWRITER: """Как Rewriter, я синтезирую множественные решения в улучшенный вариант.
Я объединяю лучшие элементы из различных подходов, устраняя противоречия и избыточность.""",
            
            AgentRole.SELECTOR: """Как Selector, я выбираю наилучшее решение из предложенных вариантов.
Я оцениваю решения на основе логической согласованности, фактической точности и полноты.""",
            
            AgentRole.MAIN: base_guidance
        }
        
        return f"{base_guidance}\n\n{role_specific_guidance.get(role, '')}"

    def get_available_functions(self) -> List[Dict]:
        """Returns enhanced functions with D2Snap, Sculptor, and X-Master support."""
        functions = [
            {
                "name": "web_search",
                "description": "Универсальный поиск информации в интернете с X-Master возможностями.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "web_parse",
                "description": "Парсинг и извлечение контента из веб-страниц с умной фильтрацией.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL для парсинга"
                        },
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос для фильтрации релевантного контента"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "code",
                "description": "Выполнение Python кода как языка взаимодействия с внешними средами (X-Master стиль).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python код для выполнения"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "browser",
                "description": "Управление браузером с D2Snap оптимизацией DOM для эффективной работы.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Действие браузера",
                            "enum": ["start", "navigate", "click", "extract", "structure", "dom_snapshot", "configure_d2snap", "close"]
                        },
                        "url": {
                            "type": "string", 
                            "description": "URL адрес сайта"
                        },
                        "identifier": {
                            "type": "string", 
                            "description": "Идентификатор элемента"
                        },
                        "target": {
                            "type": "string",
                            "description": "Цель для извлечения контента"
                        },
                        "use_d2snap": {
                            "type": "boolean",
                            "description": "Использовать D2Snap оптимизацию",
                            "default": True
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "sculptor",
                "description": "Sculptor Active Context Management - управление рабочей памятью через GigaChat эмбеддинги.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Действие Sculptor",
                            "enum": [
                                "fragment_context", "summary_fragment", "revert_summary",
                                "fold_fragment", "expand_fragment", "restore_context",
                                "search_context", "get_search_detail", "get_context_state", "get_optimized_context"
                            ]
                        },
                        "fragment_id": {
                            "type": "string",
                            "description": "ID фрагмента для операций"
                        },
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос для семантического поиска"
                        },
                        "mode": {
                            "type": "string",
                            "description": "Режим поиска",
                            "enum": ["exact", "semantic"],
                            "default": "semantic"
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "finish",
                "description": "Завершение задачи с подробным ответом.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Итоговый ответ"
                        }
                    },
                    "required": ["answer"]
                }
            }
        ]
        
        return functions

    def build_system_prompt(self, role: AgentRole = AgentRole.MAIN) -> str:
        """Builds enhanced system prompt with X-Master capabilities."""
        current_date = datetime.now().strftime("%Y-%m-%d")

        role_descriptions = {
            AgentRole.SOLVER: "🔍 X-Master SOLVER AGENT",
            AgentRole.CRITIC: "🎯 X-Master CRITIC AGENT", 
            AgentRole.REWRITER: "✍️ X-Master REWRITER AGENT",
            AgentRole.SELECTOR: "🏆 X-Master SELECTOR AGENT",
            AgentRole.MAIN: "🧠 X-Master UNIVERSAL AGENT"
        }

        return f"""{role_descriptions.get(role, "🧠 X-Master UNIVERSAL AGENT")} - Революционная архитектура tool-augmented reasoning с передовыми технологиями.
Дата: {current_date}

🎭 АРХИТЕКТУРА X-MASTER:

**CORE PRINCIPLES:**
✅ Code as Interaction Language - Python код как язык взаимодействия с внешними средами
✅ Tool-Augmented Reasoning - симбиоз внутреннего рассуждения и внешних инструментов  
✅ Scattered-and-Stacked Workflow - рассредоточенное исследование + последовательное улучшение
✅ Inference-Time Computation - масштабирование интеллекта во время вывода

**ТЕХНОЛОГИЧЕСКИЕ КОМПОНЕНТЫ:**

🔬 **D2SNAP DOM OPTIMIZATION:**
- Умное сжатие веб-страниц до 96% с сохранением UI-функций
- Адаптивные алгоритмы для оптимального размера токенов
- Классификация элементов и конвертация в Markdown
- Halton sequences для parameter exploration

🎭 **SCULPTOR ACTIVE CONTEXT MANAGEMENT:**
- GigaChat эмбеддинги (EmbeddingsGigaR) для семантического поиска
- Фрагментация длинных контекстов на управляемые части
- AI-резюмирование через GigaChat с возможностью восстановления
- Борьба с проактивной интерференцией и информационной перегрузкой

⚡ **X-MASTER TOOL INTEGRATION:**
- Доступ к любым Python библиотекам и кастомным инструментам
- Итеративное взаимодействие между рассуждением и действием
- Поддержка длинных цепочек инструментов (до 50 вызовов)
- Автоматическое управление рабочей памятью

🔥 **SCATTERED-AND-STACKED WORKFLOW:**

**SCATTERED PHASE (Рассредоточение):**
- Parallel exploration множественных решений
- Diverse solution generation через temperature variation
- Broad problem-solving across multiple reasoning paths

**STACKED PHASE (Стекирование):**
- Sequential refinement и iterative improvement
- Solution synthesis из лучших элементов
- Final selection на основе качества и согласованности

🧠 **РОЛИ В WORKFLOW:**

**SOLVER:** Генерирую начальные решения, исследуя различные подходы к проблеме.
**CRITIC:** Анализирую решения, выявляю недостатки и предлагаю улучшения.
**REWRITER:** Синтезирую множественные решения в улучшенный вариант.
**SELECTOR:** Выбираю наилучшее решение из предложенных вариантов.

🚀 **АЛГОРИТМ РАБОТЫ:**

1. **Tool-Augmented Reasoning:**
   - Использую <code>Python код</code> для взаимодействия с внешними средами
   - Результаты выполнения интегрируются в процесс рассуждения
   - Итеративное взаимодействие до получения решения

2. **Dynamic Problem-Solving:**
   - Активно ищу и использую информацию как человек-исследователь
   - Адаптирую стратегии на основе промежуточных результатов
   - Масштабирую решения через inference-time computation

3. **Context Management:**
   - Применяю Sculptor для управления рабочей памятью
   - Использую семантический поиск для релевантной информации
   - Предотвращаю информационную перегрузку через активное управление контекстом

📊 **PERFORMANCE CHARACTERISTICS:**
- Достигнут SOTA на Humanity's Last Exam (32.1%)
- Превосходит GPT-4 и Gemini на сложных научных задачах
- Поддержка цепочек до 50+ вызовов инструментов
- Эффективная работа с большими объемами информации

Я - продвинутый AI-агент, способный к сложному рассуждению и использованию инструментов для решения любых задач!"""

    def process_single_solution(self, query: str, role: AgentRole = AgentRole.SOLVER) -> Dict[str, Any]:
        """Process a single solution with specific role."""
        logger.info(f"Processing solution with role: {role.value}")
        
        # Build role-specific prompt
        initial_guidance = self.get_initial_reasoning_guidance(role)
        
        messages = [
            {"role": "system", "content": self.build_system_prompt(role)},
            {"role": "assistant", "content": f"<think>\n{initial_guidance}\n\nТеперь приступлю к решению задачи:\n{query}\n</think>"},
            {"role": "user", "content": query}
        ]

        thinking_process = []
        function_calls = []
        function_results = []
        final_answer_submitted = False
        iteration_count = 0

        for iteration in range(self.max_iterations):
            iteration_count += 1
            
            try:
                logger.info(f"Role {role.value} - Iteration {iteration + 1}/{self.max_iterations}")
                
                temperature = self.temperature_scattered if role == AgentRole.SOLVER else self.temperature_stacked
                
                response = self.client.chat(
                    messages,
                    functions=self.get_available_functions(),
                    temperature=temperature
                )

                if not response or 'choices' not in response or not response['choices']:
                    break

                choice = response['choices'][0]
                message = choice['message']
                
                # Add message to conversation
                msg_to_add = {
                    "role": message['role'],
                    "content": message.get('content', '')
                }
                if 'function_call' in message:
                    msg_to_add['function_call'] = message['function_call']
                
                messages.append(msg_to_add)
                thinking_process.append(msg_to_add)

                # Handle function calls
                if 'function_call' in message:
                    fc_data = message['function_call']
                    fc_name = fc_data.get('name')
                    raw_args = fc_data.get('arguments', {})
                    
                    # Parse arguments
                    arguments = {}
                    if isinstance(raw_args, dict):
                        arguments = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except json.JSONDecodeError:
                            arguments = {}

                    # Execute function
                    function_call = FunctionCall(
                        name=fc_name, 
                        arguments=arguments,
                        iteration=iteration_count,
                        role=role
                    )
                    function_calls.append(function_call)
                    
                    result = self.function_executor.dispatch(function_call)
                    function_results.append(result)

                    if function_call.name == "finish":
                        final_answer_submitted = True

                    # Add function result to messages
                    content_payload = {
                        "success": result.success,
                        "data": result.data if not isinstance(result.data, bytes) else f"Binary data ({len(result.data)} bytes)",
                        "metadata": result.metadata if result.metadata else {}
                    }

                    function_msg = {
                        "role": "function",
                        "name": function_call.name,
                        "content": json.dumps(content_payload, ensure_ascii=False, indent=2)
                    }
                    
                    messages.append(function_msg)
                    thinking_process.append(function_msg)

                    if final_answer_submitted and self.function_executor.final_answer:
                        break

            except Exception as e:
                logger.error(f"Error in role {role.value} iteration {iteration}: {e}")
                break

        solution = self.function_executor.final_answer if self.function_executor.final_answer else ""
        
        return {
            'role': role,
            'solution': solution,
            'thinking_process': thinking_process,
            'function_calls': function_calls,
            'function_results': function_results,
            'iteration_count': iteration_count,
            'success': final_answer_submitted
        }

    def scattered_phase(self, query: str) -> List[Dict[str, Any]]:
        """Scattered phase: generate multiple diverse solutions."""
        logger.info("🔀 Starting SCATTERED phase - generating diverse solutions")
        
        solutions = []
        
        # Generate multiple solutions with Solver role
        for i in range(self.num_scattered_solutions):
            logger.info(f"Generating solution {i+1}/{self.num_scattered_solutions}")
            
            # Reset executor for each solution
            self.function_executor.final_answer = None
            self.function_executor.iteration_count = 0
            
            solution_data = self.process_single_solution(query, AgentRole.SOLVER)
            solutions.append(solution_data)
            
            # Brief pause between solutions
            time.sleep(0.5)
        
        # Apply Critic to each solution
        logger.info("🎯 Applying CRITIC to refine solutions")
        
        for i, solution in enumerate(solutions):
            if solution['success'] and solution['solution']:
                critic_query = f"""Проанализируй и улучши следующее решение задачи:

ЗАДАЧА: {query}

РЕШЕНИЕ: {solution['solution']}

Выяви потенциальные недостатки, проверь фактическую точность и предложи улучшения."""
                
                self.function_executor.final_answer = None
                critic_result = self.process_single_solution(critic_query, AgentRole.CRITIC)
                
                if critic_result['success'] and critic_result['solution']:
                    solution['critic_feedback'] = critic_result['solution']
                    solution['improved'] = True
        
        return solutions

    def stacked_phase(self, query: str, scattered_solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stacked phase: synthesize and select best solution."""
        logger.info("📚 Starting STACKED phase - synthesis and selection")
        
        # Extract successful solutions
        valid_solutions = [s for s in scattered_solutions if s['success'] and s['solution']]
        
        if not valid_solutions:
            return {
                'success': False,
                'final_solution': "No valid solutions generated in scattered phase",
                'confidence': 0.0
            }
        
        # Prepare solutions for rewriting
        solutions_text = []
        for i, solution in enumerate(valid_solutions):
            solution_text = f"РЕШЕНИЕ {i+1}:\n{solution['solution']}"
            if solution.get('critic_feedback'):
                solution_text += f"\nКРИТИКА: {solution['critic_feedback']}"
            solutions_text.append(solution_text)
        
        # Rewriter phase: synthesize solutions
        logger.info("✍️ REWRITER phase - synthesizing solutions")
        
        rewriter_query = f"""Синтезируй лучшее решение на основе всех предложенных вариантов:

ИСХОДНАЯ ЗАДАЧА: {query}

ДОСТУПНЫЕ РЕШЕНИЯ:
{chr(10).join(solutions_text)}

Создай улучшенное решение, объединив лучшие элементы, устранив противоречия и недостатки."""

        self.function_executor.final_answer = None
        rewritten_solutions = []
        
        # Generate multiple rewritten versions
        for i in range(min(3, len(valid_solutions))):
            rewrite_result = self.process_single_solution(rewriter_query, AgentRole.REWRITER)
            if rewrite_result['success'] and rewrite_result['solution']:
                rewritten_solutions.append(rewrite_result)
        
        # Selector phase: choose best solution
        logger.info("🏆 SELECTOR phase - choosing best solution")
        
        all_candidate_solutions = rewritten_solutions if rewritten_solutions else valid_solutions
        
        if len(all_candidate_solutions) == 1:
            return {
                'success': True,
                'final_solution': all_candidate_solutions[0]['solution'],
                'confidence': 0.8,
                'selection_process': 'Only one candidate available'
            }
        
        # Prepare candidates for selection
        candidates_text = []
        for i, candidate in enumerate(all_candidate_solutions):
            candidates_text.append(f"КАНДИДАТ {i+1}:\n{candidate['solution']}")
        
        selector_query = f"""Выбери наилучшее решение из предложенных кандидатов:

ИСХОДНАЯ ЗАДАЧА: {query}

КАНДИДАТЫ:
{chr(10).join(candidates_text)}

Оцени каждого кандидата по критериям: логическая согласованность, фактическая точность, полнота ответа.
Выбери лучшего и предоставь итоговое решение."""

        self.function_executor.final_answer = None
        selection_result = self.process_single_solution(selector_query, AgentRole.SELECTOR)
        
        if selection_result['success'] and selection_result['solution']:
            return {
                'success': True,
                'final_solution': selection_result['solution'],
                'confidence': 0.95,
                'selection_process': 'Full scattered-and-stacked workflow completed',
                'candidates_count': len(all_candidate_solutions)
            }
        else:
            # Fallback to best rewritten solution
            best_candidate = rewritten_solutions[0] if rewritten_solutions else valid_solutions[0]
            return {
                'success': True,
                'final_solution': best_candidate['solution'],
                'confidence': 0.7,
                'selection_process': 'Fallback to best candidate'
            }

    def process(self, query: str) -> AgentResponse:
        """Process query with full X-Master scattered-and-stacked workflow."""
        logger.info("🚀 Starting X-Master processing with scattered-and-stacked workflow")
        
        # Initialize Sculptor with current query
        self.function_executor.sculptor.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        workflow_stages = []
        all_function_calls = []
        all_function_results = []
        total_iterations = 0
        
        try:
            # Check if scattered-and-stacked workflow is enabled
            if self.enable_scattered_stacked:
                # SCATTERED PHASE
                scattered_start = time.time()
                scattered_solutions = self.scattered_phase(query)
                scattered_time = time.time() - scattered_start
                
                workflow_stages.append(WorkflowStage(
                    stage_name="scattered",
                    role=AgentRole.SOLVER,
                    input_data=query,
                    output_data=scattered_solutions,
                    success=bool(scattered_solutions),
                    iteration_count=sum(s.get('iteration_count', 0) for s in scattered_solutions)
                ))
                
                # Collect all function calls and results from scattered phase
                for solution in scattered_solutions:
                    all_function_calls.extend(solution.get('function_calls', []))
                    all_function_results.extend(solution.get('function_results', []))
                    total_iterations += solution.get('iteration_count', 0)
                
                # STACKED PHASE
                stacked_start = time.time()
                stacked_result = self.stacked_phase(query, scattered_solutions)
                stacked_time = time.time() - stacked_start
                
                workflow_stages.append(WorkflowStage(
                    stage_name="stacked",
                    role=AgentRole.REWRITER,
                    input_data=scattered_solutions,
                    output_data=stacked_result,
                    success=stacked_result.get('success', False)
                ))
                
                final_answer = stacked_result.get('final_solution', '')
                confidence = stacked_result.get('confidence', 0.8)
                
                logger.info(f"✅ X-Master workflow completed - Scattered: {scattered_time:.2f}s, Stacked: {stacked_time:.2f}s")
                
            else:
                # Single agent processing (fallback)
                logger.info("📋 Using single agent processing")
                
                single_result = self.process_single_solution(query, AgentRole.MAIN)
                
                workflow_stages.append(WorkflowStage(
                    stage_name="single",
                    role=AgentRole.MAIN,
                    input_data=query,
                    output_data=single_result,
                    success=single_result.get('success', False),
                    iteration_count=single_result.get('iteration_count', 0)
                ))
                
                all_function_calls.extend(single_result.get('function_calls', []))
                all_function_results.extend(single_result.get('function_results', []))
                total_iterations += single_result.get('iteration_count', 0)
                
                final_answer = single_result.get('solution', '')
                confidence = 0.7
                scattered_solutions = []

        except Exception as e:
            logger.error(f"X-Master processing error: {e}", exc_info=True)
            final_answer = f"Произошла ошибка при обработке: {str(e)}"
            confidence = 0.0
            scattered_solutions = []

        # Ensure browser is closed
        try:
            if self.function_executor.browser_started and self.function_executor.browser:
                self.function_executor.browser.close_session()
        except:
            pass

        return AgentResponse(
            agent_name=self.name,
            role="x_master_universal_agent",
            content=final_answer,
            thinking_process=[],  # Could aggregate from all stages if needed
            function_calls=all_function_calls,
            function_results=all_function_results,
            final_answer=final_answer,
            confidence=confidence,
            functions_state_id=self.functions_state_id,
            sculptor_state=self.function_executor.sculptor.get_context_state(),
            workflow_stages=workflow_stages,
            total_iterations=total_iterations,
            scattered_solutions=[s.get('solution', '') for s in scattered_solutions],
            selected_solution=final_answer
        )


# --- Enhanced Streamlit UI ---

def main():
    """Main Streamlit application with X-Master enhanced agent."""
    st.set_page_config(
        page_title="🎭 X-Master Universal Agent",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🎭 X-Master Universal Agent")
    st.markdown("**Революционная архитектура с scattered-and-stacked workflow, D2Snap DOM-оптимизацией и Sculptor ACM через GigaChat**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ X-Master Конфигурация")

        client_id = st.text_input("GigaChat Client ID", type="password")
        client_secret = st.text_input("GigaChat Client Secret", type="password")

        st.markdown("---")

        st.subheader("🧠 Настройки модели")
        model_name = st.selectbox(
            "Модель GigaChat",
            ["GigaChat-2-Max", "GigaChat", "GigaChat-2-Pro"],
            index=0
        )
        
        st.markdown("---")
        
        st.subheader("🎭 X-Master Workflow")
        
        enable_scattered_stacked = st.checkbox(
            "Включить Scattered-and-Stacked Workflow", 
            value=True,
            help="Полная архитектура X-Master с рассредоточенным исследованием и последовательным улучшением"
        )
        
        if enable_scattered_stacked:
            num_scattered_solutions = st.slider(
                "Количество рассредоточенных решений",
                min_value=2,
                max_value=10,
                value=5,
                help="Количество параллельных решений в scattered фазе"
            )
            
            temperature_scattered = st.slider(
                "Температура Scattered",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Температура для генерации разнообразных решений"
            )
            
            temperature_stacked = st.slider(
                "Температура Stacked",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Температура для синтеза и выбора решений"
            )
        
        max_iterations = st.number_input(
            "Максимум итераций",
            min_value=10,
            max_value=50,
            value=50,
            help="Поддержка длинных цепочек инструментов"
        )
        
        st.markdown("---")
        
        st.subheader("🔬 Дополнительные технологии")
        
        enable_d2snap = st.checkbox(
            "D2Snap DOM оптимизация", 
            value=True,
            help="Умное сжатие веб-страниц с сохранением UI-функций"
        )
        
        enable_sculptor = st.checkbox(
            "Sculptor ACM",
            value=True,
            help="Active Context Management через GigaChat эмбеддинги"
        )
        
        st.markdown("---")
        
        st.subheader("📊 X-Master Архитектура")
        st.success("✅ Code as Interaction Language")
        st.success("✅ Tool-Augmented Reasoning")
        st.success("✅ Scattered-and-Stacked Workflow")
        st.success("✅ Inference-Time Computation")
        st.success("✅ D2Snap DOM Optimization")
        st.success("✅ Sculptor ACM with GigaChat")
        st.success("✅ 50+ Tool Chains Support")
        
        st.markdown("---")
        
        st.info(
            "**X-Master Features:**\n"
            "• SOTA на Humanity's Last Exam (32.1%)\n"
            "• Превосходит OpenAI и Google\n"
            "• Поддержка цепочек до 50 инструментов\n"
            "• Scattered-and-Stacked архитектура\n"
            "• Python код как язык взаимодействия\n"
            "• Полная интеграция с GigaChat API\n\n"
            "Основано на исследовании:\n"
            "'SciMaster: Towards General-Purpose Scientific AI Agents'"
        )

    st.markdown("---")

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 X-Master Query Interface")
        query = st.text_area(
            "Сформулируйте ваш запрос для X-Master агента:",
            height=150,
            placeholder="Примеры X-Master задач:\n• Исследуй последние достижения в области квантовых вычислений\n• Найди и проанализируй данные о климатических изменениях за последний год\n• Создай подробный анализ рынка ИИ с использованием множественных источников\n• Реши сложную научную задачу с пошаговым обоснованием\n• Проведи comparative анализ различных ML фреймворков",
            help="X-Master может решать сложные задачи, требующие множественных инструментов и длинных цепочек рассуждений"
        )
    
    with col2:
        st.markdown("### 🎭 X-Master Capabilities")
        st.markdown("""
        **Scattered-and-Stacked Workflow:**
        - 🔀 Scattered: Параллельная генерация решений
        - 📚 Stacked: Синтез и выбор лучшего
        
        **Tool-Augmented Reasoning:**
        - 🐍 Python код как interaction language
        - 🔧 Доступ к любым библиотекам и инструментам
        - 🔄 Итеративное взаимодействие
        
        **Advanced Technologies:**
        - 🔬 D2Snap: DOM оптимизация до 96%
        - 🎭 Sculptor: ACM через GigaChat эмбеддинги
        - ⚡ Поддержка цепочек до 50 инструментов
        
        **Performance:**
        - 🏆 SOTA на научных бенчмарках
        - 🧠 Human-like problem solving
        - 📈 Масштабируемый интеллект
        """)

    st.markdown("---")

    # Execution button
    execute_button = st.button(
        "🚀 Запустить X-Master Agent", 
        type="primary", 
        use_container_width=True,
        disabled=not (client_id and client_secret and query)
    )

    if execute_button:
        try:
            # Initialize X-Master agent
            client = GigaChatClient(client_id, client_secret, model=model_name)
            agent = XMasterUniversalAgent("X-Master-Universal", client)
            
            # Configure agent settings
            agent.max_iterations = max_iterations
            agent.enable_d2snap = enable_d2snap
            agent.enable_sculptor = enable_sculptor
            agent.enable_scattered_stacked = enable_scattered_stacked
            
            if enable_scattered_stacked:
                agent.num_scattered_solutions = num_scattered_solutions
                agent.temperature_scattered = temperature_scattered
                agent.temperature_stacked = temperature_stacked
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🎭 X-Master Agent решает задачу с scattered-and-stacked workflow..."):
                status_text.text("🔗 Инициализация X-Master архитектуры...")
                progress_bar.progress(10)
                
                if enable_scattered_stacked:
                    status_text.text("🔀 Scattered phase - генерация множественных решений...")
                    progress_bar.progress(30)
                    
                    status_text.text("🎯 Critic phase - анализ и улучшение решений...")
                    progress_bar.progress(50)
                    
                    status_text.text("✍️ Rewriter phase - синтез лучших элементов...")
                    progress_bar.progress(70)
                    
                    status_text.text("🏆 Selector phase - выбор оптимального решения...")
                    progress_bar.progress(90)
                else:
                    status_text.text("📋 Single agent processing...")
                    progress_bar.progress(50)
                
                # Process query
                result = agent.process(query)
                
                progress_bar.progress(100)
                status_text.text("✅ X-Master workflow завершен!")

            st.markdown("---")
            
            # Results section
            st.subheader("🎯 X-Master Результат")
            
            if result.final_answer:
                st.success(result.final_answer)
            elif result.content:
                st.info(result.content)
            elif result.error:
                st.error(f"Произошла ошибка: {result.error}")
            else:
                st.warning("X-Master не смог сформировать ответ")

            # X-Master workflow metrics
            if result.workflow_stages:
                st.markdown("---")
                st.subheader("🎭 X-Master Workflow Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Workflow Stages",
                        len(result.workflow_stages),
                        f"Iterations: {result.total_iterations}"
                    )
                
                with col2:
                    scattered_count = len(result.scattered_solutions)
                    st.metric(
                        "Scattered Solutions",
                        scattered_count,
                        f"Diversity: {scattered_count}x"
                    )
                
                with col3:
                    tool_calls = len(result.function_calls)
                    st.metric(
                        "Tool Calls",
                        tool_calls,
                        f"Chain length: {tool_calls}"
                    )
                
                with col4:
                    st.metric(
                        "Confidence",
                        f"{result.confidence:.2f}",
                        f"Quality: {'High' if result.confidence > 0.8 else 'Medium'}"
                    )

            # Workflow stages breakdown
            if result.workflow_stages:
                st.markdown("---")
                st.subheader("🔄 Workflow Stages Breakdown")
                
                for stage in result.workflow_stages:
                    with st.expander(f"📋 Stage: {stage.stage_name.title()} ({stage.role.value})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Stage Info:**")
                            st.json({
                                "Role": stage.role.value,
                                "Success": stage.success,
                                "Iterations": stage.iteration_count,
                                "Error": stage.error
                            })
                        
                        with col2:
                            if stage.stage_name == "scattered" and isinstance(stage.output_data, list):
                                st.markdown("**Generated Solutions:**")
                                for i, solution in enumerate(stage.output_data):
                                    if solution.get('solution'):
                                        st.text(f"Solution {i+1}: {solution['solution'][:100]}...")

            # Scattered solutions display
            if result.scattered_solutions:
                st.markdown("---")
                st.subheader("🔀 Scattered Solutions Analysis")
                
                for i, solution in enumerate(result.scattered_solutions):
                    if solution.strip():
                        with st.expander(f"💡 Solution {i+1}"):
                            st.write(solution)

            # Technology metrics
            if result.function_results:
                st.markdown("---")
                st.subheader("⚡ Technology Integration Metrics")
                
                # Analyze technology usage
                d2snap_usage = sum(1 for fr in result.function_results if 'compression_ratio' in (fr.metadata or {}))
                sculptor_usage = sum(1 for fr in result.function_results if 'sculptor_action' in (fr.metadata or {}))
                code_executions = sum(1 for fc in result.function_calls if fc.name == 'code')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "D2Snap Operations",
                        d2snap_usage,
                        "DOM Optimizations"
                    )
                
                with col2:
                    st.metric(
                        "Sculptor ACM",
                        sculptor_usage,
                        "Context Management"
                    )
                
                with col3:
                    st.metric(
                        "Code Executions",
                        code_executions,
                        "Tool Interactions"
                    )

            # Detailed execution log
            with st.expander("📋 Detailed X-Master Execution Log", expanded=False):
                for i, (call, res) in enumerate(zip(result.function_calls, result.function_results)):
                    st.markdown(f"### 🔧 Operation {i + 1}: `{call.name}` (Iteration {call.iteration})")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Parameters:**")
                        st.json(call.arguments)
                        st.markdown(f"**Role:** {call.role.value}")
                    
                    with col2:
                        st.markdown("**Result:**")
                        if res.success:
                            st.success(f"✅ Success (⏱️ {res.execution_time:.2f}s)")
                            if res.metadata:
                                # Show relevant metadata
                                display_metadata = {}
                                for key, value in res.metadata.items():
                                    if key in ['compression_ratio', 'sculptor_action', 'execution_time', 'tool', 'query']:
                                        display_metadata[key] = value
                                if display_metadata:
                                    st.json(display_metadata)
                        else:
                            st.error(f"❌ Error: {res.error}")

        except Exception as e:
            st.error(f"❌ Произошла ошибка: {str(e)}")
            logger.error("Error in X-Master execution", exc_info=True)


if __name__ == "__main__":
    main()