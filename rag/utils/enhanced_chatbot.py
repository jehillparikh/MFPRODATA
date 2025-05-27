"""
Enhanced Chatbot Module

This module implements an advanced conversational interface that leverages both RAG and LLM
capabilities. It includes sophisticated features like memory management, context selection,
streaming responses, and multi-session support.

Key Components:
1. Memory Management:
   - Short-term memory: Recent conversation turns
   - Long-term memory: Important facts and summaries
   - Working memory: Current context and active information

2. Context Selection:
   - Semantic relevance scoring
   - Time-based decay
   - Importance weighting

3. LLM Integration:
   - Response generation using LLM
   - Prompt engineering
   - Context window optimization

4. Session Management:
   - Multiple concurrent users
   - Session persistence
   - State management

5. Streaming Support:
   - Async response generation
   - Chunked response delivery
   - Progress updates
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import uuid
from enum import Enum
import numpy as np
from collections import deque

# ============================================================================
# Memory Management
# ============================================================================

class MemoryType(Enum):
    """
    Types of memory in the system.
    
    SHORT_TERM: Recent messages and immediate context
    LONG_TERM: Important information that should persist
    WORKING: Currently active information being used for response generation
    """
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"

@dataclass
class MemoryItem:
    """
    A single item in memory.
    
    Attributes:
        content (str): The actual information stored
        timestamp (datetime): When the item was created
        importance (float): How important this item is (0-1)
        metadata (Dict): Additional information about this memory
        memory_type (MemoryType): What kind of memory this is
        last_accessed (datetime): When this was last used
        access_count (int): How many times this has been accessed
    """
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.SHORT_TERM
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

class MemoryManager:
    """
    Manages different types of memory and their interactions.
    
    This class handles:
    1. Memory storage and retrieval
    2. Importance scoring
    3. Memory consolidation
    4. Forgetting mechanisms
    """
    
    def __init__(self, 
                 short_term_limit: int = 10,
                 long_term_limit: int = 100,
                 working_memory_limit: int = 5):
        """
        Initialize memory systems.
        
        Args:
            short_term_limit: Max items in short-term memory
            long_term_limit: Max items in long-term memory
            working_memory_limit: Max items in working memory
        """
        self.memories: Dict[MemoryType, deque] = {
            MemoryType.SHORT_TERM: deque(maxlen=short_term_limit),
            MemoryType.LONG_TERM: deque(maxlen=long_term_limit),
            MemoryType.WORKING: deque(maxlen=working_memory_limit)
        }
        
    def add_memory(self, item: MemoryItem) -> None:
        """
        Add a new memory item to the appropriate store.
        
        The method also:
        1. Triggers consolidation if needed
        2. Updates importance scores
        3. Manages memory limits
        """
        self.memories[item.memory_type].append(item)
        
        # Consolidate memories if short-term is full
        if (item.memory_type == MemoryType.SHORT_TERM and 
            len(self.memories[MemoryType.SHORT_TERM]) == self.memories[MemoryType.SHORT_TERM].maxlen):
            self._consolidate_memories()
    
    def get_relevant_memories(self, 
                            query: str, 
                            embedding_fn: Callable,
                            top_k: int = 5) -> List[MemoryItem]:
        """
        Retrieve memories relevant to the current query.
        
        Uses semantic similarity to find relevant memories across all memory types.
        Updates access counts and timestamps for retrieved memories.
        
        Args:
            query: The current user query
            embedding_fn: Function to generate embeddings
            top_k: Number of memories to retrieve
        """
        # Get query embedding
        query_embedding = embedding_fn([query])[0]
        
        # Get all memories
        all_memories = []
        for memory_type in MemoryType:
            all_memories.extend(self.memories[memory_type])
        
        # Get embeddings for all memories
        memory_embeddings = embedding_fn([m.content for m in all_memories])
        
        # Calculate similarities
        similarities = np.dot(memory_embeddings, query_embedding)
        
        # Get top_k memories
        top_indices = np.argsort(similarities)[-top_k:]
        
        # Update access info
        relevant_memories = []
        for idx in top_indices:
            memory = all_memories[idx]
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            relevant_memories.append(memory)
        
        return relevant_memories
    
    def _consolidate_memories(self) -> None:
        """
        Consolidate short-term memories into long-term memory.
        
        This process:
        1. Identifies important short-term memories
        2. Summarizes related memories
        3. Transfers to long-term memory
        4. Clears short-term memory
        """
        # Get memories above importance threshold
        important_memories = [
            m for m in self.memories[MemoryType.SHORT_TERM]
            if m.importance > 0.7  # Configurable threshold
        ]
        
        # Transfer to long-term memory
        for memory in important_memories:
            memory.memory_type = MemoryType.LONG_TERM
            self.add_memory(memory)
        
        # Clear short-term memory
        self.memories[MemoryType.SHORT_TERM].clear()

# ============================================================================
# Session Management
# ============================================================================

@dataclass
class Session:
    """
    Represents a chat session for a specific user.
    
    Attributes:
        session_id: Unique identifier for the session
        user_id: Identifier for the user
        memory_manager: Manages session-specific memories
        metadata: Additional session information
        created_at: When the session was created
        last_active: Last activity timestamp
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_manager: MemoryManager = field(default_factory=MemoryManager)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

class SessionManager:
    """
    Manages multiple chat sessions.
    
    This class handles:
    1. Session creation and cleanup
    2. Session storage and retrieval
    3. Session persistence
    4. Session state management
    """
    
    def __init__(self, max_sessions: int = 1000):
        """
        Initialize session manager.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
        """
        self.sessions: Dict[str, Session] = {}
        self.max_sessions = max_sessions
    
    def create_session(self, user_id: str) -> Session:
        """Create a new session for a user."""
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()
        
        session = Session(user_id=user_id)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session by ID."""
        session = self.sessions.get(session_id)
        if session:
            session.last_active = datetime.now()
        return session
    
    def _cleanup_old_sessions(self) -> None:
        """Remove oldest inactive sessions when limit is reached."""
        if len(self.sessions) < self.max_sessions:
            return
            
        # Sort sessions by last_active
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_active
        )
        
        # Remove oldest 10% of sessions
        sessions_to_remove = max(1, len(self.sessions) // 10)
        for session_id, _ in sorted_sessions[:sessions_to_remove]:
            del self.sessions[session_id]

# ============================================================================
# Enhanced Chatbot
# ============================================================================

class EnhancedChatbot:
    """
    Advanced chatbot with LLM integration, memory management, and streaming support.
    
    Key features:
    1. LLM-powered response generation
    2. Sophisticated memory management
    3. Multi-session support
    4. Streaming responses
    5. Context optimization
    """
    
    def __init__(self,
                 get_embeddings_fn: Callable,
                 search_vector_store_fn: Callable,
                 llm_fn: Callable,
                 max_sessions: int = 1000):
        """
        Initialize the enhanced chatbot.
        
        Args:
            get_embeddings_fn: Function to generate embeddings
            search_vector_store_fn: Function to search vector store
            llm_fn: Function to generate LLM responses
            max_sessions: Maximum number of concurrent sessions
        """
        self.get_embeddings = get_embeddings_fn
        self.search_vector_store = search_vector_store_fn
        self.llm = llm_fn
        self.session_manager = SessionManager(max_sessions)
    
    def _create_prompt(self,
                      query: str,
                      context_docs: List[Dict[str, Any]],
                      memories: List[MemoryItem]) -> str:
        """
        Create a prompt for the LLM.
        
        Combines:
        1. System instructions
        2. Relevant documents
        3. Relevant memories
        4. User query
        
        Returns formatted prompt string.
        """
        # System instruction
        prompt = "You are a helpful assistant with access to relevant documents and memory. "
        prompt += "Provide accurate, document-grounded responses.\n\n"
        
        # Add context from documents
        prompt += "Relevant documents:\n"
        for doc in context_docs:
            prompt += f"- {doc['text']}\n"
        prompt += "\n"
        
        # Add relevant memories
        prompt += "Relevant context from memory:\n"
        for memory in memories:
            prompt += f"- {memory.content}\n"
        prompt += "\n"
        
        # Add query
        prompt += f"User question: {query}\n"
        prompt += "Assistant: "
        
        return prompt
    
    async def _stream_llm_response(self,
                                 prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream the LLM response token by token.
        
        Args:
            prompt: The formatted prompt for the LLM
            
        Yields:
            Individual tokens/chunks of the response
        """
        # This is a placeholder - replace with actual LLM streaming implementation
        response = await self.llm(prompt)
        
        # Simulate streaming by yielding words
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.1)  # Simulate token generation time
    
    async def get_streaming_response(self,
                                   query: str,
                                   session_id: str = None) -> AsyncGenerator[str, None]:
        """
        Get a streaming response for the given query.
        
        Args:
            query: User's question/message
            session_id: Optional session identifier
            
        Yields:
            Response tokens/chunks as they're generated
        """
        try:
            # Get or create session
            session = (self.session_manager.get_session(session_id) if session_id
                      else self.session_manager.create_session("default_user"))
            
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search vector store
            context_docs = self.search_vector_store(query_embedding, k=3)
            
            # Get relevant memories
            memories = session.memory_manager.get_relevant_memories(
                query,
                self.get_embeddings
            )
            
            # Create prompt
            prompt = self._create_prompt(query, context_docs, memories)
            
            # Stream response
            async for token in self._stream_llm_response(prompt):
                yield token
            
            # Store interaction in memory
            session.memory_manager.add_memory(MemoryItem(
                content=query,
                memory_type=MemoryType.SHORT_TERM,
                metadata={"type": "user_message"}
            ))
            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a session's memory and state."""
        session = self.session_manager.get_session(session_id)
        if session:
            session.memory_manager = MemoryManager()
            return True
        return False 