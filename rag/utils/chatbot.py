"""
Chatbot Module

This module implements a conversational interface that leverages the RAG system
for knowledge-based responses. It maintains conversation history and generates
contextual responses based on both the conversation history and retrieved documents.

Features:
    - Conversation history management
    - Context-aware responses
    - Document-grounded answers
    - Configurable response generation
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Message:
    """
    Represents a single message in the conversation.
    
    Attributes:
        role (str): The role of the message sender ('user' or 'assistant')
        content (str): The content of the message
        timestamp (datetime): When the message was created
        metadata (Dict[str, Any]): Additional message metadata
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conversation:
    """
    Manages a conversation session.
    
    Attributes:
        messages (List[Message]): List of messages in the conversation
        max_history (int): Maximum number of messages to keep in history
    """
    messages: List[Message] = field(default_factory=list)
    max_history: int = 10

    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add a new message to the conversation."""
        self.messages.append(Message(role, content, datetime.now(), metadata or {}))
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history in a format suitable for context."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in self.messages
        ]

class Chatbot:
    """
    A RAG-powered chatbot that provides document-grounded responses.
    
    This class integrates with the RAG system to provide responses based on
    both the conversation context and relevant documents.
    """

    def __init__(self, 
                 get_embeddings_fn,
                 search_vector_store_fn,
                 max_history: int = 10,
                 max_context_docs: int = 3,
                 response_template: str = None):
        """
        Initialize the chatbot.
        
        Args:
            get_embeddings_fn: Function to generate embeddings
            search_vector_store_fn: Function to search the vector store
            max_history (int): Maximum number of messages to keep in history
            max_context_docs (int): Maximum number of documents to use for context
            response_template (str, optional): Template for response generation
        """
        self.get_embeddings = get_embeddings_fn
        self.search_vector_store = search_vector_store_fn
        self.max_context_docs = max_context_docs
        self.response_template = response_template or self._default_template()
        self.conversation = Conversation(max_history=max_history)

    def _default_template(self) -> str:
        """Default response template."""
        return """Based on the conversation history and available documents, here's what I found:

{context}

{response}

Sources:
{sources}"""

    def _format_response(self, 
                        response: str, 
                        context_docs: List[Dict[str, Any]]) -> str:
        """Format the response with context and sources."""
        # Extract relevant context
        context = "\n".join(f"- {doc['text']}" for doc in context_docs)
        
        # Format sources
        sources = "\n".join(
            f"- {doc.get('source', 'Unknown source')} (Score: {doc.get('score', 0):.2f})"
            for doc in context_docs
        )
        
        return self.response_template.format(
            context=context,
            response=response,
            sources=sources
        )

    def _generate_response(self, 
                         query: str, 
                         context_docs: List[Dict[str, Any]],
                         conversation_history: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the query, context docs, and conversation history.
        
        This is a simple implementation that can be enhanced with an LLM for
        more sophisticated response generation.
        """
        # For now, we'll just use a simple template-based response
        # This can be replaced with an LLM-based response generation
        if not context_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Use the most relevant document's text as the response
        response = f"Here's what I found: {context_docs[0]['text']}"
        
        return self._format_response(response, context_docs)

    async def get_response(self, query: str) -> str:
        """
        Get a response for the given query.
        
        Args:
            query (str): The user's question or message
            
        Returns:
            str: The chatbot's response
            
        Example:
            ```python
            response = await chatbot.get_response("What are the top holdings?")
            print(response)
            ```
        """
        try:
            # Add user message to conversation
            self.conversation.add_message("user", query)
            
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search vector store
            context_docs = self.search_vector_store(
                query_embedding, 
                k=self.max_context_docs
            )
            
            # Generate response
            response = self._generate_response(
                query,
                context_docs,
                self.conversation.get_history()
            )
            
            # Add response to conversation
            self.conversation.add_message(
                "assistant", 
                response,
                {"context_docs": [doc["text"] for doc in context_docs]}
            )
            
            return response
            
        except Exception as e:
            error_response = f"I encountered an error: {str(e)}"
            self.conversation.add_message(
                "assistant",
                error_response,
                {"error": str(e)}
            )
            return error_response

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation = Conversation(max_history=self.conversation.max_history) 