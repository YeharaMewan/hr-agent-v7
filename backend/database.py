import os
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection - supports both SQLite and PostgreSQL"""
    try:
        # Check if PostgreSQL database URL is provided
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            # PostgreSQL connection
            conn = psycopg2.connect(
                database_url,
                cursor_factory=RealDictCursor
            )
            return conn
        else:
            # SQLite fallback
            db_path = os.getenv('DB_PATH', 'hr_system.db')
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
            return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def execute_query(query: str, params: tuple = (), fetch: bool = True) -> Optional[List[Dict[str, Any]]]:
    """
    Execute a database query with parameters
    
    Args:
        query: SQL query string
        params: Query parameters
        fetch: Whether to fetch results (True for SELECT, False for INSERT/UPDATE/DELETE)
    
    Returns:
        List of dictionaries for SELECT queries, None for other queries
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(query, params)
        
        if fetch:
            # For SELECT queries
            if hasattr(cursor, 'fetchall'):
                results = cursor.fetchall()
                # Convert to list of dictionaries
                if results:
                    if hasattr(results[0], 'keys'):  # PostgreSQL RealDictCursor
                        return [dict(row) for row in results]
                    else:  # SQLite Row objects
                        return [dict(row) for row in results]
                return []
        else:
            # For INSERT/UPDATE/DELETE queries
            conn.commit()
            return None
            
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database query error: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def search_similar_documents(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar documents using embeddings
    This is a placeholder implementation
    """
    try:
        # Placeholder implementation - would use vector similarity search
        search_query = """
        SELECT id, document_name, chunk_text, 
               LENGTH(chunk_text) as relevance_score
        FROM hr_policies 
        WHERE chunk_text ILIKE %s
        ORDER BY relevance_score DESC
        LIMIT %s
        """
        results = execute_query(search_query, (f"%{query}%", limit))
        return results or []
    except Exception as e:
        logger.error(f"Document search error: {e}")
        return []