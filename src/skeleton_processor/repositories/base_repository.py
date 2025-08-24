"""Base repository class with database connection management."""

import logging
import psycopg2
import psycopg2.extras
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository class with PostgreSQL connection management."""
    
    def __init__(self, database_url: str):
        """Initialize repository with database connection."""
        self.database_url = database_url
        self._connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self._connection = psycopg2.connect(
                self.database_url,
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            self._connection.autocommit = False
            logger.info("Database connection established")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    @contextmanager
    def get_cursor(self, commit: bool = True):
        """Get database cursor with transaction management."""
        if not self._connection:
            self.connect()
        
        cursor = self._connection.cursor()
        try:
            yield cursor
            if commit:
                self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            cursor.close()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch: bool = True) -> Optional[List[Dict]]:
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return None
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> None:
        """Execute a query with multiple parameter sets."""
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
        """
        result = self.execute_query(query, (table_name,))
        return result[0]['exists'] if result else False
    
    def check_extension_exists(self, extension_name: str) -> bool:
        """Check if a PostgreSQL extension is installed."""
        query = "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = %s);"
        result = self.execute_query(query, (extension_name,))
        return result[0]['exists'] if result else False
    
    def get_table_info(self, table_name: str) -> List[Dict]:
        """Get column information for a table."""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
        """
        return self.execute_query(query, (table_name,))
    
    def format_vector_for_db(self, vector: List[float]) -> str:
        """Format vector for PostgreSQL pgvector insertion."""
        return '[' + ','.join(map(str, vector)) + ']'
    
    def parse_vector_from_db(self, vector_str: str) -> List[float]:
        """Parse vector from PostgreSQL pgvector format."""
        if not vector_str:
            return []
        # Remove brackets and split by comma
        vector_str = vector_str.strip('[]')
        return [float(x.strip()) for x in vector_str.split(',') if x.strip()]
    
    def format_json_for_db(self, data: Dict[str, Any]) -> str:
        """Format dictionary for PostgreSQL JSONB storage."""
        return json.dumps(data)
    
    def parse_json_from_db(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON from PostgreSQL JSONB format."""
        if not json_str:
            return {}
        if isinstance(json_str, dict):
            return json_str
        return json.loads(json_str)
    
    def validate_database_setup(self) -> Tuple[bool, List[str]]:
        """Validate that database is properly set up for skeleton processor."""
        issues = []
        
        try:
            # Check pgvector extension
            if not self.check_extension_exists('vector'):
                issues.append("pgvector extension is not installed")
            
            # Check required tables
            required_tables = ['documents', 'paragraphs', 'clusters', 'skeleton_documents']
            for table in required_tables:
                if not self.check_table_exists(table):
                    issues.append(f"Table '{table}' does not exist")
            
            # Check vector column in paragraphs table
            if self.check_table_exists('paragraphs'):
                columns = self.get_table_info('paragraphs')
                # PostgreSQL reports vector columns as 'USER-DEFINED' type
                embedding_columns = [col for col in columns if col['column_name'] == 'embedding']
                if not embedding_columns:
                    issues.append("Paragraphs table missing embedding column")
                elif embedding_columns[0]['data_type'] not in ['USER-DEFINED', 'vector']:
                    issues.append("Paragraphs table embedding column is not vector type")
            
        except Exception as e:
            issues.append(f"Database validation failed: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current database connection."""
        if not self._connection:
            return {"status": "disconnected"}
        
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("SELECT version();")
                pg_version = cursor.fetchone()['version']
                
                cursor.execute("SELECT current_database();")
                current_db = cursor.fetchone()['current_database']
                
                cursor.execute("SELECT current_user;")
                current_user = cursor.fetchone()['current_user']
                
                return {
                    "status": "connected",
                    "postgresql_version": pg_version,
                    "database": current_db,
                    "user": current_user,
                    "connection_url": self.database_url.split('@')[1] if '@' in self.database_url else self.database_url
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
