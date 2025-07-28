#!/usr/bin/env python3
"""
Media Embeddings Pipeline

This script extracts media analysis data from PostgreSQL and creates embeddings
in QDrant using Ollama models. It supports multiple embedding types, collection
management, and multi-tenancy.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import asyncpg
import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    OptimizersConfig,
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration management with fallback hierarchy: args -> env -> defaults"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
    def get(self, key: str, arg_name: str = None, default: Any = None) -> Any:
        """Get config value with fallback hierarchy"""
        # Try args first
        if arg_name and hasattr(self.args, arg_name):
            value = getattr(self.args, arg_name)
            if value is not None:
                return value
        
        # Try environment variable
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
            
        # Return default
        return default

    @property
    def db_host(self) -> str:
        return self.get('DB_HOST', 'db_host', 'localhost')
    
    @property
    def db_port(self) -> int:
        return int(self.get('DB_PORT', 'db_port', 5432))
    
    @property
    def db_name(self) -> str:
        return self.get('DB_NAME', 'db_name', 'media_rec')
    
    @property
    def db_user(self) -> str:
        return self.get('DB_USER', 'db_user', 'postgres')
    
    @property
    def db_password(self) -> str:
        return self.get('DB_PASSWORD', 'db_password', '')
    
    @property
    def qdrant_host(self) -> str:
        return self.get('QDRANT_HOST', 'qdrant_host', 'localhost')
    
    @property
    def qdrant_port(self) -> int:
        return int(self.get('QDRANT_PORT', 'qdrant_port', 6333))
    
    @property
    def qdrant_api_key(self) -> Optional[str]:
        return self.get('QDRANT_API_KEY', 'qdrant_api_key')
    
    @property
    def ollama_host(self) -> str:
        return self.get('OLLAMA_HOST', 'ollama_host', 'localhost')
    
    @property
    def ollama_port(self) -> int:
        return int(self.get('OLLAMA_PORT', 'ollama_port', 11434))
    
    @property
    def embedding_model(self) -> str:
        return self.get('EMBEDDING_MODEL', 'embedding_model', 'bge-m3:latest')
    
    @property
    def sparse_model(self) -> str:
        return self.get('SPARSE_MODEL', 'sparse_model', 'bge-m3:latest3')
    
    @property
    def batch_size(self) -> int:
        return int(self.get('BATCH_SIZE', 'batch_size', 10))
    
    @property
    def tenant_id(self) -> str:
        return self.get('TENANT_ID', 'tenant_id', 'default')


class OllamaEmbedder:
    """Handle embedding generation using Ollama"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = f"http://{config.ollama_host}:{config.ollama_port}"
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Get dense embedding for text"""
        model = model or self.config.embedding_model
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            logger.error(f"Error getting embedding for text (length {len(text)}): {e}")
            raise
    
    async def get_sparse_embedding(self, text: str, model: str = None) -> Dict[int, float]:
        """Get sparse embedding for text (simplified - would need actual sparse model)"""
        # Note: This is a placeholder. Real sparse embeddings would require
        # a model that outputs sparse vectors or additional processing
        model = model or self.config.sparse_model
        
        # For demo purposes, create a simple sparse representation
        # In practice, you'd use a proper sparse embedding model
        words = text.lower().split()
        sparse_vec = {}
        for i, word in enumerate(set(words)):
            sparse_vec[hash(word) % 10000] = words.count(word) / len(words)
        
        return sparse_vec
    
    async def get_batch_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text, model)
            embeddings.append(embedding)
            await asyncio.sleep(0.1)  # Rate limiting
        return embeddings


class QdrantManager:
    """Manage QDrant collections and operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            api_key=config.qdrant_api_key,
            timeout=60
        )
        self.collections = {
            'media_analysis': f"{config.tenant_id}_media_analysis",
            'media_profiles': f"{config.tenant_id}_media_profiles",
            'media_content': f"{config.tenant_id}_media_content"
        }
    
    async def ensure_collections(self, vector_size: int = 1536):
        """Create collections if they don't exist"""
        for collection_type, collection_name in self.collections.items():
            try:
                # Check if collection exists
                collections = self.client.get_collections()
                existing_names = [c.name for c in collections.collections]
                
                if collection_name not in existing_names:
                    logger.info(f"Creating collection: {collection_name}")
                    
                    # Create collection with dense and sparse vectors
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        ),
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams(),
                        },
                    )
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error managing collection {collection_name}: {e}")
                raise
    
    def upsert_points(self, collection_name: str, points: List[PointStruct]):
        """Upsert points to collection"""
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to {collection_name}")
        except Exception as e:
            logger.error(f"Error upserting points to {collection_name}: {e}")
            raise
    
    def delete_points(self, collection_name: str, point_ids: List[str]):
        """Delete points from collection"""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting points from {collection_name}: {e}")
            raise


class DatabaseManager:
    """Handle PostgreSQL database operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pool = None
    
    async def __aenter__(self):
        self.pool = await asyncpg.create_pool(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            min_size=1,
            max_size=5
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()
    
    async def get_media_analysis_data(self, limit: int = None) -> List[Dict]:
        """Get all ollama analysis data"""
        query = """
        SELECT 
            oa.id,
            oa.media_item_id,
            oa.analysis_type,
            oa.model_used,
            oa.analysis_result,
            oa.confidence_score,
            oa.created_at,
            mi.title,
            mi.overview,
            mi.media_type,
            mi.release_date
        FROM ollama_analysis oa
        JOIN media_items mi ON oa.media_item_id = mi.id
        ORDER BY oa.created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
    
    async def get_media_profiles(self, limit: int = None) -> List[Dict]:
        """Get media profiles data"""
        query = """
        SELECT 
            mp.media_item_id,
            mp.primary_themes,
            mp.mood_tags,
            mp.style_descriptors,
            mp.target_audience,
            mp.complexity_level,
            mp.emotional_intensity,
            mp.content_warnings,
            mp.similar_to_ids,
            mp.created_at,
            mi.title,
            mi.overview,
            mi.media_type,
            mi.release_date
        FROM media_profiles mp
        JOIN media_items mi ON mp.media_item_id = mi.id
        ORDER BY mp.created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]
    
    async def update_embedding_status(self, media_item_id: str, analysis_id: str, embedding_model: str):
        """Update embedding status for processed item"""
        query = """
        INSERT INTO embedding_status (analysis_id, embedded_at, embedding_model)
        VALUES ($1, $2, $3)
        ON CONFLICT (analysis_id) DO UPDATE SET
            embedded_at = $2,
            embedding_model = $3
        """
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(query, analysis_id, datetime.now(timezone.utc), embedding_model)
        except Exception as e:
            logger.warning(f"Could not update embedding status (table may not exist): {e}")
    
    async def get_unembedded_items(self, limit: int = None) -> List[Dict]:
        """Get items that haven't been embedded yet"""
        # First try with embedding_status table
        query_with_status = """
        SELECT 
            oa.id,
            oa.media_item_id,
            oa.analysis_type,
            oa.model_used,
            oa.analysis_result,
            oa.confidence_score,
            oa.created_at,
            mi.title,
            mi.overview,
            mi.media_type,
            mi.release_date
        FROM ollama_analysis oa
        JOIN media_items mi ON oa.media_item_id = mi.id
        LEFT JOIN embedding_status es ON oa.id = es.analysis_id
        WHERE es.analysis_id IS NULL
        ORDER BY oa.created_at DESC
        """
        
        # Fallback query without embedding_status
        query_fallback = """
        SELECT 
            oa.id,
            oa.media_item_id,
            oa.analysis_type,
            oa.model_used,
            oa.analysis_result,
            oa.confidence_score,
            oa.created_at,
            mi.title,
            mi.overview,
            mi.media_type,
            mi.release_date
        FROM ollama_analysis oa
        JOIN media_items mi ON oa.media_item_id = mi.id
        ORDER BY oa.created_at DESC
        """
        
        if limit:
            query_with_status += f" LIMIT {limit}"
            query_fallback += f" LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(query_with_status)
                return [dict(row) for row in rows]
            except Exception:
                logger.warning("embedding_status table not found, processing all items")
                rows = await conn.fetch(query_fallback)
                return [dict(row) for row in rows]


class MediaEmbeddingsPipeline:
    """Main pipeline for processing media embeddings"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_embedding_text(self, analysis_data: Dict) -> str:
        """Create text representation for embedding"""
        parts = []
        
        # Basic media info
        if analysis_data.get('title'):
            parts.append(f"Title: {analysis_data['title']}")
        if analysis_data.get('overview'):
            parts.append(f"Overview: {analysis_data['overview']}")
        if analysis_data.get('media_type'):
            parts.append(f"Type: {analysis_data['media_type']}")
        
        # Analysis results
        analysis_result = analysis_data.get('analysis_result', {})
        if isinstance(analysis_result, str):
            try:
                analysis_result = json.loads(analysis_result)
            except json.JSONDecodeError:
                analysis_result = {}
        
        analysis_type = analysis_data.get('analysis_type', '')
        
        if analysis_type == 'mood_analysis':
            mood_parts = []
            if analysis_result.get('overall_mood'):
                mood_parts.append(f"Mood: {analysis_result['overall_mood']}")
            if analysis_result.get('atmosphere_tags'):
                mood_parts.append(f"Atmosphere: {', '.join(analysis_result['atmosphere_tags'])}")
            if analysis_result.get('viewer_experience'):
                mood_parts.append(f"Experience: {analysis_result['viewer_experience']}")
            parts.extend(mood_parts)
        
        elif analysis_type == 'theme_analysis':
            if analysis_result.get('major_themes'):
                parts.append(f"Major themes: {', '.join(analysis_result['major_themes'])}")
            if analysis_result.get('minor_themes'):  # Note: your data has 'minor_ themes' with space
                parts.append(f"Minor themes: {', '.join(analysis_result['minor_themes'])}")
        
        elif analysis_type == 'content_profile':
            if analysis_result.get('mood_tags'):
                parts.append(f"Mood tags: {', '.join(analysis_result['mood_tags'])}")
            if analysis_result.get('key_elements'):
                parts.append(f"Key elements: {', '.join(analysis_result['key_elements'])}")
            if analysis_result.get('target_audience'):
                parts.append(f"Target audience: {analysis_result['target_audience']}")
        
        elif analysis_type == 'similarity_analysis':
            if analysis_result.get('distinctive_elements'):
                parts.append(f"Distinctive: {', '.join(analysis_result['distinctive_elements'])}")
            if analysis_result.get('audience_overlap'):
                parts.append(f"Similar to: {', '.join(analysis_result['audience_overlap'])}")
        
        elif analysis_type == 'recommendation_profile':
            if analysis_result.get('discovery_tags'):
                parts.append(f"Discovery: {', '.join(analysis_result['discovery_tags'])}")
            if analysis_result.get('recommendation_contexts'):
                parts.append(f"Contexts: {', '.join(analysis_result['recommendation_contexts'])}")
        
        return ' | '.join(parts)
    
    def create_point_struct(self, analysis_data: Dict, dense_embedding: List[float], 
                          sparse_embedding: Dict[int, float]) -> PointStruct:
        """Create QDrant point structure"""
        
        # Create payload with all relevant metadata
        payload = {
            'media_item_id': str(analysis_data['media_item_id']),
            'analysis_id': str(analysis_data['id']),
            'analysis_type': analysis_data['analysis_type'],
            'model_used': analysis_data['model_used'],
            'confidence_score': float(analysis_data['confidence_score']),
            'title': analysis_data.get('title', ''),
            'media_type': analysis_data.get('media_type', ''),
            'tenant_id': self.config.tenant_id,
            'created_at': analysis_data['created_at'].isoformat() if analysis_data.get('created_at') else None,
        }
        
        # Add parsed analysis results
        analysis_result = analysis_data.get('analysis_result', {})
        if isinstance(analysis_result, str):
            try:
                analysis_result = json.loads(analysis_result)
            except json.JSONDecodeError:
                analysis_result = {}
        
        payload['analysis_result'] = analysis_result
        
        # Convert sparse embedding to SparseVector format
        sparse_vector = SparseVector(
            indices=list(sparse_embedding.keys()),
            values=list(sparse_embedding.values())
        )
        
        return PointStruct(
            id=str(analysis_data['id']),
            vector=dense_embedding,  # Default dense vector
            payload={
                **payload,
                "sparse_vector": sparse_vector  # Store sparse in payload for now
            }
        )
    
    async def process_batch(self, db_manager: DatabaseManager, 
                          qdrant_manager: QdrantManager, 
                          embedder: OllamaEmbedder, 
                          batch_data: List[Dict]) -> int:
        """Process a batch of analysis data"""
        
        try:
            # Create embedding texts
            texts = [self.create_embedding_text(item) for item in batch_data]
            
            # Get dense embeddings
            logger.info(f"Getting dense embeddings for {len(texts)} items...")
            dense_embeddings = await embedder.get_batch_embeddings(texts)
            
            # Get sparse embeddings
            logger.info(f"Getting sparse embeddings for {len(texts)} items...")
            sparse_embeddings = []
            for text in texts:
                sparse_emb = await embedder.get_sparse_embedding(text)
                sparse_embeddings.append(sparse_emb)
            
            # Create points for each collection type
            points_by_collection = {}
            
            for i, analysis_data in enumerate(batch_data):
                if i >= len(dense_embeddings) or i >= len(sparse_embeddings):
                    logger.warning(f"Skipping item {i} due to missing embeddings")
                    continue
                
                point = self.create_point_struct(
                    analysis_data,
                    dense_embeddings[i],
                    sparse_embeddings[i]
                )
                
                # Determine target collection based on analysis type
                analysis_type = analysis_data.get('analysis_type', '')
                if analysis_type in ['mood_analysis', 'theme_analysis']:
                    collection_key = 'media_analysis'
                elif analysis_type in ['recommendation_profile', 'similarity_analysis']:
                    collection_key = 'media_profiles'
                else:
                    collection_key = 'media_content'
                
                collection_name = qdrant_manager.collections[collection_key]
                
                if collection_name not in points_by_collection:
                    points_by_collection[collection_name] = []
                points_by_collection[collection_name].append(point)
            
            # Upsert points to appropriate collections
            total_processed = 0
            for collection_name, points in points_by_collection.items():
                if points:
                    qdrant_manager.upsert_points(collection_name, points)
                    total_processed += len(points)
            
            # Update embedding status
            for analysis_data in batch_data:
                await db_manager.update_embedding_status(
                    str(analysis_data['media_item_id']),
                    str(analysis_data['id']),
                    self.config.embedding_model
                )
            
            logger.info(f"Successfully processed batch of {total_processed} items")
            return total_processed
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    async def run(self):
        """Run the embedding pipeline"""
        logger.info("Starting Media Embeddings Pipeline")
        logger.info(f"Configuration: DB={self.config.db_host}:{self.config.db_port}, "
                   f"QDrant={self.config.qdrant_host}:{self.config.qdrant_port}, "
                   f"Ollama={self.config.ollama_host}:{self.config.ollama_port}")
        
        async with DatabaseManager(self.config) as db_manager:
            async with OllamaEmbedder(self.config) as embedder:
                qdrant_manager = QdrantManager(self.config)
                
                # Test embedding to get vector size
                test_embedding = await embedder.get_embedding("test text")
                vector_size = len(test_embedding)
                logger.info(f"Vector size: {vector_size}")
                
                # Ensure collections exist
                await qdrant_manager.ensure_collections(vector_size)
                
                # Get unembedded items
                unembedded_items = await db_manager.get_unembedded_items()
                total_items = len(unembedded_items)
                
                if total_items == 0:
                    logger.info("No unembedded items found")
                    return
                
                logger.info(f"Found {total_items} unembedded items")
                
                # Process in batches
                batch_size = self.config.batch_size
                total_processed = 0
                
                for i in range(0, total_items, batch_size):
                    batch = unembedded_items[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_items + batch_size - 1) // batch_size
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} "
                               f"({len(batch)} items)")
                    
                    try:
                        processed = await self.process_batch(
                            db_manager, qdrant_manager, embedder, batch
                        )
                        total_processed += processed
                        
                        # Small delay between batches
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_num}: {e}")
                        continue
                
                logger.info(f"Pipeline completed. Processed {total_processed}/{total_items} items")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Media Embeddings Pipeline - Extract from PostgreSQL and embed in QDrant"
    )
    
    # PostgreSQL arguments
    db_group = parser.add_argument_group('Database')
    db_group.add_argument('--db-host', help='Database host')
    db_group.add_argument('--db-port', type=int, help='Database port')
    db_group.add_argument('--db-name', help='Database name')
    db_group.add_argument('--db-user', help='Database username')
    db_group.add_argument('--db-password', help='Database password')
    
    # QDrant arguments
    qdrant_group = parser.add_argument_group('QDrant')
    qdrant_group.add_argument('--qdrant-host', help='QDrant host')
    qdrant_group.add_argument('--qdrant-port', type=int, help='QDrant port')
    qdrant_group.add_argument('--qdrant-api-key', help='QDrant API key')
    
    # Ollama arguments
    ollama_group = parser.add_argument_group('Ollama')
    ollama_group.add_argument('--ollama-host', help='Ollama host')
    ollama_group.add_argument('--ollama-port', type=int, help='Ollama port')
    ollama_group.add_argument('--embedding-model', help='Ollama embedding model')
    ollama_group.add_argument('--sparse-model', help='Ollama sparse embedding model')
    
    # Processing arguments
    processing_group = parser.add_argument_group('Processing')
    processing_group.add_argument('--batch-size', type=int, help='Batch size for processing')
    processing_group.add_argument('--tenant-id', help='Tenant ID for multi-tenancy')
    
    # Other arguments
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no actual embedding)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = Config(args)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual embedding will be performed")
        return
    
    try:
        pipeline = MediaEmbeddingsPipeline(config)
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())