#!/usr/bin/env python3
"""
Vector Embedding Script for Media Recommendation System

This script processes media analysis data from PostgreSQL and creates
vector embeddings using Ollama, storing them in Qdrant collections.
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import psycopg2
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct


class VectorEmbedder:
    """Main class for handling vector embedding operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.postgres_conn = None
        self.qdrant_client = None
        
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def connect_postgres(self):
        """Establish PostgreSQL connection."""
        try:
            self.postgres_conn = psycopg2.connect(
                host=self.config['db_host'],
                port=self.config['db_port'],
                user=self.config['db_user'],
                password=self.config['db_password'],
                database=self.config['db_name']
            )
            self.postgres_conn.autocommit = True  # Enable autocommit for DDL operations
            self.logger.info("Connected to PostgreSQL")
            self._create_tracking_table()
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)
            
    def _create_tracking_table(self):
        """Create the embedding_status tracking table."""
        try:
            cursor = self.postgres_conn.cursor() # type: ignore
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_status (
                    analysis_id UUID PRIMARY KEY,
                    embedded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    embedding_model VARCHAR(100)
                )
            """)
            cursor.close()
            self.logger.info("Ensured embedding_status table exists")
        except Exception as e:
            self.logger.error(f"Failed to create tracking table: {e}")
            
    def connect_qdrant(self):
        """Establish Qdrant connection and ensure collections exist."""
        try:
            self.qdrant_client = QdrantClient(
                host=self.config['qdrant_host'],
                port=self.config['qdrant_port']
            )
            self.logger.info("Connected to Qdrant")
            self.setup_qdrant_collections()
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            sys.exit(1)
            
    def setup_qdrant_collections(self):
        """Create or update Qdrant collections with proper schema."""
        collections_config = {
            "media_content": {
                "vectors": VectorParams(size=768, distance=Distance.COSINE),
                "indexes": [
                    ("media_item_id", "keyword"),
                    ("media_type", "keyword"),
                    ("genres", "keyword"),
                    ("release_year", "integer")
                ]
            },
            "media_analysis": {
                "vectors": VectorParams(size=768, distance=Distance.COSINE),
                "indexes": [
                    ("media_item_id", "keyword"),
                    ("analysis_type", "keyword"),
                    ("mood_tags", "keyword"),
                    ("target_audience", "keyword"),
                    ("complexity_level", "integer"),
                    ("emotional_intensity", "integer")
                ]
            }
        }
        
        for collection_name, config in collections_config.items():
            try:
                # Check if collection exists
                collections = self.qdrant_client.get_collections().collections # type: ignore
                collection_exists = any(c.name == collection_name for c in collections)
                
                if not collection_exists:
                    self.qdrant_client.create_collection( # type: ignore
                        collection_name=collection_name,
                        vectors_config=config["vectors"]
                    )
                    self.logger.info(f"Created collection: {collection_name}")
                else:
                    self.logger.info(f"Collection already exists: {collection_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to setup collection {collection_name}: {e}")
                
    def _format_dict_or_list(self, data) -> str:
        """Format dictionary or list data safely."""
        if isinstance(data, dict):
            return ', '.join([f"{k}: {v}" for k, v in data.items()])
        elif isinstance(data, list):
            return ', '.join(str(item) for item in data)
        else:
            return str(data)
            
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from Ollama."""
        try:
            response = requests.post(
                f"http://{self.config['embed_host']}:{self.config['embed_port']}/api/embeddings",
                json={
                    "model": self.config['embed_model'],
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['embedding']
        except Exception as e:
            self.logger.error(f"Failed to get embedding: {e}")
            return []
            
    def get_existing_embeddings(self) -> set:
        """Get set of already embedded analysis IDs from PostgreSQL tracking table."""
        try:
            cursor = self.postgres_conn.cursor() # type: ignore
            
            # Get existing embeddings for current model
            cursor.execute("""
                SELECT analysis_id FROM embedding_status 
                WHERE embedding_model = %s
            """, (self.config['embed_model'],))
            
            existing_ids = {str(row[0]) for row in cursor.fetchall()}
            cursor.close()
            
            self.logger.info(f"Found {len(existing_ids)} existing embeddings for model {self.config['embed_model']}")
            return existing_ids
            
        except Exception as e:
            self.logger.warning(f"Could not fetch existing embeddings: {e}")
            return set()
    
    def fetch_media_analysis_data(self, existing_ids: set = None) -> List[Dict[str, Any]]: # type: ignore
        """Fetch media analysis data from PostgreSQL, optionally filtering already processed."""
        base_query = """
        SELECT 
            oa.id,
            oa.media_item_id,
            oa.analysis_type,
            oa.model_used,
            oa.analysis_result,
            oa.confidence_score,
            oa.created_at,
            mi.title,
            mi.media_type,
            mi.release_date,
            mi.overview,
            mi.tagline,
            array_agg(DISTINCT g.name) as genres
        FROM ollama_analysis oa
        JOIN media_items mi ON oa.media_item_id = mi.id
        LEFT JOIN media_genres mg ON mi.id = mg.media_item_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE oa.analysis_result IS NOT NULL
        """
        
        # Add filter for new records only
        if existing_ids and len(existing_ids) > 0:
            placeholders = ','.join(['%s'] * len(existing_ids))
            base_query += f" AND oa.id::text NOT IN ({placeholders})"
        
        query = base_query + """
        GROUP BY oa.id, oa.media_item_id, oa.analysis_type, oa.model_used, 
                 oa.analysis_result, oa.confidence_score, oa.created_at,
                 mi.title, mi.media_type, mi.release_date, mi.overview, mi.tagline
        ORDER BY oa.created_at DESC
        """
        
        try:
            cursor = self.postgres_conn.cursor() # type: ignore
            if existing_ids and len(existing_ids) > 0:
                cursor.execute(query, list(existing_ids))
            else:
                cursor.execute(query)
                
            columns = [desc[0] for desc in cursor.description] # type: ignore
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
                
            cursor.close()
            self.logger.info(f"Fetched {len(results)} new analysis records")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            return []
            
    def create_embeddings_text(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """Create text for embeddings based on analysis type."""
        analysis_result = record['analysis_result']
        base_info = f"Title: {record['title']}\nType: {record['media_type']}\n"
        
        if record['overview']:
            base_info += f"Overview: {record['overview']}\n"
        if record['tagline']:
            base_info += f"Tagline: {record['tagline']}\n"
        if record['genres'] and record['genres'][0]:
            base_info += f"Genres: {', '.join(record['genres'])}\n"
            
        analysis_type = record['analysis_type'].lower()
        
        if analysis_type == 'content_profile':
            content_text = base_info + f"""
Content Profile:
Primary Themes: {', '.join(analysis_result.get('primary_themes', []))}
Mood Tags: {', '.join(analysis_result.get('mood_tags', []))}
Style Descriptors: {', '.join(analysis_result.get('style_descriptors', []))}
Target Audience: {analysis_result.get('target_audience', '')}
Key Elements: {', '.join(analysis_result.get('key_elements', []))}
Standout Features: {', '.join(analysis_result.get('standout_features', []))}
""".strip()
            
            analysis_text = f"""
Analysis Type: Content Profile
Complexity Level: {analysis_result.get('complexity_level', 0)}
Emotional Intensity: {analysis_result.get('emotional_intensity', 0)}
Similar Vibes: {', '.join(analysis_result.get('similar_vibes', []))}
Content Warnings: {', '.join(analysis_result.get('content_warnings', []))}
Recommended Context: {analysis_result.get('recommended_viewing_context', '')}
""".strip()
            
        elif analysis_type == 'mood_analysis':
            content_text = base_info + f"""
Mood Analysis:
Overall Mood: {analysis_result.get('overall_mood', '')}
Pace: {analysis_result.get('pace', '')}
Energy Level: {analysis_result.get('energy_level', 0)}
Emotional Weight: {analysis_result.get('emotional_weight', '')}
""".strip()
            
            analysis_text = f"""
Analysis Type: Mood Analysis
Humor Level: {analysis_result.get('humor_level', 0)}
Tension Level: {analysis_result.get('tension_level', 0)}
Darkness Level: {analysis_result.get('darkness_level', 0)}
Atmosphere: {', '.join(analysis_result.get('atmosphere_tags', []))}
Mood Progression: {analysis_result.get('mood_progression', '')}
Viewer Experience: {analysis_result.get('viewer_experience', '')}
""".strip()
            
        elif analysis_type == 'theme_analysis':
            content_text = base_info + f"""
Theme Analysis:
Major Themes: {', '.join(analysis_result.get('major_themes', []))}
Minor Themes: {', '.join(analysis_result.get('minor_themes', []))}
Cultural Context: {analysis_result.get('cultural_context', '')}
""".strip()
            
            analysis_text = f"""
Analysis Type: Theme Analysis
Emotional Journey: {analysis_result.get('emotional_journey', '')}
Symbolic Elements: {self._format_dict_or_list(analysis_result.get('symbolic_elements', {}))}
Character Archetypes: {self._format_dict_or_list(analysis_result.get('character_archetypes', {}))}
Philosophical Questions: {', '.join(analysis_result.get('philosophical_questions', []))}
""".strip()
        else:
            content_text = base_info + f"Analysis: {json.dumps(analysis_result, indent=2)}"
            analysis_text = f"Analysis Type: {analysis_type}\nContent: {json.dumps(analysis_result)}"
            
        return content_text, analysis_text
        
    def extract_metadata(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract metadata for Qdrant payloads."""
        analysis_result = record['analysis_result']
        
        # Common metadata
        common_meta = {
            "media_item_id": str(record['media_item_id']),
            "title": record['title'],
            "media_type": record['media_type'],
            "confidence_score": float(record['confidence_score']),
            "created_at": record['created_at'].isoformat() if record['created_at'] else None
        }
        
        # Extract release year
        release_year = None
        if record['release_date']:
            release_year = record['release_date'].year
            
        # Media content metadata
        content_meta = {
            **common_meta,
            "genres": [g for g in record['genres'] if g] if record['genres'] else [],
            "release_year": release_year,
            "overview": record['overview'] or "",
            "tagline": record['tagline'] or ""
        }
        
        # Media analysis metadata
        analysis_meta = {
            **common_meta,
            "analysis_type": record['analysis_type'].lower(),
            "model_used": record['model_used'],
            "mood_tags": analysis_result.get('mood_tags', []),
            "target_audience": analysis_result.get('target_audience', ''),
            "complexity_level": analysis_result.get('complexity_level', 0),
            "emotional_intensity": analysis_result.get('emotional_intensity', 0)
        }
        
        return content_meta, analysis_meta
        
    def update_embedding_status(self, analysis_ids: List[str]):
        """Update tracking table with successfully embedded analysis IDs."""
        if not analysis_ids:
            return
            
        try:
            cursor = self.postgres_conn.cursor() # type: ignore
            values = [(aid, self.config['embed_model']) for aid in analysis_ids]
            
            cursor.executemany("""
                INSERT INTO embedding_status (analysis_id, embedding_model)
                VALUES (%s, %s)
                ON CONFLICT (analysis_id) DO UPDATE SET
                    embedded_at = CURRENT_TIMESTAMP,
                    embedding_model = EXCLUDED.embedding_model
            """, values)
            
            cursor.close()
            self.logger.info(f"Updated embedding status for {len(analysis_ids)} records")
            
        except Exception as e:
            self.logger.error(f"Failed to update embedding status: {e}")
            
    def process_batch(self, records: List[Dict[str, Any]]) -> int:
        """Process a batch of records and store in Qdrant."""
        content_points = []
        analysis_points = []
        processed_count = 0
        processed_ids = []
        
        for record in records:
            try:
                # Create embedding texts
                content_text, analysis_text = self.create_embeddings_text(record)
                
                # Get embeddings
                content_embedding = self.get_embedding(content_text)
                analysis_embedding = self.get_embedding(analysis_text)
                
                if not content_embedding or not analysis_embedding:
                    self.logger.warning(f"Failed to get embeddings for record {record['id']}")
                    continue
                    
                # Extract metadata
                content_meta, analysis_meta = self.extract_metadata(record)
                
                # Create points with proper UUID format
                content_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"content_{record['id']}"))
                analysis_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"analysis_{record['id']}"))
                
                content_points.append(PointStruct(
                    id=content_id,
                    vector=content_embedding,
                    payload=content_meta
                ))
                
                analysis_points.append(PointStruct(
                    id=analysis_id,
                    vector=analysis_embedding,
                    payload=analysis_meta
                ))
                
                processed_count += 1
                processed_ids.append(str(record['id']))
                
            except Exception as e:
                self.logger.error(f"Failed to process record {record['id']}: {e}")
                continue
                
        # Store in Qdrant
        try:
            if content_points:
                self.qdrant_client.upsert( # type: ignore
                    collection_name="media_content",
                    points=content_points
                )
                
            if analysis_points:
                self.qdrant_client.upsert( # type: ignore
                    collection_name="media_analysis",
                    points=analysis_points
                )
                
            # Track successful embeddings
            if processed_ids:
                self.update_embedding_status(processed_ids)
                
            self.logger.info(f"Stored {len(content_points)} content points and {len(analysis_points)} analysis points")
            
        except Exception as e:
            self.logger.error(f"Failed to store batch in Qdrant: {e}")
            
        return processed_count
        
    def run(self):
        """Main execution method."""
        self.logger.info("Starting vector embedding process")
        
        # Establish connections
        self.connect_postgres()
        self.connect_qdrant()
        
        # Get existing embeddings to skip (unless forced reprocessing)
        existing_ids = set() if self.config['force_reprocess'] else self.get_existing_embeddings()
        
        if self.config['force_reprocess']:
            self.logger.info("Force reprocessing enabled - will process all records")
        
        # Fetch only new data
        records = self.fetch_media_analysis_data(existing_ids)
        if not records:
            self.logger.info("No new data to process")
            return
            
        # Process in batches
        batch_size = self.config['batch_size']
        total_processed = 0
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_num = i // batch_size + 1
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            processed = self.process_batch(batch)
            total_processed += processed
            
        self.logger.info(f"Processing complete. Total records processed: {total_processed}")
        
        # Cleanup
        if self.postgres_conn:
            self.postgres_conn.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vector Embedding Script for Media Analysis")
    
    # Database arguments
    parser.add_argument('--db-host', help='PostgreSQL host')
    parser.add_argument('--db-port', type=int, help='PostgreSQL port')
    parser.add_argument('--db-user', help='PostgreSQL username')
    parser.add_argument('--db-password', help='PostgreSQL password')
    parser.add_argument('--db-name', help='PostgreSQL database name')
    
    # Qdrant arguments
    parser.add_argument('--qdrant-host', help='Qdrant host')
    parser.add_argument('--qdrant-port', type=int, help='Qdrant port')
    
    # Embedding arguments
    parser.add_argument('--embed-host', help='Embedding service host')
    parser.add_argument('--embed-port', type=int, help='Embedding service port')
    parser.add_argument('--embed-model', help='Embedding model name')
    
    # Processing arguments
    parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing all records, ignoring existing embeddings')
    
    return parser.parse_args()


def get_config():
    """Get configuration from arguments, environment, and defaults."""
    args = parse_arguments()
    
    config = {
        'db_host': args.db_host or os.getenv('POSTGRES_HOST', 'localhost'),
        'db_port': args.db_port or int(os.getenv('POSTGRES_PORT', '5432')),
        'db_user': args.db_user or os.getenv('POSTGRES_USERNAME', 'postgres'),
        'db_password': args.db_password or os.getenv('POSTGRES_PASSWORD', ''),
        'db_name': args.db_name or os.getenv('POSTGRES_DATABASE', 'media_rec'),
        'qdrant_host': args.qdrant_host or os.getenv('QDRANT_HOST', 'localhost'),
        'qdrant_port': args.qdrant_port or int(os.getenv('QDRANT_PORT', '6333')),
        'embed_host': args.embed_host or os.getenv('EMBED_HOST', 'localhost'),
        'embed_port': args.embed_port or int(os.getenv('EMBED_PORT', '11434')),
        'embed_model': args.embed_model or os.getenv('EMBED_MODEL', 'nomic-embed-text:latest'),
        'batch_size': args.batch_size or int(os.getenv('BATCH_SIZE', '10')),
        'force_reprocess': args.force_reprocess
    }
    
    return config


def main():
    """Main entry point."""
    try:
        config = get_config()
        embedder = VectorEmbedder(config)
        embedder.run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
