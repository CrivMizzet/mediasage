#!/usr/bin/env python3
"""
Vector Embedding Script for Media Recommendation System

This script processes media analysis data from PostgreSQL and creates
vector embeddings using BGE-M3 model with both dense and sparse vectors,
storing them in Qdrant collections.

python bge_embedding.py --db-host "192.168.0.39" --db-user postgres --db-password 8g1k9ap2 --db-name media_rec --qdrant-host 192.168.0.20 --qdrant-port 6333 --embed-host 192.168.0.150 --embed-port 11434 --embed-model bge-m3:latest --batch-size 10

"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any

import psycopg2
from psycopg2 import extras
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseIndexParams


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
            self.postgres_conn.autocommit = False
            self.logger.info("Connected to PostgreSQL")
            self._create_tracking_table()
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)
            
    def _create_tracking_table(self):
        """Create or update the embedding_status tracking table."""
        if not self.postgres_conn:
            self.logger.error("PostgreSQL connection not available.")
            return
        try:
            with self.postgres_conn.cursor() as cursor:
                # Check if embedding_status table exists
                cursor.execute("SELECT to_regclass('public.embedding_status');")
                result = cursor.fetchone()
                table_exists = result[0] if result else None

                if table_exists:
                    # Check if the primary key is on 'analysis_id'
                    cursor.execute("""
                        SELECT COUNT(*)
                        FROM pg_constraint
                        JOIN pg_attribute ON pg_attribute.attrelid = pg_constraint.conrelid AND pg_attribute.attnum = pg_constraint.conkey[1]
                        WHERE pg_constraint.conrelid = 'embedding_status'::regclass
                        AND pg_constraint.contype = 'p'
                        AND pg_attribute.attname = 'analysis_id'
                        AND array_length(pg_constraint.conkey, 1) = 1;
                    """)
                    result = cursor.fetchone()
                    is_old_schema = result[0] > 0 if result else False
                    if is_old_schema:
                        self.logger.info("Old embedding_status table schema detected. Dropping and recreating.")
                        cursor.execute("DROP TABLE embedding_status;")

                # Create the table with the correct schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_status (
                        media_item_id UUID NOT NULL,
                        embedding_model VARCHAR(100) NOT NULL,
                        embedded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (media_item_id, embedding_model)
                    )
                """)
            self.postgres_conn.commit()
            self.logger.info("Ensured embedding_status table exists with correct schema.")
        except psycopg2.Error as e:
            self.logger.error(f"Failed to create or update tracking table: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            
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
        """Create or update Qdrant collections with proper schema for BGE-M3."""
        if not self.qdrant_client:
            self.logger.error("Qdrant client not available.")
            return
        analysis_types = [
            "content_profile", "theme_analysis", "mood_analysis", 
            "similarity_analysis", "recommendation_profile"
        ]
        collection_names = ["media_content"] + [f"analysis_{atype}" for atype in analysis_types]

        for collection_name in collection_names:
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_exists = any(c.name == collection_name for c in collections)

                if not collection_exists:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                        },
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams(
                                index=models.SparseIndexParams(on_disk=False)
                            )
                        }
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

    def _parse_date(self, date_val: Optional[Any]) -> Optional[datetime]:
        """Safely parse date values that might be strings or date/datetime objects."""
        if isinstance(date_val, datetime):
            return date_val
        if isinstance(date_val, date):
            return datetime.combine(date_val, datetime.min.time())
        if isinstance(date_val, str):
            try:
                # Handle ISO format with or without timezone
                if 'Z' in date_val or '+' in date_val:
                    return datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                return datetime.fromisoformat(date_val)
            except ValueError:
                try:
                    # Fallback for other common formats
                    return datetime.strptime(date_val, '%Y-%m-%d %H:%M:%S.%f%z')
                except ValueError:
                    return None
        return None
            
    def get_bge_m3_embedding(self, text: str) -> Dict[str, Any]:
        """Get both dense and sparse embedding vectors from BGE-M3 model."""
        try:
            response = requests.post(
                f"http://{self.config['embed_host']}:{self.config['embed_port']}/api/embeddings",
                json={
                    "model": self.config['embed_model'],
                    "prompt": text,
                    "options": {
                        "embedding_only": True,
                        "truncate": True
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if 'embedding' in result:
                dense_embedding = result['embedding']
                sparse_embedding = self._create_sparse_embedding(text) if 'sparse_embedding' not in result else result['sparse_embedding']
                
                return {
                    'dense': dense_embedding,
                    'sparse': sparse_embedding
                }
            else:
                self.logger.error(f"Unexpected response format: {result}")
                return {'dense': [], 'sparse': {'indices': [], 'values': []}}
                
        except Exception as e:
            self.logger.error(f"Failed to get BGE-M3 embedding: {e}")
            return {'dense': [], 'sparse': {'indices': [], 'values': []}}
    
    def _create_sparse_embedding(self, text: str) -> Dict[str, List]:
        """Create a simple sparse embedding representation from text tokens."""
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        temp_vector = {}
        for word, count in word_counts.items():
            idx = hash(word) & (2**32 - 1)
            temp_vector[idx] = max(temp_vector.get(idx, 0.0), float(count))

        sorted_items = sorted(temp_vector.items())
        
        indices = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        return {'indices': indices, 'values': values}
            
    def get_existing_embeddings(self) -> set:
        """Get set of already embedded media_item_ids from PostgreSQL tracking table."""
        if not self.postgres_conn:
            self.logger.error("PostgreSQL connection not available.")
            return set()
        try:
            cursor = self.postgres_conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT media_item_id FROM embedding_status
                WHERE embedding_model = %s
            """, (self.config['embed_model'],))
            
            existing_ids = {str(row[0]) for row in cursor.fetchall()}
            cursor.close()
            
            self.logger.info(f"Found {len(existing_ids)} existing media item embeddings for model {self.config['embed_model']}")
            return existing_ids
            
        except Exception as e:
            self.logger.warning(f"Could not fetch existing embeddings: {e}")
            return set()

    def fetch_media_analysis_data(self, existing_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Fetch media analysis data from PostgreSQL, grouped by media item."""
        if not self.postgres_conn:
            self.logger.error("PostgreSQL connection not available.")
            return []
        
        base_query = """
        SELECT 
            mi.id as media_item_id,
            mi.title,
            mi.media_type,
            mi.release_date,
            mi.overview,
            mi.tagline,
            array_agg(DISTINCT g.name) as genres,
            json_object_agg(
                oa.analysis_type, 
                json_build_object(
                    'analysis_result', oa.analysis_result,
                    'model_used', oa.model_used,
                    'confidence_score', oa.confidence_score,
                    'created_at', oa.created_at
                )
            ) as analyses
        FROM media_items mi
        JOIN ollama_analysis oa ON mi.id = oa.media_item_id
        LEFT JOIN media_genres mg ON mi.id = mg.media_item_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE oa.analysis_result IS NOT NULL
        """
        
        params = []
        if existing_ids and len(existing_ids) > 0:
            placeholders = ','.join(['%s'] * len(existing_ids))
            base_query += f" AND mi.id::text NOT IN ({placeholders})"
            params.extend(list(existing_ids))
        
        query = base_query + """
        GROUP BY mi.id, mi.title, mi.media_type, mi.release_date, mi.overview, mi.tagline
        ORDER BY mi.date_added DESC
        """
        
        try:
            cursor = self.postgres_conn.cursor()
            cursor.execute(query, params)

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            cursor.close()
            self.logger.info(f"Fetched {len(results)} new media items for embedding")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            return []

    def create_embeddings_text(self, record: Dict[str, Any], analysis_type: str, analysis_data: Dict[str, Any]) -> str:
        """Create text for a specific analysis type."""
        base_info = f"Title: {record['title']}\nType: {record['media_type']}\n"
        
        if record['overview']:
            base_info += f"Overview: {record['overview']}\n"
        if record['tagline']:
            base_info += f"Tagline: {record['tagline']}\n"
        if record['genres'] and record['genres'][0]:
            base_info += f"Genres: {', '.join(record['genres'])}\n"

        analysis_result = analysis_data.get('analysis_result', {})
        
        if analysis_type == 'content_profile':
            return base_info + f"""
Content Profile:
Primary Themes: {', '.join(analysis_result.get('primary_themes', []))}
Mood Tags: {', '.join(analysis_result.get('mood_tags', []))}
Style Descriptors: {', '.join(analysis_result.get('style_descriptors', []))}
Target Audience: {analysis_result.get('target_audience', '')}
Key Elements: {', '.join(analysis_result.get('key_elements', []))}
Standout Features: {', '.join(analysis_result.get('standout_features', []))}
""".strip()
            
        elif analysis_type == 'mood_analysis':
            return base_info + f"""
Mood Analysis:
Overall Mood: {analysis_result.get('overall_mood', '')}
Pace: {analysis_result.get('pace', '')}
Energy Level: {analysis_result.get('energy_level', 0)}
Emotional Weight: {analysis_result.get('emotional_weight', '')}
""".strip()
            
        elif analysis_type == 'theme_analysis':
            return base_info + f"""
Theme Analysis:
Major Themes: {', '.join(analysis_result.get('major_themes', []))}
Minor Themes: {', '.join(analysis_result.get('minor_themes', []))}
Cultural Context: {analysis_result.get('cultural_context', '')}
""".strip()
            
        elif analysis_type == 'similarity_analysis':
            return base_info + f"""
Similarity Analysis:
Audience Overlap: {', '.join(analysis_result.get('audience_overlap', []))}
Distinctive Elements: {', '.join(analysis_result.get('distinctive_elements', []))}
Appeal Factors: {self._format_dict_or_list(analysis_result.get('appeal_factors', {}))}
""".strip()
            
        elif analysis_type == 'recommendation_profile':
            return base_info + f"""
Recommendation Profile:
Discovery Tags: {', '.join(analysis_result.get('discovery_tags', []))}
User Appeal Factors: {self._format_dict_or_list(analysis_result.get('user_appeal_factors', {}))}
Recommendation Contexts: {', '.join(analysis_result.get('recommendation_contexts', []))}
""".strip()
            
        else:
            return base_info + f"Analysis: {json.dumps(analysis_result, indent=2)}"

    def extract_metadata(self, record: Dict[str, Any], analysis_type: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for a specific analysis type."""
        analysis_result = analysis_data.get('analysis_result', {})
        
        created_at_val = self._parse_date(analysis_data.get('created_at'))
        common_meta = {
            "media_item_id": str(record['media_item_id']),
            "title": record['title'],
            "media_type": record['media_type'],
            "confidence_score": float(analysis_data.get('confidence_score', 0.0)),
            "model_used": analysis_data.get('model_used', ''),
            "created_at": created_at_val.isoformat() if created_at_val else None
        }

        if analysis_type == 'content_profile':
            return {
                **common_meta,
                "primary_themes": analysis_result.get('primary_themes', []),
                "mood_tags": analysis_result.get('mood_tags', []),
                "target_audience": analysis_result.get('target_audience', ''),
                "complexity_level": int(analysis_result.get('complexity_level', 0)) if str(analysis_result.get('complexity_level', '0')).isdigit() else 0,
                "emotional_intensity": int(analysis_result.get('emotional_intensity', 0)) if str(analysis_result.get('emotional_intensity', '0')).isdigit() else 0
            }
        return {**common_meta, **analysis_result}
        
    def update_embedding_status(self, media_item_ids: List[str]):
        """Update tracking table with successfully embedded media_item_ids."""
        if not media_item_ids:
            return
        if not self.postgres_conn:
            self.logger.error("PostgreSQL connection not available.")
            return
            
        try:
            with self.postgres_conn.cursor() as cursor:
                values = [(mid, self.config['embed_model']) for mid in media_item_ids]
                
                extras.execute_values(cursor, """
                    INSERT INTO embedding_status (media_item_id, embedding_model)
                    VALUES %s
                    ON CONFLICT (media_item_id, embedding_model) DO UPDATE SET
                        embedded_at = CURRENT_TIMESTAMP
                """, values)
            self.postgres_conn.commit()
            self.logger.info(f"Updated embedding status for {len(media_item_ids)} media items")
            
        except Exception as e:
            self.logger.error(f"Failed to update embedding status: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()

    def process_batch(self, records: List[Dict[str, Any]]) -> int:
        """Process a batch of media items and their analyses."""
        processed_count = 0
        processed_media_ids = []
        
        for record in records:
            media_item_id = str(record['media_item_id'])
            try:
                # 1. Create a single content embedding for the media item
                content_text = self.create_generic_content_text(record)
                content_embeddings = self.get_bge_m3_embedding(content_text)
                if not content_embeddings['dense']:
                    self.logger.warning(f"Failed to get content embedding for media item {media_item_id}")
                    continue

                content_meta = self.extract_content_metadata(record)
                content_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"content_{media_item_id}"))
                
                content_point = PointStruct(
                    id=content_id,
                    vector={"dense": content_embeddings['dense'], "sparse": models.SparseVector(**content_embeddings['sparse'])},
                    payload=content_meta
                )
                if self.qdrant_client:
                    self.qdrant_client.upsert(collection_name="media_content", points=[content_point], wait=True)

                # 2. Create embeddings for each analysis type
                analyses = record.get('analyses', {})
                for analysis_type, analysis_data in analyses.items():
                    if not analysis_data or not analysis_data.get('analysis_result'):
                        continue

                    analysis_text = self.create_embeddings_text(record, analysis_type, analysis_data)
                    analysis_embeddings = self.get_bge_m3_embedding(analysis_text)
                    if not analysis_embeddings['dense']:
                        self.logger.warning(f"Skipping {analysis_type} for {media_item_id} due to embedding failure.")
                        continue

                    analysis_meta = self.extract_metadata(record, analysis_type, analysis_data)
                    analysis_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"analysis_{media_item_id}_{analysis_type}"))
                    
                    analysis_point = PointStruct(
                        id=analysis_id,
                        vector={"dense": analysis_embeddings['dense'], "sparse": models.SparseVector(**analysis_embeddings['sparse'])},
                        payload=analysis_meta
                    )
                    
                    collection_name = f"analysis_{analysis_type}"
                    if self.qdrant_client:
                        self.qdrant_client.upsert(collection_name=collection_name, points=[analysis_point], wait=True)

                processed_count += 1
                processed_media_ids.append(media_item_id)
                self.logger.info(f"Successfully processed media item {media_item_id}")

            except Exception as e:
                self.logger.error(f"Failed to process media item {media_item_id}: {e}")
                continue
        
        if processed_media_ids:
            self.update_embedding_status(processed_media_ids)
            
        return processed_count

    def create_generic_content_text(self, record: Dict[str, Any]) -> str:
        """Create a generic text representation for the media content."""
        text = f"Title: {record['title']}\nType: {record['media_type']}\n"
        if record['overview']:
            text += f"Overview: {record['overview']}\n"
        if record['tagline']:
            text += f"Tagline: {record['tagline']}\n"
        if record['genres'] and record['genres'][0] is not None:
            text += f"Genres: {', '.join(record['genres'])}\n"
        return text.strip()

    def extract_content_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata for the media_content collection."""
        release_date_obj = self._parse_date(record.get('release_date'))
        release_year = release_date_obj.year if release_date_obj else None

        return {
            "media_item_id": str(record['media_item_id']),
            "title": record['title'],
            "media_type": record['media_type'],
            "release_year": release_year,
            "genres": [g for g in record['genres'] if g] if record.get('genres') else [],
            "overview": record.get('overview', ''),
            "tagline": record.get('tagline', '')
        }
        
    def run(self):
        """Main execution method."""
        self.logger.info("Starting BGE-M3 vector embedding process")
        
        self.connect_postgres()
        self.connect_qdrant()
        
        existing_ids = set() if self.config['force_reprocess'] else self.get_existing_embeddings()
        
        if self.config['force_reprocess']:
            self.logger.info("Force reprocessing enabled - will process all records")
        
        records = self.fetch_media_analysis_data(existing_ids)
        if not records:
            self.logger.info("No new data to process")
            return
            
        batch_size = self.config['batch_size']
        total_processed = 0
        total_batches = (len(records) + batch_size - 1) // batch_size
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_num = i // batch_size + 1
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            processed = self.process_batch(batch)
            total_processed += processed
            
        self.logger.info(f"BGE-M3 embedding process complete. Total records processed: {total_processed}")
        
        if self.postgres_conn:
            self.postgres_conn.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BGE-M3 Vector Embedding Script for Media Analysis")
    
    parser.add_argument('--db-host', help='PostgreSQL host')
    parser.add_argument('--db-port', type=int, help='PostgreSQL port')
    parser.add_argument('--db-user', help='PostgreSQL username')
    parser.add_argument('--db-password', help='PostgreSQL password')
    parser.add_argument('--db-name', help='PostgreSQL database name')
    
    parser.add_argument('--qdrant-host', help='Qdrant host')
    parser.add_argument('--qdrant-port', type=int, help='Qdrant port')
    
    parser.add_argument('--embed-host', help='Embedding service host')
    parser.add_argument('--embed-port', type=int, help='Embedding service port')
    parser.add_argument('--embed-model', help='Embedding model name')
    
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
        'embed_model': args.embed_model or os.getenv('EMBED_MODEL', 'bge-m3:latest'),
        'batch_size': args.batch_size or int(os.getenv('BATCH_SIZE', '5')),
        'force_reprocess': args.force_reprocess
    }
    
    return config


def main():
    """Main entry point."""
    try:
        config = get_config()
        extras.register_uuid()
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
