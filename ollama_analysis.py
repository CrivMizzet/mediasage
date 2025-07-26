#!/usr/bin/env python3
"""
Enhanced Ollama Integration Module with Debugging Features

Test Command:
python ollama_analysis.py  --ollama-url "http://192.168.0.150:11434" --db-host "192.168.0.20" --db-user postgres --db-password 8g1k9ap2 --batch-size 10 --model Mjh1051702/youtube --analysis-type mood_analysis
"""

import json
import asyncio
import aiohttp
import logging
import os
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import time
import uuid
import re

# Database imports
import psycopg2
from psycopg2.extras import RealDictCursor


class AnalysisType(Enum):
    """Types of analysis Ollama can perform"""
    CONTENT_PROFILE = "content_profile"
    THEME_ANALYSIS = "theme_analysis" 
    MOOD_ANALYSIS = "mood_analysis"
    SIMILARITY_ANALYSIS = "similarity_analysis"
    RECOMMENDATION_PROFILE = "recommendation_profile"


@dataclass
class MediaItem:
    """Media item data structure"""
    id: str
    title: str
    media_type: str
    overview: str
    genres: List[str]
    release_date: Optional[str]
    runtime_minutes: Optional[int]
    content_rating: Optional[str]
    community_rating: Optional[float]
    tagline: Optional[str]


@dataclass
class AnalysisResult:
    """Analysis result data structure"""
    media_item_id: str
    analysis_type: str
    model_used: str
    prompt_version: str
    analysis_version: str
    analysis_result: Dict[str, Any]
    confidence_score: Optional[float]
    processing_time_ms: int
    raw_response: Optional[str] = None  # Added for debugging


class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, model: str, prompt: str, system: str = None) -> Dict[str, Any]: # type: ignore
        """Generate response from Ollama model"""
        if not self.session:
            raise RuntimeError("OllamaClient must be used as async context manager")
            
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Enable JSON mode
            "options": {
                "temperature": 0.1,  # Very low temperature for consistency
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
        
        if system:
            payload["system"] = system
            
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            logging.error(f"Ollama API error: {str(e)}")
            raise


class PromptManager:
    """Manages analysis prompts for different analysis types"""
    
    PROMPTS = {
        AnalysisType.CONTENT_PROFILE: {
            "system": """You are a media content analyst specializing in creating detailed profiles for movies and TV shows. Your analysis should be objective, comprehensive, and focused on helping users find similar content they might enjoy. You must respond with valid JSON only.""",
            
            "template": """Analyze the following {media_type} and provide a comprehensive content profile:

Title: {title}
Overview: {overview}
Genres: {genres}
Release Date: {release_date}
Runtime: {runtime_minutes} minutes
Content Rating: {content_rating}
Community Rating: {community_rating}/10
Tagline: {tagline}

You must respond with ONLY valid JSON in this exact format (no additional text):

{{
    "primary_themes": ["theme1", "theme2", "theme3"],
    "mood_tags": ["mood1", "mood2", "mood3"],
    "style_descriptors": ["style1", "style2", "style3"],
    "target_audience": "description of target audience",
    "complexity_level": 5,
    "emotional_intensity": 6,
    "recommended_viewing_context": "when/how to watch this",
    "content_warnings": ["warning1", "warning2"],
    "key_elements": ["element1", "element2", "element3"],
    "similar_vibes": ["similar_title1", "similar_title2"],
    "standout_features": ["feature1", "feature2"]
}}

Respond with JSON only."""
        },
        
        AnalysisType.THEME_ANALYSIS: {
            "system": """You are a thematic analysis expert who identifies and categorizes the deeper themes, messages, and narrative elements in media content. You must respond with valid JSON only.""",
            
            "template": """Perform a thematic analysis of this {media_type}:

Title: {title}
Overview: {overview}
Genres: {genres}

You must respond with ONLY valid JSON in this exact format (no additional text):

{{
    "major_themes": ["theme1", "theme2"],
    "minor_themes": ["theme3", "theme4"], 
    "narrative_structure": "description",
    "character_archetypes": ["archetype1", "archetype2"],
    "symbolic_elements": ["symbol1", "symbol2"],
    "cultural_context": "relevant cultural elements",
    "philosophical_questions": ["question1", "question2"],
    "emotional_journey": "description of emotional arc"
}}

Respond with JSON only."""
        },
        
        AnalysisType.MOOD_ANALYSIS: {
            "system": """You are a mood and atmosphere analyst who identifies the emotional tone, pacing, and viewing experience characteristics of media content. You must respond with valid JSON only.""",
            
            "template": """Analyze the mood and atmosphere of this {media_type}:

Title: {title}
Overview: {overview}
Genres: {genres}
Runtime: {runtime_minutes} minutes

You must respond with ONLY valid JSON in this exact format (no additional text):

{{
    "overall_mood": "primary mood descriptor",
    "mood_progression": "how mood changes throughout",
    "energy_level": 7,
    "tension_level": 5,
    "humor_level": 3,
    "darkness_level": 4,
    "pace": "moderate",
    "atmosphere_tags": ["atmospheric1", "atmospheric2"],
    "emotional_weight": "moderate",
    "viewer_experience": "what the viewing experience feels like"
}}

Respond with JSON only."""
        }
    }
    
    @classmethod
    def get_prompt(cls, analysis_type: AnalysisType, media_item: MediaItem) -> tuple[str, str]:
        """Get system prompt and formatted user prompt for analysis type"""
        if analysis_type not in cls.PROMPTS:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        prompt_config = cls.PROMPTS[analysis_type]
        system_prompt = prompt_config["system"]
        
        user_prompt = prompt_config["template"].format(
            media_type=media_item.media_type,
            title=media_item.title,
            overview=media_item.overview or "No overview available",
            genres=", ".join(media_item.genres) if media_item.genres else "Unknown",
            release_date=media_item.release_date or "Unknown",
            runtime_minutes=media_item.runtime_minutes or "Unknown",
            content_rating=media_item.content_rating or "Not Rated",
            community_rating=media_item.community_rating or "Not Rated",
            tagline=media_item.tagline or "No tagline"
        )
        
        return system_prompt, user_prompt


class DatabaseManager:
    """Handles database operations for analysis data"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params) # type: ignore
    
    def get_media_for_analysis(self, limit: int = 50, analysis_type: AnalysisType = AnalysisType.CONTENT_PROFILE) -> List[MediaItem]:
        """Get media items that need analysis"""
        query = """
        SELECT DISTINCT
            mi.id,
            mi.title,
            mi.media_type,
            mi.overview,
            mi.release_date::text,
            mi.runtime_minutes,
            mi.content_rating,
            mi.community_rating,
            mi.tagline,
            mi.date_added,
            COALESCE(
                ARRAY_AGG(g.name) FILTER (WHERE g.name IS NOT NULL), 
                ARRAY[]::varchar[]
            ) as genres
        FROM media_items mi
        LEFT JOIN media_genres mg ON mi.id = mg.media_item_id
        LEFT JOIN genres g ON mg.genre_id = g.id
        WHERE mi.id NOT IN (
            SELECT DISTINCT media_item_id 
            FROM ollama_analysis 
            WHERE analysis_type = %s
        )
        AND mi.is_available = true
        GROUP BY mi.id, mi.title, mi.media_type, mi.overview, 
                 mi.release_date, mi.runtime_minutes, mi.content_rating,
                 mi.community_rating, mi.tagline, mi.date_added
        ORDER BY mi.date_added DESC
        LIMIT %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (analysis_type.value, limit))
                rows = cur.fetchall()
                
                return [
                    MediaItem(
                        id=row['id'],
                        title=row['title'],
                        media_type=row['media_type'],
                        overview=row['overview'],
                        genres=row['genres'] or [],
                        release_date=row['release_date'],
                        runtime_minutes=row['runtime_minutes'],
                        content_rating=row['content_rating'],
                        community_rating=float(row['community_rating']) if row['community_rating'] else None,
                        tagline=row['tagline']
                    )
                    for row in rows
                ]
    
    def queue_analysis(self, media_item_id: str, analysis_type: AnalysisType, priority: int = 5) -> str:
        """Queue media item for analysis"""
        queue_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO analysis_queue (id, media_item_id, analysis_type, priority, status)
        VALUES (%s, %s, %s, %s, 'pending')
        ON CONFLICT (media_item_id, analysis_type) 
        DO UPDATE SET 
            priority = EXCLUDED.priority,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (queue_id, media_item_id, analysis_type.value, priority))
                result = cur.fetchone()
                conn.commit()
                return result[0] if result else queue_id
    
    def save_analysis_result(self, result: AnalysisResult) -> None:
        """Save analysis result to database"""
        query = """
        INSERT INTO ollama_analysis (
            id, media_item_id, analysis_type, model_used, prompt_version,
            analysis_version, analysis_result, confidence_score, processing_time_ms
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (media_item_id, analysis_type, model_used)
        DO UPDATE SET
            analysis_result = EXCLUDED.analysis_result,
            confidence_score = EXCLUDED.confidence_score,
            processing_time_ms = EXCLUDED.processing_time_ms,
            created_at = CURRENT_TIMESTAMP
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    str(uuid.uuid4()),
                    result.media_item_id,
                    result.analysis_type,
                    result.model_used,
                    result.prompt_version,
                    result.analysis_version,
                    json.dumps(result.analysis_result),
                    result.confidence_score,
                    result.processing_time_ms
                ))
                conn.commit()
    
    def update_queue_status(self, media_item_id: str, analysis_type: str, 
                           status: str, error_message: str = None) -> None: # type: ignore
        """Update analysis queue status"""
        query = """
        UPDATE analysis_queue 
        SET status = %s, error_message = %s, updated_at = CURRENT_TIMESTAMP
        WHERE media_item_id = %s AND analysis_type = %s
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (status, error_message, media_item_id, analysis_type))
                conn.commit()

    def save_debug_response(self, media_item_id: str, analysis_type: str, raw_response: str) -> None:
        """Save raw response for debugging (optional helper method)"""
        query = """
        INSERT INTO ollama_debug_responses (id, media_item_id, analysis_type, raw_response, created_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT DO NOTHING
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (str(uuid.uuid4()), media_item_id, analysis_type, raw_response))
                    conn.commit()
        except Exception as e:
            # If debug table doesn't exist, that's fine
            logging.debug(f"Could not save debug response: {e}")


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract JSON from response text, handling various formats"""
    
    # First, try to parse the entire response as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON within the response using regex
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Basic nested JSON
        r'\{.*\}',  # Any content between first { and last }
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found, return the raw response wrapped
    return {"raw_response": response_text.strip(), "parsing_error": True}


class MediaAnalyzer:
    """Main class for analyzing media content with Ollama"""
    
    def __init__(self, db_manager: DatabaseManager, ollama_url: str, default_model: str = None, debug_mode: bool = False): # type: ignore
        self.db = db_manager
        self.ollama_url = ollama_url
        self.default_model = default_model or 'llama3:latest'
        self.prompt_version = "1.1"  # Updated version
        self.analysis_version = "1.1"  # Updated version
        self.debug_mode = debug_mode
        
    async def analyze_media_item(self, media_item: MediaItem, 
                                analysis_type: AnalysisType = AnalysisType.CONTENT_PROFILE,
                                model: str = None) -> Optional[AnalysisResult]: # type: ignore
        """Analyze a single media item"""
        model_to_use = model or self.default_model
        start_time = time.time()
        
        try:
            # Update queue status
            self.db.update_queue_status(
                media_item.id, analysis_type.value, "processing"
            )
            
            # Get prompts
            system_prompt, user_prompt = PromptManager.get_prompt(analysis_type, media_item)
            
            if self.debug_mode:
                logging.info(f"System prompt for {media_item.title}: {system_prompt[:100]}...")
                logging.info(f"User prompt for {media_item.title}: {user_prompt[:200]}...")
            
            # Call Ollama
            async with OllamaClient(self.ollama_url) as client:
                response = await client.generate(
                    model=model_to_use,
                    prompt=user_prompt,
                    system=system_prompt
                )
            
            processing_time = int((time.time() - start_time) * 1000)
            raw_response = response.get('response', '')
            
            if self.debug_mode:
                logging.info(f"Raw response for {media_item.title}: {raw_response[:300]}...")
                # Optionally save to debug table
                self.db.save_debug_response(media_item.id, analysis_type.value, raw_response)
            
            # Parse response with improved extraction
            try:
                analysis_data = extract_json_from_response(raw_response)
                
                # Check if we got a parsing error
                if analysis_data.get("parsing_error"):
                    logging.warning(f"Could not parse JSON response for {media_item.title}")
                    confidence_score = 0.3  # Lower confidence for unparseable responses
                else:
                    confidence_score = self._calculate_confidence(response, analysis_data)
                    logging.info(f"Successfully parsed JSON for {media_item.title}")
                    
            except Exception as e:
                logging.error(f"Error processing response for {media_item.title}: {e}")
                analysis_data = {"raw_response": raw_response, "processing_error": str(e)}
                confidence_score = 0.2
            
            # Create result
            result = AnalysisResult(
                media_item_id=media_item.id,
                analysis_type=analysis_type.value,
                model_used=model_to_use,
                prompt_version=self.prompt_version,
                analysis_version=self.analysis_version,
                analysis_result=analysis_data,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                raw_response=raw_response if self.debug_mode else None
            )
            
            # Save to database
            self.db.save_analysis_result(result)
            self.db.update_queue_status(media_item.id, analysis_type.value, "completed")
            
            logging.info(f"Successfully analyzed {media_item.title} ({analysis_type.value}) - Confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to analyze {media_item.title}: {error_msg}")
            self.db.update_queue_status(
                media_item.id, analysis_type.value, "failed", error_msg
            )
            return None
    
    def _calculate_confidence(self, response: Dict, analysis_data: Dict) -> float:
        """Calculate confidence score based on response quality"""
        confidence = 0.5  # baseline
        
        # Check if we got structured data without parsing errors
        if isinstance(analysis_data, dict) and not analysis_data.get("parsing_error"):
            confidence += 0.2
        
        # Check if we have the expected fields for the analysis type
        expected_fields = {
            "mood_analysis": ["overall_mood", "energy_level", "tension_level"],
            "content_profile": ["primary_themes", "mood_tags", "style_descriptors"],
            "theme_analysis": ["major_themes", "minor_themes", "narrative_structure"]
        }
        
        # Get expected fields for current analysis (this would need to be passed in)
        # For now, just check for common fields
        common_expected = ["overall_mood", "primary_themes", "major_themes"]
        found_expected = sum(1 for field in common_expected if field in analysis_data)
        
        if found_expected > 0:
            confidence += 0.2 * (found_expected / len(common_expected))
        
        # Check response completeness
        if 'response' in response and len(response['response']) > 100:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    async def batch_analyze(self, limit: int = 10, 
                           analysis_type: AnalysisType = AnalysisType.CONTENT_PROFILE,
                           model: str = None) -> List[AnalysisResult]: # type: ignore
        """Analyze multiple media items in batch"""
        media_items = self.db.get_media_for_analysis(limit, analysis_type)
        
        if not media_items:
            logging.info("No media items found for analysis")
            return []
        
        logging.info(f"Starting batch analysis of {len(media_items)} items")
        results = []
        
        for i, media_item in enumerate(media_items, 1):
            logging.info(f"Processing item {i}/{len(media_items)}: {media_item.title}")
            
            # Queue the analysis
            self.db.queue_analysis(media_item.id, analysis_type)
            
            # Perform analysis
            result = await self.analyze_media_item(media_item, analysis_type, model)
            if result:
                results.append(result)
            
            # Small delay to avoid overwhelming Ollama
            await asyncio.sleep(1)
        
        logging.info(f"Completed batch analysis: {len(results)} successful")
        return results


def get_ollama_url(args_url: str = None) -> str: # type: ignore
    """Get Ollama URL from environment variable or argument"""
    if args_url:
        return args_url
    
    env_url = os.getenv('OLLAMA_URL')
    if env_url:
        return env_url
    
    return "http://localhost:11434"

def get_ollama_model(args_model: str = None) -> str: # type: ignore
    """Get Ollama model from environment variable or argument"""
    if args_model:
        return args_model
    
    env_model = os.getenv('OLLAMA_MODEL')
    if env_model:
        return env_model
    
    return "llama3:latest"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ollama Media Analysis Integration')
    parser.add_argument(
        '--ollama-url', 
        type=str,
        help='Ollama API URL (default: $OLLAMA_URL or http://localhost:11434)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=10,
        help='Number of items to analyze in batch (default: 10)'
    )
    parser.add_argument(
        '--analysis-type',
        choices=[t.value for t in AnalysisType],
        default=AnalysisType.CONTENT_PROFILE.value,
        help='Type of analysis to perform (default: content_profile)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Ollama model to use (default: $OLLAMA_MODEL or llama3:latest)'
    )
    parser.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='Database host (default: localhost)'
    )
    parser.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='Database port (default: 5432)'
    )
    parser.add_argument(
        '--db-name',
        type=str,
        default='media_rec',
        help='Database name (default: media_rec)'
    )
    parser.add_argument(
        '--db-user',
        type=str,
        required=True,
        help='Database username'
    )
    parser.add_argument(
        '--db-password',
        type=str,
        help='Database password (can also use DB_PASSWORD env var)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    parser.add_argument(
        '--test-single',
        type=str,
        help='Test analysis on a single media item by title'
    )
    
    return parser.parse_args()


async def test_single_item(analyzer: MediaAnalyzer, title_search: str, analysis_type: AnalysisType):
    """Test analysis on a single item by title search"""
    query = """
    SELECT DISTINCT
        mi.id, mi.title, mi.media_type, mi.overview, mi.release_date::text,
        mi.runtime_minutes, mi.content_rating, mi.community_rating, mi.tagline,
        COALESCE(ARRAY_AGG(g.name) FILTER (WHERE g.name IS NOT NULL), ARRAY[]::varchar[]) as genres
    FROM media_items mi
    LEFT JOIN media_genres mg ON mi.id = mg.media_item_id
    LEFT JOIN genres g ON mg.genre_id = g.id
    WHERE LOWER(mi.title) LIKE LOWER(%s)
    AND mi.is_available = true
    GROUP BY mi.id, mi.title, mi.media_type, mi.overview, mi.release_date,
             mi.runtime_minutes, mi.content_rating, mi.community_rating, mi.tagline
    LIMIT 1
    """
    
    with analyzer.db.get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (f"%{title_search}%",))
            row = cur.fetchone()
            
            if not row:
                logging.error(f"No media item found matching '{title_search}'")
                return
            
            media_item = MediaItem(
                id=row['id'],
                title=row['title'],
                media_type=row['media_type'],
                overview=row['overview'],
                genres=row['genres'] or [],
                release_date=row['release_date'],
                runtime_minutes=row['runtime_minutes'],
                content_rating=row['content_rating'],
                community_rating=float(row['community_rating']) if row['community_rating'] else None,
                tagline=row['tagline']
            )
            
            logging.info(f"Testing analysis on: {media_item.title}")
            result = await analyzer.analyze_media_item(media_item, analysis_type)
            
            if result:
                print(f"\nAnalysis Result for '{media_item.title}':")
                print(f"Confidence: {result.confidence_score:.2f}")
                print(f"Processing Time: {result.processing_time_ms}ms")
                print(f"Analysis Result: {json.dumps(result.analysis_result, indent=2)}")
                if result.raw_response:
                    print(f"\nRaw Response: {result.raw_response[:500]}...")


async def main():
    """Main function with enhanced debugging"""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get Ollama URL and model
    ollama_url = get_ollama_url(args.ollama_url)
    model = get_ollama_model(args.model)
    
    logging.info(f"Using Ollama URL: {ollama_url}")
    logging.info(f"Using model: {model}")
    
    # Database configuration
    db_config = {
        'host': args.db_host or os.getenv('POSTGRES_HOST'),
        'database': args.db_name or os.getenv('POSTGRES_DATABASE'),
        'user': args.db_user or os.getenv('POSTGRES_USERNAME'),
        'password': args.db_password or os.getenv('DB_PASSWORD'),
        'port': str(args.db_port) if args.db_port else os.getenv('DB_PORT', '5432')
    }
    
    # Validate required config
    if not db_config['password']:
        print("Error: Database password required via --db-password or DB_PASSWORD env var")
        return 1
    
    # Initialize components
    db_manager = DatabaseManager(db_config)
    analyzer = MediaAnalyzer(
        db_manager, 
        ollama_url=ollama_url, 
        default_model=model,
        debug_mode=args.debug
    )
    
    # Convert analysis type string to enum
    analysis_type = AnalysisType(args.analysis_type)
    
    try:
        # Test single item if requested
        if args.test_single:
            await test_single_item(analyzer, args.test_single, analysis_type)
            return 0
        
        # Analyze a batch of media items
        results = await analyzer.batch_analyze(
            limit=args.batch_size,
            analysis_type=analysis_type,
            model=model
        )
        
        print(f"\nAnalyzed {len(results)} media items using model: {model}")
        for result in results:
            print(f"- {result.media_item_id}: {result.analysis_type} "
                  f"(confidence: {result.confidence_score:.2f})")
        
        return 0
                  
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)