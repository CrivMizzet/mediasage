#!/usr/bin/env python3
"""
External API Integration Module
Integrates TMDB and TVDB APIs for additional metadata enrichment
Implements rate limiting, caching, and data merging with local analysis
"""

import argparse
import json
import os
import sys
import time
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    tmdb_api_key: str
    tvdb_api_key: str
    omdb_api_key: str
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tvdb_base_url: str = "https://api4.thetvdb.com/v4"
    omdb_base_url: str = "http://www.omdbapi.com"
    tmdb_rate_limit: int = 40  # requests per 10 seconds
    tvdb_rate_limit: int = 1000  # requests per day
    omdb_rate_limit: int = 1000  # requests per day
    cache_ttl_hours: int = 168  # 1 week

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_period: int, period_seconds: int = 10):
        self.calls_per_period = calls_per_period
        self.period_seconds = period_seconds
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make an API call"""
        now = time.time()
        # Remove calls older than the period
        self.calls = [call_time for call_time in self.calls if now - call_time < self.period_seconds]
        
        if len(self.calls) >= self.calls_per_period:
            # Wait until we can make another call
            sleep_time = self.period_seconds - (now - self.calls[0]) + 0.1
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class ExternalAPIManager:
    """Manages external API integrations with rate limiting and caching"""
    
    def __init__(self, api_config: APIConfig, db_config: DatabaseConfig):
        self.api_config = api_config
        self.db_config = db_config
        self.tmdb_limiter = RateLimiter(api_config.tmdb_rate_limit, 10)
        self.tvdb_limiter = RateLimiter(api_config.tvdb_rate_limit, 86400)  # 24 hours
        self.omdb_limiter = RateLimiter(api_config.omdb_rate_limit, 86400)  # 24 hours
        self.session = None
        self.db_connection = None
        self.tvdb_token = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        self.db_connection = psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.database,
            user=self.db_config.username,
            password=self.db_config.password,
            cursor_factory=RealDictCursor
        )
        
        # Authenticate with TVDB
        await self._authenticate_tvdb()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.db_connection:
            self.db_connection.close()
    
    async def _authenticate_tvdb(self):
        """Authenticate with TVDB API to get access token"""
        try:
            auth_data = {
                "apikey": self.api_config.tvdb_api_key
            }
            
            async with self.session.post(
                f"{self.api_config.tvdb_base_url}/login",
                json=auth_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.tvdb_token = result.get('data', {}).get('token')
                    logger.info("Successfully authenticated with TVDB")
                else:
                    logger.error(f"TVDB authentication failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error authenticating with TVDB: {e}")
    
    def _is_cache_valid(self, created_at: datetime, expires_at: Optional[datetime] = None) -> bool:
        """Check if cached data is still valid"""
        # Get current time in UTC
        now = datetime.now(timezone.utc)
        
        # Ensure created_at is timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        if expires_at:
            # Ensure expires_at is timezone-aware
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return now < expires_at
        
        # Calculate expiration based on created_at + TTL
        expiration = created_at + timedelta(hours=self.api_config.cache_ttl_hours)
        return now < expiration
    
    async def _get_cached_response(self, api_source: str, endpoint: str, params: Dict) -> Optional[Dict]:
        """Retrieve cached API response if available and valid"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT response_data, expires_at, created_at
                    FROM external_api_cache
                    WHERE api_source = %s AND endpoint = %s AND query_params = %s
                """, (api_source, endpoint, json.dumps(params, sort_keys=True)))
                
                result = cursor.fetchone()
                if result and self._is_cache_valid(result['created_at'], result['expires_at']):
                    logger.debug(f"Cache hit for {api_source}:{endpoint}")
                    return result['response_data']
                elif result:
                    # Expired cache entry, delete it
                    cursor.execute("""
                        DELETE FROM external_api_cache
                        WHERE api_source = %s AND endpoint = %s AND query_params = %s
                    """, (api_source, endpoint, json.dumps(params, sort_keys=True)))
                    self.db_connection.commit()
        except Exception as e:
            logger.error(f"Error retrieving cached response: {e}")
        
        return None
    
    async def _cache_response(self, api_source: str, endpoint: str, params: Dict, response_data: Dict):
        """Cache API response"""
        try:
            # Use timezone-aware datetime
            expires_at = datetime.now(timezone.utc) + timedelta(hours=self.api_config.cache_ttl_hours)
            
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO external_api_cache (api_source, endpoint, query_params, response_data, expires_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (api_source, endpoint, query_params) 
                    DO UPDATE SET response_data = EXCLUDED.response_data, 
                                  expires_at = EXCLUDED.expires_at,
                                  created_at = CURRENT_TIMESTAMP
                """, (api_source, endpoint, json.dumps(params, sort_keys=True), json.dumps(response_data), expires_at))
                self.db_connection.commit()
                logger.debug(f"Cached response for {api_source}:{endpoint}")
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _make_api_request(self, url: str, params: Dict, rate_limiter: RateLimiter, headers: Dict = None) -> Optional[Dict]:
        """Make an API request with rate limiting and retry logic"""
        await rate_limiter.acquire()
        
        try:
            # Log the actual request being made (without API keys for security)
            safe_params = {k: v for k, v in params.items() if 'key' not in k.lower()}
            logger.debug(f"Making API request to: {url} with params: {safe_params}")
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    # Log some basic info about the response
                    if 'results' in result:
                        logger.debug(f"API response: {result.get('total_results', 0)} total results, {len(result.get('results', []))} returned")
                    elif 'Response' in result:
                        logger.debug(f"OMDB response: {result.get('Response')} - {result.get('Error', 'Success')}")
                    elif 'data' in result:
                        logger.debug(f"TVDB response: Success")
                    return result
                elif response.status == 429:  # Rate limited
                    logger.warning(f"Rate limited by API: {url}")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return await self._make_api_request(url, params, rate_limiter, headers)
                else:
                    logger.error(f"API request failed: {response.status} - {url}")
                    try:
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")
                    except:
                        pass
                    return None
        except Exception as e:
            logger.error(f"Error making API request to {url}: {e}")
            return None
    
    async def get_tmdb_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get detailed movie information from TMDB"""
        endpoint = f"/movie/{movie_id}"
        params = {"api_key": self.api_config.tmdb_api_key, "append_to_response": "credits,keywords,reviews"}
        
        # Check cache first
        cached_response = await self._get_cached_response("tmdb", endpoint, params)
        if cached_response:
            return cached_response
        
        url = f"{self.api_config.tmdb_base_url}{endpoint}"
        response = await self._make_api_request(url, params, self.tmdb_limiter)
        
        if response:
            await self._cache_response("tmdb", endpoint, params, response)
        
        return response
    
    async def get_tmdb_tv_details(self, tv_id: int) -> Optional[Dict]:
        """Get detailed TV show information from TMDB"""
        endpoint = f"/tv/{tv_id}"
        params = {"api_key": self.api_config.tmdb_api_key, "append_to_response": "credits,keywords"}
        
        # Check cache first
        cached_response = await self._get_cached_response("tmdb", endpoint, params)
        if cached_response:
            return cached_response
        
        url = f"{self.api_config.tmdb_base_url}{endpoint}"
        response = await self._make_api_request(url, params, self.tmdb_limiter)
        
        if response:
            await self._cache_response("tmdb", endpoint, params, response)
        
        return response
    
    async def get_tvdb_episode_details(self, episode_id: int) -> Optional[Dict]:
        """Get detailed episode information from TVDB"""
        if not self.tvdb_token:
            logger.error("TVDB token not available")
            return None
        
        endpoint = f"/episodes/{episode_id}/extended"
        params = {}
        headers = {"Authorization": f"Bearer {self.tvdb_token}"}
        
        # Check cache first
        cached_response = await self._get_cached_response("tvdb", endpoint, params)
        if cached_response:
            return cached_response
        
        url = f"{self.api_config.tvdb_base_url}{endpoint}"
        response = await self._make_api_request(url, params, self.tvdb_limiter, headers)
        
        if response:
            await self._cache_response("tvdb", endpoint, params, response)
        
        return response
    
    async def get_tvdb_series_details(self, series_id: int) -> Optional[Dict]:
        """Get detailed series information from TVDB"""
        if not self.tvdb_token:
            logger.error("TVDB token not available")
            return None
        
        endpoint = f"/series/{series_id}/extended"
        params = {}
        headers = {"Authorization": f"Bearer {self.tvdb_token}"}
        
        # Check cache first
        cached_response = await self._get_cached_response("tvdb", endpoint, params)
        if cached_response:
            return cached_response
        
        url = f"{self.api_config.tvdb_base_url}{endpoint}"
        response = await self._make_api_request(url, params, self.tvdb_limiter, headers)
        
        if response:
            await self._cache_response("tvdb", endpoint, params, response)
        
        return response
    
    async def search_omdb(self, title: str, year: Optional[int] = None, media_type: Optional[str] = None) -> Optional[Dict]:
        """Search for media on OMDB"""
        endpoint = "/"
        params = {"apikey": self.api_config.omdb_api_key, "t": title}
        if year:
            params["y"] = year
        if media_type:
            params["type"] = media_type  # movie, series, episode
        
        # Check cache first
        cached_response = await self._get_cached_response("omdb", endpoint, params)
        if cached_response:
            return cached_response
        
        url = f"{self.api_config.omdb_base_url}{endpoint}"
        response = await self._make_api_request(url, params, self.omdb_limiter)
        
        if response:
            await self._cache_response("omdb", endpoint, params, response)
        
        return response
    
    async def enrich_media_item(self, media_item: Dict) -> Dict:
        """Enrich a media item with external API data using existing IDs"""
        enriched_data = {}
        
        title = media_item.get('title', '')
        media_type = media_item.get('media_type', '')
        tmdb_id = media_item.get('tmdb_id')
        tvdb_id = media_item.get('tvdb_id')
        release_date = media_item.get('release_date')
        year = None
        
        if release_date:
            try:
                year = datetime.strptime(str(release_date), '%Y-%m-%d').year
            except:
                pass
        
        logger.info(f"Enriching: '{title}' (TMDB ID: {tmdb_id}, TVDB ID: {tvdb_id}, Type: {media_type})")
        
        # Skip items without any external ID (folders)
        if not tmdb_id and not tvdb_id:
            logger.info(f"Skipping '{title}' - no external ID (likely a folder)")
            return enriched_data
        
        # Get TMDB data if TMDB ID exists
        if tmdb_id:
            try:
                tmdb_id = int(tmdb_id)
                if media_type.lower() in ['movie', 'film']:
                    tmdb_data = await self.get_tmdb_movie_details(tmdb_id)
                elif media_type.lower() in ['series', 'show', 'tv']:
                    tmdb_data = await self.get_tmdb_tv_details(tmdb_id)
                else:
                    # Try movie first, then TV if that fails
                    tmdb_data = await self.get_tmdb_movie_details(tmdb_id)
                    if not tmdb_data:
                        tmdb_data = await self.get_tmdb_tv_details(tmdb_id)
                
                if tmdb_data:
                    enriched_data['tmdb'] = tmdb_data
                    logger.info(f"Successfully enriched with TMDB data for ID {tmdb_id}")
                else:
                    logger.warning(f"No TMDB data found for ID {tmdb_id}")
            
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid TMDB ID '{tmdb_id}': {e}")
        
        # Get TVDB data if TVDB ID exists
        if tvdb_id:
            try:
                tvdb_id = int(tvdb_id)
                
                # Try episode first, then series
                tvdb_data = await self.get_tvdb_episode_details(tvdb_id)
                if not tvdb_data:
                    tvdb_data = await self.get_tvdb_series_details(tvdb_id)
                
                if tvdb_data:
                    enriched_data['tvdb'] = tvdb_data
                    logger.info(f"Successfully enriched with TVDB data for ID {tvdb_id}")
                else:
                    logger.warning(f"No TVDB data found for ID {tvdb_id}")
            
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid TVDB ID '{tvdb_id}': {e}")
        
        # For OMDB, we need to search by title
        # Use data from TMDB/TVDB to improve OMDB search accuracy
        omdb_data = None
        search_title = title
        omdb_type = None
        
        # Use official title from TMDB/TVDB if available
        if 'tmdb' in enriched_data:
            tmdb_info = enriched_data['tmdb']
            if 'title' in tmdb_info:  # Movie
                search_title = tmdb_info['title']
                omdb_type = 'movie'
                if not year and tmdb_info.get('release_date'):
                    try:
                        year = datetime.strptime(tmdb_info['release_date'], '%Y-%m-%d').year
                    except:
                        pass
            elif 'name' in tmdb_info:  # TV Show
                search_title = tmdb_info['name']
                omdb_type = 'series'
                if not year and tmdb_info.get('first_air_date'):
                    try:
                        year = datetime.strptime(tmdb_info['first_air_date'], '%Y-%m-%d').year
                    except:
                        pass
        
        elif 'tvdb' in enriched_data:
            tvdb_info = enriched_data['tvdb'].get('data', {})
            if tvdb_info.get('name'):
                search_title = tvdb_info['name']
                omdb_type = 'series' if media_type.lower() in ['episode', 'series', 'show', 'tv'] else None
                if not year and tvdb_info.get('aired'):
                    try:
                        year = datetime.strptime(tvdb_info['aired'], '%Y-%m-%d').year
                    except:
                        pass
        
        else:
            # Fallback to guessing type based on media_type
            if media_type.lower() in ['movie', 'film']:
                omdb_type = 'movie'
            elif media_type.lower() in ['series', 'show', 'tv', 'episode']:
                omdb_type = 'series'
        
        # Search OMDB
        if search_title:
            logger.debug(f"Searching OMDB for: '{search_title}' (year: {year}, type: {omdb_type})")
            omdb_data = await self.search_omdb(search_title, year, omdb_type)
            
            if omdb_data and omdb_data.get('Response') == 'True':
                enriched_data['omdb'] = omdb_data
                logger.info(f"Successfully enriched with OMDB data")
            else:
                # Try without year if that didn't work
                if year:
                    logger.debug(f"Retrying OMDB search without year")
                    omdb_data = await self.search_omdb(search_title, None, omdb_type)
                    if omdb_data and omdb_data.get('Response') == 'True':
                        enriched_data['omdb'] = omdb_data
                        logger.info(f"Successfully enriched with OMDB data (no year)")
                    else:
                        logger.warning(f"No OMDB data found for '{search_title}'")
                else:
                    logger.warning(f"No OMDB data found for '{search_title}'")
        
        return enriched_data
    
    async def store_external_data(self, media_item_id: str, external_data: Dict):
        """Store external API data in the database"""
        try:
            with self.db_connection.cursor() as cursor:
                for api_source, data in external_data.items():
                    # Extract appropriate external ID
                    external_id = ""
                    if api_source == 'tmdb':
                        external_id = str(data.get('id', ''))
                    elif api_source == 'tvdb':
                        external_id = str(data.get('data', {}).get('id', ''))
                    elif api_source == 'omdb':
                        external_id = data.get('imdbID', '')
                    
                    cursor.execute("""
                        INSERT INTO media_external_data (media_item_id, api_source, external_id, data_type, raw_data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (media_item_id, api_source, data_type)
                        DO UPDATE SET raw_data = EXCLUDED.raw_data, 
                                      external_id = EXCLUDED.external_id,
                                      last_updated = CURRENT_TIMESTAMP
                    """, (
                        media_item_id,
                        api_source,
                        external_id,
                        'enrichment',
                        json.dumps(data)
                    ))
                self.db_connection.commit()
                logger.info(f"Stored external data for media item {media_item_id}")
        except Exception as e:
            logger.error(f"Error storing external data: {e}")
            self.db_connection.rollback()

class ExternalAPIIntegrationService:
    """Main service for external API integration"""
    
    def __init__(self, api_config: APIConfig, db_config: DatabaseConfig):
        self.api_config = api_config
        self.db_config = db_config
    
    async def process_media_library(self, limit: Optional[int] = None, offset: int = 0):
        """Process the entire media library for external data enrichment"""
        async with ExternalAPIManager(self.api_config, self.db_config) as api_manager:
            # Get media items that need enrichment
            media_items = self._get_media_items_for_enrichment(limit, offset)
            
            logger.info(f"Processing {len(media_items)} media items for external data enrichment")
            
            for i, media_item in enumerate(media_items, 1):
                logger.info(f"Processing item {i}/{len(media_items)}: {media_item['title']}")
                
                try:
                    # Enrich the media item
                    external_data = await api_manager.enrich_media_item(media_item)
                    
                    if external_data:
                        # Store the enriched data
                        await api_manager.store_external_data(media_item['id'], external_data)
                        logger.info(f"Successfully enriched: {media_item['title']}")
                    else:
                        logger.warning(f"No external data found for: {media_item['title']}")
                
                except Exception as e:
                    logger.error(f"Error processing {media_item['title']}: {e}")
                    continue
                
                # Small delay to be respectful to APIs
                await asyncio.sleep(0.1)
    
    def _get_media_items_for_enrichment(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        """Get media items that need external data enrichment"""
        try:
            connection = psycopg2.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password,
                cursor_factory=RealDictCursor
            )
            
            with connection.cursor() as cursor:
                query = """
                    SELECT mi.id, mi.title, mi.original_title, mi.media_type, mi.release_date, 
                           mi.tmdb_id, mi.tvdb_id
                    FROM media_items mi
                    LEFT JOIN media_external_data med ON mi.id = med.media_item_id
                    WHERE (mi.tmdb_id IS NOT NULL OR mi.tvdb_id IS NOT NULL)
                    AND (med.media_item_id IS NULL OR med.last_updated < CURRENT_TIMESTAMP - INTERVAL '7 days')
                    ORDER BY mi.created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
                
                cursor.execute(query)
                return cursor.fetchall()
        
        except Exception as e:
            logger.error(f"Error fetching media items for enrichment: {e}")
            return []
        finally:
            if connection:
                connection.close()

def load_config() -> Tuple[APIConfig, DatabaseConfig]:
    """Load configuration from various sources"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="External API Integration for Media Analysis")
    parser.add_argument("--tmdb-api-key", help="TMDB API Key")
    parser.add_argument("--tvdb-api-key", help="TVDB API Key")
    parser.add_argument("--omdb-api-key", help="OMDB API Key")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--db-name", default="media_rec", help="Database name")
    parser.add_argument("--db-user", default="postgres", help="Database username")
    parser.add_argument("--db-password", help="Database password")
    parser.add_argument("--cache-ttl-hours", type=int, default=168, help="Cache TTL in hours")
    parser.add_argument("--limit", type=int, help="Limit number of items to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for processing")
    
    args = parser.parse_args()
    
    # Load from environment variables with CLI override
    tmdb_api_key = args.tmdb_api_key or os.getenv("TMDB_API_KEY", "")
    tvdb_api_key = args.tvdb_api_key or os.getenv("TVDB_API_KEY", "")
    omdb_api_key = args.omdb_api_key or os.getenv("OMDB_API_KEY", "")
    
    if not tmdb_api_key or not tvdb_api_key or not omdb_api_key:
        logger.error("TMDB, TVDB, and OMDB API keys are required")
        sys.exit(1)
    
    api_config = APIConfig(
        tmdb_api_key=tmdb_api_key,
        tvdb_api_key=tvdb_api_key,
        omdb_api_key=omdb_api_key,
        cache_ttl_hours=args.cache_ttl_hours
    )
    
    db_config = DatabaseConfig(
        host=args.db_host or os.getenv("DB_HOST", "localhost"),
        port=args.db_port or int(os.getenv("DB_PORT", "5432")),
        database=args.db_name or os.getenv("DB_NAME", "media_rec"),
        username=args.db_user or os.getenv("DB_USER", "postgres"),
        password=args.db_password or os.getenv("DB_PASSWORD", "")
    )
    
    return api_config, db_config, args

async def main():
    """Main execution function"""
    api_config, db_config, args = load_config()
    
    service = ExternalAPIIntegrationService(api_config, db_config)
    
    logger.info("Starting external API integration process...")
    await service.process_media_library(limit=args.limit, offset=args.offset)
    logger.info("External API integration process completed.")

if __name__ == "__main__":
    asyncio.run(main())
