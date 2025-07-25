#!/usr/bin/env python3
"""
Jellyfin Library Fetcher for Media Recommendation System

This script fetches media library data from Jellyfin and populates the PostgreSQL database.
It supports incremental syncing and handles various media types (movies, TV shows, etc.).

NICK'S CONNECTION!!!!! python jellyfin_fetcher.py --db-host http://192.168.0.20 --db-username postgres --db-password 8g1k9ap2 --db-name media_rec --jellyfin-host http://192.168.0.20:8096 --incremental

Usage:
    python jellyfin_library_fetcher.py --jellyfin-host http://localhost:8096 --db-host localhost --db-username postgres --db-password mypass --db-name media_rec
    Or using environment variables:
    export JELLYFIN_HOST=http://localhost:8096
    export POSTGRES_HOST=localhost
    export POSTGRES_USERNAME=postgres  
    export POSTGRES_PASSWORD=mypass
    export POSTGRES_DATABASE=media_rec
    python jellyfin_library_fetcher.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('jellyfin_sync.log')
    ]
)
logger = logging.getLogger(__name__)

class JellyfinLibraryFetcher:
    """Fetches library data from Jellyfin and syncs it to PostgreSQL database."""
    
    def __init__(self, jellyfin_host: str, db_config: Dict[str, str], token_file: str = "jellyfin_token.json"):
        """
        Initialize the Jellyfin library fetcher.
        
        Args:
            jellyfin_host: Jellyfin server URL
            db_config: Database connection configuration
            token_file: Path to Jellyfin authentication token file
        """
        self.jellyfin_host = jellyfin_host.rstrip('/')
        self.db_config = db_config
        self.token_file = token_file
        self.auth_token = None
        self.user_id = None
        self.connection = None
        self.session = requests.Session()
        
        # Performance tracking
        self.stats = {
            'items_processed': 0,
            'items_added': 0,
            'items_updated': 0,
            'items_skipped': 0,
            'genres_added': 0,
            'people_added': 0,
            'errors': 0,
            'start_time': None
        }
    
    def load_auth_token(self) -> bool:
        """
        Load authentication token from file.
        
        Returns:
            True if token loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.token_file):
                logger.error(f"Token file '{self.token_file}' not found. Please run jellyfin authentication first.")
                return False
                
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
                
            # Handle both PascalCase and snake_case formats
            self.auth_token = (token_data.get('AccessToken') or 
                             token_data.get('access_token'))
            self.user_id = (token_data.get('UserId') or 
                          token_data.get('user_id'))
            
            if not self.auth_token or not self.user_id:
                logger.error("Invalid token file format. Missing access token or user ID.")
                logger.error("Expected keys: 'AccessToken'/'access_token' and 'UserId'/'user_id'")
                logger.error(f"Found keys: {list(token_data.keys())}")
                return False
                
            # Set up session headers
            self.session.headers.update({
                'X-Emby-Token': self.auth_token,
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })
            
            logger.info("Authentication token loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load authentication token: {e}")
            return False
    
    def connect_to_database(self) -> bool:
        """
        Connect to PostgreSQL database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config.get('port', 5432),
                user=self.db_config['username'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def test_jellyfin_connection(self) -> bool:
        """
        Test connection to Jellyfin server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.session.get(f"{self.jellyfin_host}/System/Info")
            response.raise_for_status()
            
            server_info = response.json()
            logger.info(f"Connected to Jellyfin server: {server_info.get('ServerName', 'Unknown')} "
                       f"(Version: {server_info.get('Version', 'Unknown')})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Jellyfin server: {e}")
            return False
    
    def get_libraries(self) -> List[Dict]:
        """
        Get all available libraries from Jellyfin.
        
        Returns:
            List of library dictionaries
        """
        try:
            response = self.session.get(f"{self.jellyfin_host}/Users/{self.user_id}/Views")
            response.raise_for_status()
            
            libraries = response.json().get('Items', [])
            logger.info(f"Found {len(libraries)} libraries")
            
            for lib in libraries:
                logger.info(f"  - {lib.get('Name')} ({lib.get('CollectionType', 'mixed')})")
            
            return libraries
            
        except Exception as e:
            logger.error(f"Failed to get libraries: {e}")
            return []
    
    def get_library_items(self, library_id: str, start_index: int = 0, limit: int = 1000) -> Tuple[List[Dict], int]:
        """
        Get items from a specific library with pagination.
        
        Args:
            library_id: Library ID to fetch from
            start_index: Starting index for pagination
            limit: Maximum number of items to fetch
            
        Returns:
            Tuple of (items_list, total_record_count)
        """
        try:
            params = {
                'ParentId': library_id,
                'UserId': self.user_id,
                'StartIndex': start_index,
                'Limit': limit,
                'Recursive': 'true',
                'Fields': 'BasicSyncInfo,CanDelete,Container,PrimaryImageAspectRatio,ProductionYear,Status,EndDate,DateModified,DateCreated,Etag',
                'IncludeItemTypes': 'Movie,Series,Season,Episode,MusicVideo,Video'
            }
            
            response = self.session.get(f"{self.jellyfin_host}/Users/{self.user_id}/Items", params=params)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('Items', [])
            total_count = data.get('TotalRecordCount', 0)
            
            return items, total_count
            
        except Exception as e:
            logger.error(f"Failed to get library items: {e}")
            return [], 0
    
    def check_etag_column_exists(self, cursor) -> bool:
        """
        Check if the jellyfin_etag column exists in the media_items table.
        
        Args:
            cursor: Database cursor
            
        Returns:
            True if column exists, False otherwise
        """
        try:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'media_items' AND column_name = 'jellyfin_etag'
            """)
            return cursor.fetchone() is not None
        except Exception:
            return False
    
    def should_skip_item(self, cursor, jellyfin_id: str, item_data: Dict, force_refresh_hours: int = 24, has_etag_column: bool = False) -> bool:
        """
        Determine if an item should be skipped based on last analysis time and modification date.
        
        Args:
            cursor: Database cursor
            jellyfin_id: Jellyfin item ID
            item_data: Item data from Jellyfin
            force_refresh_hours: Hours after which to force refresh regardless of modification
            has_etag_column: Whether the jellyfin_etag column exists
            
        Returns:
            True if item should be skipped, False if it needs processing
        """
        try:
            # Build query based on available columns
            if has_etag_column:
                query = "SELECT last_analyzed, date_modified, jellyfin_etag FROM media_items WHERE jellyfin_id = %s"
            else:
                query = "SELECT last_analyzed, date_modified FROM media_items WHERE jellyfin_id = %s"
            
            cursor.execute(query, (jellyfin_id,))
            existing = cursor.fetchone()
            
            if not existing:
                # New item, don't skip
                return False
            
            # Parse Jellyfin modification date
            jellyfin_modified = self.parse_jellyfin_datetime(item_data.get('DateModified'))
            jellyfin_etag = item_data.get('Etag')
            
            # If we have etag support and an etag match, skip unless forced refresh is needed
            if has_etag_column and jellyfin_etag and existing.get('jellyfin_etag') == jellyfin_etag:
                if existing['last_analyzed']:
                    hours_since_analysis = (datetime.now(timezone.utc) - existing['last_analyzed']).total_seconds() / 3600
                    if hours_since_analysis < force_refresh_hours:
                        logger.debug(f"Skipping {item_data.get('Name', 'Unknown')} - etag match and recent analysis")
                        return True
            
            # If Jellyfin modification date is available, compare it
            if jellyfin_modified and existing['date_modified']:
                # Convert existing date_modified to timezone-aware datetime if it isn't already
                db_modified = existing['date_modified']
                if db_modified.tzinfo is None:
                    db_modified = db_modified.replace(tzinfo=timezone.utc)
                
                # If Jellyfin item hasn't been modified since our last analysis, skip it
                if existing['last_analyzed'] and jellyfin_modified <= db_modified:
                    hours_since_analysis = (datetime.now(timezone.utc) - existing['last_analyzed']).total_seconds() / 3600
                    if hours_since_analysis < force_refresh_hours:
                        logger.debug(f"Skipping {item_data.get('Name', 'Unknown')} - no modifications since last analysis")
                        return True
            
            # If we have a recent analysis and no clear indication of changes, skip
            if existing['last_analyzed']:
                hours_since_analysis = (datetime.now(timezone.utc) - existing['last_analyzed']).total_seconds() / 3600
                if hours_since_analysis < force_refresh_hours and not jellyfin_modified:
                    logger.debug(f"Skipping {item_data.get('Name', 'Unknown')} - recent analysis and no modification date")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if item should be skipped: {e}")
            return False  # On error, don't skip to be safe
    
    def parse_jellyfin_datetime(self, date_string: Optional[str]) -> Optional[datetime]:
        """
        Parse Jellyfin datetime string to timezone-aware datetime object.
        
        Args:
            date_string: Date string from Jellyfin
            
        Returns:
            Timezone-aware datetime object or None
        """
        if not date_string:
            return None
            
        try:
            # Handle various datetime formats Jellyfin might return
            # Remove the extra zeros from microseconds if present
            cleaned_date = date_string
            
            # Handle the .0000000Z format by truncating to 6 digits for microseconds
            if '.' in cleaned_date and cleaned_date.endswith('Z'):
                parts = cleaned_date[:-1].split('.')  # Remove Z and split on .
                if len(parts) == 2 and len(parts[1]) > 6:
                    # Truncate microseconds to 6 digits
                    parts[1] = parts[1][:6]
                    cleaned_date = parts[0] + '.' + parts[1] + 'Z'
            
            # Try different datetime formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',  # With microseconds
                '%Y-%m-%dT%H:%M:%SZ',     # Without microseconds
                '%Y-%m-%dT%H:%M:%S'       # Without Z
            ]
            
            for fmt in formats:
                try:
                    if cleaned_date.endswith('Z') and fmt.endswith('Z'):
                        dt = datetime.strptime(cleaned_date, fmt)
                        return dt.replace(tzinfo=timezone.utc)
                    elif not cleaned_date.endswith('Z') and not fmt.endswith('Z'):
                        dt = datetime.strptime(cleaned_date, fmt)
                        return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse datetime: {date_string}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing datetime {date_string}: {e}")
            return None
    
    def get_item_details(self, item_id: str) -> Optional[Dict]:
        """
        Get detailed information for a specific item.
        
        Args:
            item_id: Jellyfin item ID
            
        Returns:
            Item details dictionary or None if failed
        """
        try:
            params = {
                'UserId': self.user_id,
                'Fields': 'People,Genres,Studios,Taglines,Overview,CommunityRating,CriticRating,OfficialRating,ProductionYear,PremiereDate,RunTimeTicks,ProviderIds,MediaSources,DateModified,DateCreated,Etag'
            }
            
            response = self.session.get(f"{self.jellyfin_host}/Users/{self.user_id}/Items/{item_id}", params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get item details for {item_id}: {e}")
            return None
    
    def get_user_data(self, item_id: str) -> Optional[Dict]:
        """
        Get user-specific data for an item (play count, rating, etc.).
        
        Args:
            item_id: Jellyfin item ID
            
        Returns:
            User data dictionary or None if failed
        """
        try:
            response = self.session.get(f"{self.jellyfin_host}/Users/{self.user_id}/Items/{item_id}/UserData")
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get user data for {item_id}: {e}")
            return None
    
    def map_media_type(self, jellyfin_type: str) -> str:
        """
        Map Jellyfin media type to our database enum.
        
        Args:
            jellyfin_type: Jellyfin media type
            
        Returns:
            Database-compatible media type
        """
        type_mapping = {
            'Movie': 'movie',
            'Series': 'series',
            'Season': 'season',
            'Episode': 'episode',
            'MusicVideo': 'music',
            'Video': 'movie',  # Default videos to movie type
            'Audio': 'music',
            'Book': 'book'
        }
        
        return type_mapping.get(jellyfin_type, 'movie')
    
    def ensure_genre_exists(self, cursor, genre_name: str, jellyfin_id: str = None) -> int:
        """
        Ensure a genre exists in the database and return its ID.
        
        Args:
            cursor: Database cursor
            genre_name: Genre name
            jellyfin_id: Jellyfin genre ID (optional)
            
        Returns:
            Genre ID
        """
        try:
            # Try to find existing genre
            cursor.execute(
                "SELECT id FROM genres WHERE name = %s",
                (genre_name,)
            )
            result = cursor.fetchone()
            
            if result:
                return result['id']
            
            # Insert new genre
            cursor.execute(
                "INSERT INTO genres (name, jellyfin_id) VALUES (%s, %s) RETURNING id",
                (genre_name, jellyfin_id)
            )
            
            genre_id = cursor.fetchone()['id']
            self.stats['genres_added'] += 1
            logger.debug(f"Added genre: {genre_name}")
            
            return genre_id
            
        except Exception as e:
            logger.error(f"Failed to ensure genre exists: {e}")
            raise
    
    def ensure_person_exists(self, cursor, person_data: Dict) -> int:
        """
        Ensure a person exists in the database and return their ID.
        
        Args:
            cursor: Database cursor
            person_data: Person information from Jellyfin
            
        Returns:
            Person ID
        """
        try:
            name = person_data.get('Name', '')
            jellyfin_id = person_data.get('Id')
            
            if not name:
                raise ValueError("Person name is required")
            
            # Try to find existing person
            cursor.execute(
                "SELECT id FROM people WHERE name = %s AND (jellyfin_id = %s OR jellyfin_id IS NULL)",
                (name, jellyfin_id)
            )
            result = cursor.fetchone()
            
            if result:
                return result['id']
            
            # Insert new person
            cursor.execute("""
                INSERT INTO people (name, jellyfin_id, birth_date, biography, image_url)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                name,
                jellyfin_id,
                person_data.get('PremiereDate'),  # Some APIs use this for birth date
                person_data.get('Overview'),
                person_data.get('ImageTags', {}).get('Primary')  # Could be used to construct image URL
            ))
            
            person_id = cursor.fetchone()['id']
            self.stats['people_added'] += 1
            logger.debug(f"Added person: {name}")
            
            return person_id
            
        except Exception as e:
            logger.error(f"Failed to ensure person exists: {e}")
            raise
    
    def ensure_studio_exists(self, cursor, studio_name: str, jellyfin_id: str = None) -> int:
        """
        Ensure a studio exists in the database and return its ID.
        
        Args:
            cursor: Database cursor
            studio_name: Studio name
            jellyfin_id: Jellyfin studio ID (optional)
            
        Returns:
            Studio ID
        """
        try:
            # Try to find existing studio
            cursor.execute(
                "SELECT id FROM studios WHERE name = %s",
                (studio_name,)
            )
            result = cursor.fetchone()
            
            if result:
                return result['id']
            
            # Insert new studio
            cursor.execute(
                "INSERT INTO studios (name, jellyfin_id) VALUES (%s, %s) RETURNING id",
                (studio_name, jellyfin_id)
            )
            
            return cursor.fetchone()['id']
            
        except Exception as e:
            logger.error(f"Failed to ensure studio exists: {e}")
            raise
    
    def parse_runtime(self, runtime_ticks: Optional[int]) -> Optional[int]:
        """
        Convert Jellyfin runtime ticks to minutes.
        
        Args:
            runtime_ticks: Runtime in ticks (100 nanoseconds)
            
        Returns:
            Runtime in minutes or None
        """
        if runtime_ticks is None:
            return None
        
        # Convert ticks to minutes (1 tick = 100 nanoseconds, 1 minute = 60 seconds)
        return int(runtime_ticks / 10_000_000 / 60)
    
    def parse_date(self, date_string: Optional[str]) -> Optional[str]:
        """
        Parse Jellyfin date string to database format.
        
        Args:
            date_string: Date string from Jellyfin
            
        Returns:
            Formatted date string or None
        """
        if not date_string:
            return None
            
        try:
            # Handle various date formats Jellyfin might return
            for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(date_string[:19] + 'Z' if 'T' in date_string else date_string, 
                                         fmt if fmt.endswith('Z') else '%Y-%m-%d')
                    return dt.date().isoformat()
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_string}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date {date_string}: {e}")
            return None
    
    def get_file_info(self, media_sources: List[Dict]) -> Tuple[Optional[str], Optional[int]]:
        """
        Extract file path and size from media sources.
        
        Args:
            media_sources: List of media source dictionaries
            
        Returns:
            Tuple of (file_path, file_size_bytes)
        """
        if not media_sources:
            return None, None
        
        # Use the first media source
        source = media_sources[0]
        file_path = source.get('Path')
        file_size = source.get('Size')
        
        return file_path, file_size
    
    def insert_or_update_media_item(self, cursor, item_data: Dict, detailed_data: Dict, has_etag_column: bool = False) -> Tuple[str, bool]:
        """
        Insert or update a media item in the database.
        
        Args:
            cursor: Database cursor
            item_data: Basic item data from library listing
            detailed_data: Detailed item data from individual API call
            has_etag_column: Whether the jellyfin_etag column exists
            
        Returns:
            Tuple of (media_item_id, is_new_item)
        """
        try:
            jellyfin_id = item_data['Id']
            current_time = datetime.now(timezone.utc)
            
            # Check if item already exists
            cursor.execute(
                "SELECT id FROM media_items WHERE jellyfin_id = %s",
                (jellyfin_id,)
            )
            existing = cursor.fetchone()
            
            # Prepare media item data
            provider_ids = detailed_data.get('ProviderIds', {})
            media_sources = detailed_data.get('MediaSources', [])
            file_path, file_size = self.get_file_info(media_sources)
            
            # Parse Jellyfin modification date
            jellyfin_modified = self.parse_jellyfin_datetime(detailed_data.get('DateModified'))
            jellyfin_created = self.parse_jellyfin_datetime(detailed_data.get('DateCreated'))
            
            media_data = {
                'jellyfin_id': jellyfin_id,
                'title': detailed_data.get('Name', item_data.get('Name', '')),
                'original_title': detailed_data.get('OriginalTitle'),
                'media_type': self.map_media_type(item_data.get('Type', 'Movie')),
                'release_date': self.parse_date(detailed_data.get('PremiereDate')),
                'runtime_minutes': self.parse_runtime(detailed_data.get('RunTimeTicks')),
                'overview': detailed_data.get('Overview'),
                'tagline': detailed_data.get('Taglines', [None])[0] if detailed_data.get('Taglines') else None,
                'content_rating': detailed_data.get('OfficialRating'),
                'community_rating': detailed_data.get('CommunityRating'),
                'critic_rating': detailed_data.get('CriticRating'),
                'imdb_id': provider_ids.get('Imdb'),
                'tmdb_id': int(provider_ids['Tmdb']) if provider_ids.get('Tmdb') and provider_ids['Tmdb'].isdigit() else None,
                'tvdb_id': int(provider_ids['Tvdb']) if provider_ids.get('Tvdb') and provider_ids['Tvdb'].isdigit() else None,
                'file_path': file_path,
                'file_size_bytes': file_size,
                'is_available': True,
                'date_modified': jellyfin_modified or current_time,
                'last_analyzed': current_time  # Set the last_analyzed timestamp
            }
            
            # Add etag only if column exists
            if has_etag_column:
                media_data['jellyfin_etag'] = detailed_data.get('Etag')
            
            if existing:
                # Update existing item
                media_item_id = existing['id']
                
                update_fields = []
                update_values = []
                
                for field, value in media_data.items():
                    if field != 'jellyfin_id':  # Don't update the primary key
                        update_fields.append(f"{field} = %s")
                        update_values.append(value)
                
                update_values.append(media_item_id)
                
                cursor.execute(f"""
                    UPDATE media_items 
                    SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, update_values)
                
                is_new = False
                
            else:
                # Insert new item
                media_item_id = str(uuid.uuid4())
                media_data['id'] = media_item_id
                
                # Set date_added for new items
                if jellyfin_created:
                    media_data['date_added'] = jellyfin_created
                
                fields = list(media_data.keys())
                placeholders = ', '.join(['%s'] * len(fields))
                
                cursor.execute(f"""
                    INSERT INTO media_items ({', '.join(fields)})
                    VALUES ({placeholders})
                """, list(media_data.values()))
                
                is_new = True
                self.stats['items_added'] += 1
            
            if not is_new:
                self.stats['items_updated'] += 1
            
            return media_item_id, is_new
            
        except Exception as e:
            logger.error(f"Failed to insert/update media item {jellyfin_id}: {e}")
            raise
    
    def sync_media_relationships(self, cursor, media_item_id: str, detailed_data: Dict):
        """
        Sync genres, people, and studios for a media item.
        
        Args:
            cursor: Database cursor
            media_item_id: Media item UUID
            detailed_data: Detailed item data from Jellyfin
        """
        try:
            # Clear existing relationships
            cursor.execute("DELETE FROM media_genres WHERE media_item_id = %s", (media_item_id,))
            cursor.execute("DELETE FROM media_credits WHERE media_item_id = %s", (media_item_id,))
            cursor.execute("DELETE FROM media_studios WHERE media_item_id = %s", (media_item_id,))
            
            # Sync genres
            genres = detailed_data.get('Genres', [])
            for genre_name in genres:
                genre_id = self.ensure_genre_exists(cursor, genre_name)
                cursor.execute(
                    "INSERT INTO media_genres (media_item_id, genre_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (media_item_id, genre_id)
                )
            
            # Sync people and credits
            people = detailed_data.get('People', [])
            for person_data in people:
                person_id = self.ensure_person_exists(cursor, person_data)
                
                role_type = person_data.get('Type', '').lower()
                if role_type not in ['actor', 'director', 'writer', 'producer', 'composer']:
                    role_type = 'other'
                
                cursor.execute("""
                    INSERT INTO media_credits (media_item_id, person_id, role_type, character_name, sort_order)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    media_item_id,
                    person_id,
                    role_type,
                    person_data.get('Role'),  # Character name for actors
                    person_data.get('SortOrder', 0)
                ))
            
            # Sync studios
            studios = detailed_data.get('Studios', [])
            for studio_data in studios:
                studio_name = studio_data.get('Name')
                if studio_name:
                    studio_id = self.ensure_studio_exists(cursor, studio_name, studio_data.get('Id'))
                    cursor.execute(
                        "INSERT INTO media_studios (media_item_id, studio_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (media_item_id, studio_id)
                    )
            
        except Exception as e:
            logger.error(f"Failed to sync relationships for {media_item_id}: {e}")
            raise
    
    def sync_user_activity(self, cursor, media_item_id: str, jellyfin_item_id: str):
        """
        Sync user activity data for a media item.
        
        Args:
            cursor: Database cursor
            media_item_id: Media item UUID
            jellyfin_item_id: Jellyfin item ID
        """
        try:
            # Get user data from Jellyfin
            user_data = self.get_user_data(jellyfin_item_id)
            if not user_data:
                return
            
            # Ensure user exists
            cursor.execute(
                "SELECT id FROM users WHERE jellyfin_user_id = %s",
                (self.user_id,)
            )
            user_record = cursor.fetchone()
            
            if not user_record:
                # Create user record
                cursor.execute("""
                    INSERT INTO users (jellyfin_user_id, username, display_name)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (self.user_id, 'primary_user', 'Primary User'))
                db_user_id = cursor.fetchone()['id']
            else:
                db_user_id = user_record['id']
            
            # Sync watch status
            if user_data.get('Played'):
                cursor.execute("""
                    INSERT INTO user_activity (user_id, media_item_id, activity_type, watch_progress_percent, watch_count, last_watched)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, media_item_id, activity_type)
                    DO UPDATE SET 
                        watch_progress_percent = EXCLUDED.watch_progress_percent,
                        watch_count = EXCLUDED.watch_count,
                        last_watched = EXCLUDED.last_watched,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    db_user_id,
                    media_item_id,
                    'watched',
                    user_data.get('PlayedPercentage', 100),
                    user_data.get('PlayCount', 1),
                    datetime.now(timezone.utc) if user_data.get('LastPlayedDate') else None
                ))
            
            # Sync user rating
            if user_data.get('UserRating') is not None:
                cursor.execute("""
                    INSERT INTO user_activity (user_id, media_item_id, activity_type, rating)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, media_item_id, activity_type)
                    DO UPDATE SET 
                        rating = EXCLUDED.rating,
                        updated_at = CURRENT_TIMESTAMP
                """, (db_user_id, media_item_id, 'rated', user_data['UserRating']))
            
            # Sync favorites
            if user_data.get('IsFavorite'):
                cursor.execute("""
                    INSERT INTO user_activity (user_id, media_item_id, activity_type)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, media_item_id, activity_type) DO NOTHING
                """, (db_user_id, media_item_id, 'favorited'))
            
        except Exception as e:
            logger.error(f"Failed to sync user activity for {media_item_id}: {e}")
            # Don't re-raise as this is not critical for the sync
    
    def record_sync_operation(self, cursor, operation_type: str, status: str = 'running') -> str:
        """
        Record a sync operation in the database.
        
        Args:
            cursor: Database cursor
            operation_type: Type of sync operation
            status: Operation status
            
        Returns:
            Sync operation ID
        """
        try:
            sync_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO sync_operations (id, operation_type, status, items_total)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (sync_id, operation_type, status, 0))
            
            return sync_id
            
        except Exception as e:
            logger.error(f"Failed to record sync operation: {e}")
            return str(uuid.uuid4())  # Return dummy ID to continue
    
    def update_sync_operation(self, cursor, sync_id: str, status: str, error_message: str = None):
        """
        Update sync operation status.
        
        Args:
            cursor: Database cursor
            sync_id: Sync operation ID
            status: New status
            error_message: Error message if failed
        """
        try:
            cursor.execute("""
                UPDATE sync_operations 
                SET status = %s, 
                    completed_at = CURRENT_TIMESTAMP,
                    items_processed = %s,
                    items_total = %s,
                    error_message = %s,
                    metadata = %s
                WHERE id = %s
            """, (
                status,
                self.stats['items_processed'],
                self.stats['items_processed'],  # We'll update this with actual total if known
                error_message,
                json.dumps(self.stats),
                sync_id
            ))
            
        except Exception as e:
            logger.error(f"Failed to update sync operation: {e}")
    
    def sync_library(self, library_filter: List[str] = None, incremental: bool = False, force_refresh_hours: int = 24) -> bool:
        """
        Sync all or specified libraries from Jellyfin.
        
        Args:
            library_filter: List of library names to sync (None for all)
            incremental: Whether to perform incremental sync based on modification dates
            force_refresh_hours: Hours after which to force refresh regardless of modification
            
        Returns:
            True if sync completed successfully, False otherwise
        """
        self.stats['start_time'] = time.time()
        cursor = self.connection.cursor()
        
        try:
            # Check if jellyfin_etag column exists
            has_etag_column = self.check_etag_column_exists(cursor)
            if not has_etag_column:
                logger.warning("jellyfin_etag column not found - ETag optimization disabled. Consider running: ALTER TABLE media_items ADD COLUMN jellyfin_etag VARCHAR(255);")
            
            # Record sync operation
            operation_type = 'incremental_sync' if incremental else 'full_sync'
            sync_id = self.record_sync_operation(cursor, operation_type)
            
            # Get libraries
            libraries = self.get_libraries()
            if not libraries:
                logger.error("No libraries found")
                self.update_sync_operation(cursor, sync_id, 'failed', "No libraries found")
                return False
            
            # Filter libraries if specified
            if library_filter:
                libraries = [lib for lib in libraries if lib.get('Name') in library_filter]
                logger.info(f"Filtering to libraries: {library_filter}")
            
            total_items = 0
            
            # Process each library
            for library in libraries:
                library_name = library.get('Name', 'Unknown')
                library_id = library.get('Id')
                
                if not library_id:
                    logger.warning(f"Skipping library '{library_name}' - no ID")
                    continue
                
                logger.info(f"Processing library: {library_name}")
                
                # Get all items from this library with pagination
                start_index = 0
                limit = 100
                
                while True:
                    items, total_count = self.get_library_items(library_id, start_index, limit)
                    
                    if not items:
                        break
                    
                    logger.info(f"Processing batch {len(items)} items from {library_name} "
                               f"({start_index + 1}-{start_index + len(items)} of {total_count})")
                    
                    # Process each item
                    for item in items:
                        try:
                            self.stats['items_processed'] += 1
                            
                            # Check if we should skip this item
                            if incremental and self.should_skip_item(cursor, item['Id'], item, force_refresh_hours, has_etag_column):
                                self.stats['items_skipped'] += 1
                                continue
                            
                            # Get detailed item information
                            detailed_data = self.get_item_details(item['Id'])
                            if not detailed_data:
                                logger.warning(f"Could not get details for {item.get('Name', 'Unknown')}")
                                self.stats['errors'] += 1
                                continue
                            
                            # Insert/update media item
                            media_item_id, is_new = self.insert_or_update_media_item(cursor, item, detailed_data, has_etag_column)
                            
                            # Sync relationships (genres, people, studios)
                            self.sync_media_relationships(cursor, media_item_id, detailed_data)
                            
                            # Sync user activity
                            self.sync_user_activity(cursor, media_item_id, item['Id'])
                            
                            # Log progress
                            if self.stats['items_processed'] % 50 == 0:
                                logger.info(f"Processed {self.stats['items_processed']} items "
                                           f"(skipped: {self.stats['items_skipped']}) so far...")
                            
                        except Exception as e:
                            logger.error(f"Failed to process item {item.get('Name', 'Unknown')}: {e}")
                            self.stats['errors'] += 1
                            continue
                    
                    # Update pagination
                    start_index += len(items)
                    total_items += len(items)
                    
                    # Break if we've processed all items
                    if start_index >= total_count:
                        break
                
                logger.info(f"Completed library '{library_name}': processed {total_count} items "
                           f"(skipped: {self.stats['items_skipped']})")
            
            # Update final statistics
            elapsed_time = time.time() - self.stats['start_time']
            logger.info(f"Sync completed successfully!")
            logger.info(f"  Total items processed: {self.stats['items_processed']}")
            logger.info(f"  Items added: {self.stats['items_added']}")
            logger.info(f"  Items updated: {self.stats['items_updated']}")
            logger.info(f"  Items skipped: {self.stats['items_skipped']}")
            logger.info(f"  Genres added: {self.stats['genres_added']}")
            logger.info(f"  People added: {self.stats['people_added']}")
            logger.info(f"  Errors: {self.stats['errors']}")
            logger.info(f"  Time elapsed: {elapsed_time:.2f} seconds")
            
            # Calculate time savings
            if self.stats['items_skipped'] > 0:
                processed_non_skipped = max(self.stats['items_processed'] - self.stats['items_skipped'], 1)
                potential_time = elapsed_time * (self.stats['items_processed']) / processed_non_skipped
                time_saved = potential_time - elapsed_time
                logger.info(f"  Estimated time saved by skipping: {time_saved:.2f} seconds")
            
            # Update sync operation
            status = 'completed' if self.stats['errors'] == 0 else 'completed_with_errors'
            self.update_sync_operation(cursor, sync_id, status)
            
            # Update system config with last sync time
            cursor.execute("""
                INSERT INTO system_config (key, value, description, updated_at)
                VALUES ('last_full_sync', %s, 'Timestamp of last full Jellyfin sync', CURRENT_TIMESTAMP)
                ON CONFLICT (key) 
                DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """, (json.dumps(datetime.now(timezone.utc).isoformat()),))
            
            return True
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.update_sync_operation(cursor, sync_id, 'failed', str(e))
            return False
            
        finally:
            cursor.close()
    
    def get_library_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current library state.
        
        Returns:
            Dictionary of library statistics
        """
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Total items by type
            cursor.execute("""
                SELECT media_type, COUNT(*) as count
                FROM media_items
                GROUP BY media_type
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            stats['items_by_type'] = {str(row['media_type']): int(row['count']) for row in results}
            
            # Total items
            stats['total_items'] = sum(stats['items_by_type'].values())
            
            # Recent additions
            cursor.execute("""
                SELECT COUNT(*) 
                FROM media_items 
                WHERE date_added >= NOW() - INTERVAL '7 days'
            """)
            stats['added_last_week'] = int(cursor.fetchone()['count'])
            
            # Recently analyzed items
            cursor.execute("""
                SELECT COUNT(*) 
                FROM media_items 
                WHERE last_analyzed >= NOW() - INTERVAL '24 hours'
            """)
            stats['analyzed_last_24h'] = int(cursor.fetchone()['count'])
            
            # Items never analyzed
            cursor.execute("""
                SELECT COUNT(*) 
                FROM media_items 
                WHERE last_analyzed IS NULL
            """)
            stats['never_analyzed'] = int(cursor.fetchone()['count'])
            
            # Genres count
            cursor.execute("SELECT COUNT(*) FROM genres")
            stats['total_genres'] = int(cursor.fetchone()['count'])
            
            # People count
            cursor.execute("SELECT COUNT(*) FROM people")
            stats['total_people'] = int(cursor.fetchone()['count'])
            
            # Studios count
            cursor.execute("SELECT COUNT(*) FROM studios")
            stats['total_studios'] = int(cursor.fetchone()['count'])
            
            # User activity
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_activity")
            stats['active_users'] = int(cursor.fetchone()['count'])
            
            # Last sync info
            cursor.execute("""
                SELECT status, completed_at, items_processed, metadata
                FROM sync_operations
                WHERE operation_type IN ('full_sync', 'incremental_sync')
                ORDER BY started_at DESC
                LIMIT 1
            """)
            last_sync = cursor.fetchone()
            if last_sync:
                try:
                    metadata = json.loads(last_sync['metadata'] or '{}')
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                    
                stats['last_sync'] = {
                    'status': str(last_sync['status']),
                    'completed_at': last_sync['completed_at'].isoformat() if last_sync['completed_at'] else None,
                    'items_processed': int(last_sync['items_processed'] or 0),
                    'items_skipped': int(metadata.get('items_skipped', 0)),
                    'items_added': int(metadata.get('items_added', 0)),
                    'items_updated': int(metadata.get('items_updated', 0))
                }
            
            cursor.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get library stats: {e}")
            return {}
    
    def cleanup_orphaned_data(self) -> bool:
        """
        Clean up orphaned data (people, genres, studios not linked to any media).
        
        Returns:
            True if cleanup completed successfully, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Count orphaned records before cleanup
            cursor.execute("""
                SELECT COUNT(*) FROM people p
                LEFT JOIN media_credits mc ON p.id = mc.person_id
                WHERE mc.person_id IS NULL
            """)
            orphaned_people = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(*) FROM genres g
                LEFT JOIN media_genres mg ON g.id = mg.genre_id
                WHERE mg.genre_id IS NULL
            """)
            orphaned_genres = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT COUNT(*) FROM studios s
                LEFT JOIN media_studios ms ON s.id = ms.studio_id
                WHERE ms.studio_id IS NULL
            """)
            orphaned_studios = cursor.fetchone()['count']
            
            logger.info(f"Found {orphaned_people} orphaned people, {orphaned_genres} orphaned genres, {orphaned_studios} orphaned studios")
            
            # Clean up orphaned people
            cursor.execute("""
                DELETE FROM people
                WHERE id IN (
                    SELECT p.id FROM people p
                    LEFT JOIN media_credits mc ON p.id = mc.person_id
                    WHERE mc.person_id IS NULL
                )
            """)
            
            # Clean up orphaned genres
            cursor.execute("""
                DELETE FROM genres
                WHERE id IN (
                    SELECT g.id FROM genres g
                    LEFT JOIN media_genres mg ON g.id = mg.genre_id
                    WHERE mg.genre_id IS NULL
                )
            """)
            
            # Clean up orphaned studios
            cursor.execute("""
                DELETE FROM studios
                WHERE id IN (
                    SELECT s.id FROM studios s
                    LEFT JOIN media_studios ms ON s.id = ms.studio_id
                    WHERE ms.studio_id IS NULL
                )
            """)
            
            logger.info("Cleanup completed successfully")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False
    
    def reset_analysis_timestamps(self, older_than_hours: int = 24) -> bool:
        """
        Reset last_analyzed timestamps for items older than specified hours.
        Useful for forcing a re-analysis of items.
        
        Args:
            older_than_hours: Reset timestamps for items analyzed more than this many hours ago
            
        Returns:
            True if reset completed successfully, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
            
            cursor.execute("""
                UPDATE media_items 
                SET last_analyzed = NULL 
                WHERE last_analyzed < %s
            """, (cutoff_time,))
            
            reset_count = cursor.rowcount
            logger.info(f"Reset last_analyzed for {reset_count} items older than {older_than_hours} hours")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset analysis timestamps: {e}")
            return False
    
    def close_connections(self):
        """Close all connections."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
        
        if self.session:
            self.session.close()
            logger.info("HTTP session closed")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch and sync Jellyfin library data to PostgreSQL database"
    )
    
    # Jellyfin settings
    parser.add_argument(
        '--jellyfin-host',
        default=os.getenv('JELLYFIN_HOST', 'http://localhost:8096'),
        help='Jellyfin server URL (default: http://localhost:8096, env: JELLYFIN_HOST)'
    )
    
    parser.add_argument(
        '--token-file',
        default=os.getenv('JELLYFIN_TOKEN_FILE', 'jellyfin_token.json'),
        help='Path to Jellyfin token file (default: jellyfin_token.json, env: JELLYFIN_TOKEN_FILE)'
    )
    
    # Database settings
    parser.add_argument(
        '--db-host',
        default=os.getenv('POSTGRES_HOST', 'localhost'),
        help='PostgreSQL server host (default: localhost, env: POSTGRES_HOST)'
    )
    
    parser.add_argument(
        '--db-port',
        type=int,
        default=int(os.getenv('POSTGRES_PORT', '5432')),
        help='PostgreSQL server port (default: 5432, env: POSTGRES_PORT)'
    )
    
    parser.add_argument(
        '--db-username',
        default=os.getenv('POSTGRES_USERNAME', 'postgres'),
        help='Database username (default: postgres, env: POSTGRES_USERNAME)'
    )
    
    parser.add_argument(
        '--db-password',
        default=os.getenv('POSTGRES_PASSWORD'),
        help='Database password (env: POSTGRES_PASSWORD)'
    )
    
    parser.add_argument(
        '--db-name',
        default=os.getenv('POSTGRES_DATABASE', 'media_recommendation'),
        help='Database name (default: media_recommendation, env: POSTGRES_DATABASE)'
    )
    
    # Sync options
    parser.add_argument(
        '--libraries',
        nargs='+',
        help='Specific library names to sync (default: all libraries)'
    )
    
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Perform incremental sync based on modification dates and analysis timestamps'
    )
    
    parser.add_argument(
        '--force-refresh-hours',
        type=int,
        default=24,
        help='Hours after which to force refresh items regardless of modification (default: 24)'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only display library statistics without syncing'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up orphaned data after sync'
    )
    
    parser.add_argument(
        '--reset-analysis',
        type=int,
        metavar='HOURS',
        help='Reset last_analyzed for items older than specified hours (forces re-analysis)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate required arguments
    if not args.db_password:
        logger.error("Database password is required. Set via --db-password or POSTGRES_PASSWORD environment variable.")
        sys.exit(1)
    
    logger.info("Starting Jellyfin library sync...")
    logger.info(f"Jellyfin: {args.jellyfin_host}")
    logger.info(f"Database: {args.db_username}@{args.db_host}:{args.db_port}/{args.db_name}")
    
    if args.incremental:
        logger.info(f"Incremental sync enabled (force refresh after {args.force_refresh_hours} hours)")
    
    # Prepare database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'username': args.db_username,
        'password': args.db_password,
        'database': args.db_name
    }
    
    # Initialize fetcher
    fetcher = JellyfinLibraryFetcher(
        jellyfin_host=args.jellyfin_host,
        db_config=db_config,
        token_file=args.token_file
    )
    
    try:
        # Load authentication token
        if not fetcher.load_auth_token():
            logger.error("Failed to load authentication token. Please run jellyfin authentication first.")
            sys.exit(1)
        
        # Connect to database
        if not fetcher.connect_to_database():
            sys.exit(1)
        
        # Test Jellyfin connection
        if not fetcher.test_jellyfin_connection():
            sys.exit(1)
        
        # Handle reset analysis option
        if args.reset_analysis:
            logger.info(f"Resetting analysis timestamps for items older than {args.reset_analysis} hours...")
            if not fetcher.reset_analysis_timestamps(args.reset_analysis):
                logger.error("Failed to reset analysis timestamps")
                sys.exit(1)
        
        if args.stats_only:
            # Display statistics only
            stats = fetcher.get_library_stats()
            
            print("\n=== LIBRARY STATISTICS ===")
            print(f"Total items: {stats.get('total_items', 0)}")
            print(f"Added last week: {stats.get('added_last_week', 0)}")
            print(f"Analyzed in last 24h: {stats.get('analyzed_last_24h', 0)}")
            print(f"Never analyzed: {stats.get('never_analyzed', 0)}")
            print(f"Total genres: {stats.get('total_genres', 0)}")
            print(f"Total people: {stats.get('total_people', 0)}")
            print(f"Total studios: {stats.get('total_studios', 0)}")
            print(f"Active users: {stats.get('active_users', 0)}")
            
            print("\nItems by type:")
            for media_type, count in stats.get('items_by_type', {}).items():
                print(f"  {media_type}: {count}")
            
            if 'last_sync' in stats:
                last_sync = stats['last_sync']
                print(f"\nLast sync: {last_sync['status']} at {last_sync['completed_at']}")
                print(f"Items processed: {last_sync['items_processed']}")
                print(f"Items skipped: {last_sync.get('items_skipped', 0)}")
                print(f"Items added: {last_sync.get('items_added', 0)}")
                print(f"Items updated: {last_sync.get('items_updated', 0)}")
            
        else:
            # Perform sync
            success = fetcher.sync_library(
                library_filter=args.libraries,
                incremental=args.incremental,
                force_refresh_hours=args.force_refresh_hours
            )
            
            if not success:
                logger.error("Sync failed")
                sys.exit(1)
            
            # Cleanup if requested
            if args.cleanup:
                logger.info("Running cleanup...")
                if not fetcher.cleanup_orphaned_data():
                    logger.warning("Cleanup completed with errors")
            
            # Display final stats
            stats = fetcher.get_library_stats()
            print(f"\nSync completed! Total items in library: {stats.get('total_items', 0)}")
    
    except KeyboardInterrupt:
        logger.info("Sync cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        fetcher.close_connections()

if __name__ == "__main__":
    main()