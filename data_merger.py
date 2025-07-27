#!/usr/bin/env python3
"""
Data Merger Module
Merges external API data with existing media profiles
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import psycopg2.extras
import re
from decimal import Decimal

# Import utilities
from api_utilities import (
    DataValidator, MediaMatcher, ConfigManager, ErrorHandler, 
    ProgressTracker, calculate_confidence_score, merge_arrays,
    safe_get_nested, extract_year_from_date, format_runtime
)

logger = logging.getLogger(__name__)

class DataMerger:
    """Merges external API data with existing media profiles"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection_pool = None
        self.validator = DataValidator()
        self.matcher = MediaMatcher()
        self.merge_stats = {
            'items_processed': 0,
            'items_merged': 0,
            'items_updated': 0,
            'errors': 0,
            'api_sources_merged': {'tmdb': 0, 'tvdb': 0, 'omdb': 0},
            'start_time': None,
            'end_time': None
        }
    
    def connect(self):
        """Establish database connection pool"""
        try:
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            logger.info("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        self.connection_pool.putconn(conn)
    
    def process_media_items(self, limit: Optional[int] = None, offset: int = 0):
        """Process media items for data merging"""
        self.merge_stats['start_time'] = datetime.now()
        
        conn = None
        try:
            conn = self.get_connection()
            
            # Get items with external data to merge
            items = self._get_items_for_merging(conn, limit, offset)
            
            if not items:
                logger.info("No items found for merging")
                return self.merge_stats
            
            logger.info(f"Processing {len(items)} items for data merging")
            
            # Initialize progress tracker
            progress = ProgressTracker(len(items), "Data Merging")
            
            for item in items:
                try:
                    self._process_single_item(conn, item)
                    self.merge_stats['items_processed'] += 1
                    progress.update()
                    
                except Exception as e:
                    self.merge_stats['errors'] += 1
                    ErrorHandler.log_error(
                        'data_merger', 
                        'process_item', 
                        str(e),
                        {'media_item_id': str(item['id'])}
                    )
                    logger.error(f"Error processing item {item['id']}: {e}")
            
            progress.finish()
            
        except Exception as e:
            logger.error(f"Error in process_media_items: {e}")
            self.merge_stats['errors'] += 1
            
        finally:
            if conn:
                self.return_connection(conn)
            
            self.merge_stats['end_time'] = datetime.now()
            self._log_final_stats()
        
        return self.merge_stats
    
    def _get_items_for_merging(self, conn, limit: Optional[int], offset: int) -> List[Dict]:
        """Get media items that have external data to merge"""
        query = """
        SELECT mi.id, mi.jellyfin_id, mi.title, mi.media_type, 
               mi.release_date, mi.tmdb_id, mi.tvdb_id, mi.imdb_id,
               mi.overview, mi.runtime_minutes, mi.community_rating,
               mi.critic_rating, mi.content_rating, mi.tagline, mi.date_modified
        FROM media_items mi
        INNER JOIN media_external_data med ON mi.id = med.media_item_id
        WHERE med.data_type = 'enrichment'
        AND mi.is_available = true
        ORDER BY mi.date_modified DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    
    def _process_single_item(self, conn, item: Dict):
        """Process a single media item for data merging"""
        media_item_id = item['id']
        
        # Get external data for this item
        external_data = self._get_external_data(conn, media_item_id)
        
        if not external_data:
            logger.debug(f"No external data found for item {media_item_id}")
            return
        
        # Merge data from different API sources
        merged_data = self._merge_api_data(item, external_data)
        
        if not merged_data:
            logger.debug(f"No data to merge for item {media_item_id}")
            return
        
        # Update media_items table with merged data
        self._update_media_item(conn, media_item_id, merged_data)
        
        # Update or create media profile
        self._update_media_profile(conn, media_item_id, merged_data)
        
        self.merge_stats['items_merged'] += 1
        logger.debug(f"Successfully merged data for item {media_item_id}")
    
    def _get_external_data(self, conn, media_item_id: str) -> Dict[str, Dict]:
        """Get all external data for a media item"""
        query = """
        SELECT api_source, external_id, raw_data, processed_data
        FROM media_external_data
        WHERE media_item_id = %s AND data_type = 'enrichment'
        ORDER BY last_updated DESC
        """
        
        external_data = {}
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (media_item_id,))
            rows = cursor.fetchall()
            
            for row in rows:
                api_source = row['api_source']
                data = row['processed_data'] or row['raw_data']
                
                if data and self._validate_external_data(api_source, data):
                    external_data[api_source] = data
                    self.merge_stats['api_sources_merged'][api_source] += 1
        
        return external_data
    
    def _validate_external_data(self, api_source: str, data: Dict) -> bool:
        """Validate external API data"""
        if api_source == 'tmdb':
            return self.validator.validate_tmdb_data(data)
        elif api_source == 'omdb':
            return self.validator.validate_omdb_data(data)
        elif api_source == 'tvdb':
            # Basic TVDB validation
            return isinstance(data, dict) and 'id' in data
        
        return False
    
    def _merge_api_data(self, original_item: Dict, external_data: Dict[str, Dict]) -> Dict:
        """Merge data from multiple API sources"""
        merged = {}
        
        # Define merge priorities (higher number = higher priority)
        source_priorities = {'omdb': 1, 'tvdb': 2, 'tmdb': 3}
        
        # Fields to merge with their mapping and merge strategy
        merge_fields = {
            # Basic info
            'title': {'sources': ['tmdb', 'tvdb', 'omdb'], 'strategy': 'priority'},
            'original_title': {'sources': ['tmdb'], 'strategy': 'single'},
            'overview': {'sources': ['tmdb', 'tvdb'], 'strategy': 'longest'},
            'tagline': {'sources': ['tmdb'], 'strategy': 'single'},
            
            # Dates and runtime
            'release_date': {'sources': ['tmdb', 'tvdb', 'omdb'], 'strategy': 'earliest'},
            'runtime_minutes': {'sources': ['tmdb', 'omdb'], 'strategy': 'average'},
            
            # Ratings
            'community_rating': {'sources': ['tmdb', 'omdb'], 'strategy': 'weighted_average'},
            'critic_rating': {'sources': ['omdb'], 'strategy': 'single'},
            'content_rating': {'sources': ['omdb', 'tmdb'], 'strategy': 'priority'},
            
            # External IDs (preserve existing)
            'imdb_id': {'sources': ['tmdb', 'omdb'], 'strategy': 'preserve_existing'},
            'tmdb_id': {'sources': ['tmdb'], 'strategy': 'preserve_existing'},
            'tvdb_id': {'sources': ['tvdb'], 'strategy': 'preserve_existing'},
        }
        
        for field, config in merge_fields.items():
            merged_value = self._merge_field_value(
                field, config, original_item, external_data, source_priorities
            )
            
            if merged_value is not None:
                merged[field] = merged_value
        
        # Add additional metadata
        if merged:
            merged['last_external_update'] = datetime.now()
            merged['external_data_sources'] = list(external_data.keys())
        
        return merged
    
    def _merge_field_value(self, field: str, config: Dict, original_item: Dict, 
                          external_data: Dict[str, Dict], source_priorities: Dict) -> Any:
        """Merge a specific field value from multiple sources"""
        strategy = config['strategy']
        sources = config['sources']
        
        # Collect values from available sources
        values = []
        
        for source in sources:
            if source not in external_data:
                continue
            
            value = self._extract_field_value(field, source, external_data[source])
            if value is not None:
                values.append({
                    'value': value,
                    'source': source,
                    'priority': source_priorities.get(source, 0)
                })
        
        if not values:
            return None
        
        # Apply merge strategy
        if strategy == 'priority':
            return max(values, key=lambda x: x['priority'])['value']
        
        elif strategy == 'single':
            return values[0]['value']
        
        elif strategy == 'longest':
            text_values = [v for v in values if isinstance(v['value'], str)]
            if text_values:
                return max(text_values, key=lambda x: len(x['value']))['value']
            return values[0]['value']
        
        elif strategy == 'earliest':
            date_values = []
            for v in values:
                parsed_date = self._parse_date(v['value'])
                if parsed_date:
                    date_values.append({'date': parsed_date, 'original': v['value']})
            
            if date_values:
                earliest = min(date_values, key=lambda x: x['date'])
                return earliest['original']
            return values[0]['value']
        
        elif strategy == 'average':
            numeric_values = []
            for v in values:
                num_val = self._clean_numeric_value(v['value'])
                if num_val is not None:
                    numeric_values.append(num_val)
            
            if numeric_values:
                return int(sum(numeric_values) / len(numeric_values))
            return values[0]['value']
        
        elif strategy == 'weighted_average':
            return self._calculate_weighted_rating(values)
        
        elif strategy == 'preserve_existing':
            # Only update if original doesn't have value
            original_value = original_item.get(field)
            if original_value:
                return None  # Don't update
            return max(values, key=lambda x: x['priority'])['value']
        
        return values[0]['value']
    
    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """Clean and validate numeric values, handling N/A cases"""
        if value is None:
            return None
        
        # Handle string representations
        if isinstance(value, str):
            value = value.strip()
            # Handle N/A cases
            if value.upper() in ['N/A', 'NULL', 'NONE', '', 'TBD', 'TBA']:
                return None
            
            # Try to extract numeric value from string
            try:
                return float(value)
            except ValueError:
                # Try to extract numbers from strings like "120 min"
                import re
                numbers = re.findall(r'[\d.]+', value)
                if numbers:
                    try:
                        return float(numbers[0])
                    except ValueError:
                        pass
                return None
        
        # Handle numeric types
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        
        return None
    
    def _extract_field_value(self, field: str, source: str, data: Dict) -> Any:
        """Extract field value from external API data"""
        if source == 'tmdb':
            field_mappings = {
                'title': 'title',
                'original_title': 'original_title',
                'overview': 'overview',
                'tagline': 'tagline',
                'release_date': 'release_date',
                'runtime_minutes': 'runtime',
                'community_rating': 'vote_average',
                'imdb_id': 'imdb_id',
                'tmdb_id': 'id'
            }
            
        elif source == 'tvdb':
            field_mappings = {
                'title': 'name',
                'overview': 'overview',
                'release_date': 'firstAired',
                'tvdb_id': 'id'
            }
            
        elif source == 'omdb':
            field_mappings = {
                'title': 'Title',
                'release_date': 'Released',
                'runtime_minutes': 'Runtime',
                'community_rating': 'imdbRating',
                'critic_rating': 'Metascore',
                'content_rating': 'Rated',
                'imdb_id': 'imdbID'
            }
        
        else:
            return None
        
        api_field = field_mappings.get(field)
        if not api_field:
            return None
        
        value = safe_get_nested(data, api_field.split('.'))
        
        # Post-process specific fields
        if field == 'runtime_minutes':
            return self._clean_numeric_value(value)
        
        elif field in ['community_rating', 'critic_rating']:
            # Clean and validate rating values
            cleaned_value = self._clean_numeric_value(value)
            if cleaned_value is not None:
                # Normalize ratings to 0-10 scale
                if source == 'omdb' and cleaned_value <= 10:
                    return float(cleaned_value)
                elif source == 'tmdb' and cleaned_value <= 10:
                    return float(cleaned_value)
            return None
        
        elif field == 'release_date' and value:
            return self.validator.clean_date_value(value)
        
        return value
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        date_formats = ['%Y-%m-%d', '%Y', '%d %b %Y', '%b %d, %Y']
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        
        return None
    
    def _calculate_weighted_rating(self, values: List[Dict]) -> Optional[float]:
        """Calculate weighted average rating"""
        if not values:
            return None
        
        # Weight sources differently
        source_weights = {'tmdb': 0.4, 'omdb': 0.6}
        
        weighted_sum = 0
        total_weight = 0
        
        for v in values:
            rating = self._clean_numeric_value(v['value'])
            if rating is not None:
                weight = source_weights.get(v['source'], 0.3)
                weighted_sum += rating * weight
                total_weight += weight
        
        if total_weight > 0:
            return round(weighted_sum / total_weight, 1)
        
        return None
    
    def _update_media_item(self, conn, media_item_id: str, merged_data: Dict):
        """Update media_items table with merged data"""
        if not merged_data:
            return
        
        # Build update query
        update_fields = []
        values = []
        
        for field, value in merged_data.items():
            if field in ['last_external_update', 'external_data_sources']:
                continue  # Skip metadata fields
            
            # Additional validation for numeric fields before database update
            if field in ['runtime_minutes', 'community_rating', 'critic_rating']:
                cleaned_value = self._clean_numeric_value(value)
                if cleaned_value is None:
                    continue  # Skip fields with invalid numeric values
                value = cleaned_value
            
            update_fields.append(f"{field} = %s")
            values.append(value)
        
        if not update_fields:
            return
        
        # Add updated timestamp
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(media_item_id)
        
        query = f"""
        UPDATE media_items 
        SET {', '.join(update_fields)}
        WHERE id = %s
        """
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, values)
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.merge_stats['items_updated'] += 1
                    logger.debug(f"Updated media item {media_item_id}")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating media item {media_item_id}: {e}")
            raise
    
    def _update_media_profile(self, conn, media_item_id: str, merged_data: Dict):
        """Update or create media profile with merged data"""
        # Check if profile exists
        profile_exists = self._check_profile_exists(conn, media_item_id)
        
        if profile_exists:
            self._enhance_existing_profile(conn, media_item_id, merged_data)
        else:
            self._create_basic_profile(conn, media_item_id, merged_data)
    
    def _check_profile_exists(self, conn, media_item_id: str) -> bool:
        """Check if media profile exists"""
        query = "SELECT 1 FROM media_profiles WHERE media_item_id = %s"
        
        with conn.cursor() as cursor:
            cursor.execute(query, (media_item_id,))
            return cursor.fetchone() is not None
    
    def _enhance_existing_profile(self, conn, media_item_id: str, merged_data: Dict):
        """Enhance existing media profile with merged data"""
        # Get current profile
        query = """
        SELECT primary_themes, mood_tags, style_descriptors, 
               target_audience, complexity_level, content_warnings
        FROM media_profiles 
        WHERE media_item_id = %s
        """
        
        current_profile = None
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (media_item_id,))
            current_profile = cursor.fetchone()
        
        if not current_profile:
            return
        
        # Enhance profile with external data insights
        enhanced_data = self._generate_profile_enhancements(merged_data, current_profile)
        
        if enhanced_data:
            self._update_profile_data(conn, media_item_id, enhanced_data)
    
    def _create_basic_profile(self, conn, media_item_id: str, merged_data: Dict):
        """Create basic media profile from merged data"""
        profile_data = {
            'primary_themes': [],
            'mood_tags': [],
            'style_descriptors': [],
            'target_audience': self._infer_target_audience(merged_data),
            'complexity_level': self._infer_complexity_level(merged_data),
            'content_warnings': self._extract_content_warnings(merged_data),
            'profile_complete': False,
            'profile_version': '1.0'
        }
        
        query = """
        INSERT INTO media_profiles (
            media_item_id, primary_themes, mood_tags, style_descriptors,
            target_audience, complexity_level, content_warnings,
            profile_complete, profile_version, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, (
                    media_item_id,
                    profile_data['primary_themes'],
                    profile_data['mood_tags'],
                    profile_data['style_descriptors'],
                    profile_data['target_audience'],
                    profile_data['complexity_level'],
                    profile_data['content_warnings'],
                    profile_data['profile_complete'],
                    profile_data['profile_version']
                ))
                conn.commit()
                logger.debug(f"Created basic profile for {media_item_id}")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating profile for {media_item_id}: {e}")
    
    def _generate_profile_enhancements(self, merged_data: Dict, current_profile: Dict) -> Dict:
        """Generate profile enhancements from merged external data"""
        enhancements = {}
        
        # Enhance target audience based on content rating
        content_rating = merged_data.get('content_rating')
        if content_rating:
            new_audience = self._map_content_rating_to_audience(content_rating)
            if new_audience and new_audience != current_profile.get('target_audience'):
                enhancements['target_audience'] = new_audience
        
        # Add content warnings based on rating
        if content_rating:
            warnings = self._extract_content_warnings(merged_data)
            if warnings:
                existing_warnings = current_profile.get('content_warnings', [])
                merged_warnings = merge_arrays([existing_warnings, warnings], deduplicate=True)
                if len(merged_warnings) > len(existing_warnings):
                    enhancements['content_warnings'] = merged_warnings
        
        return enhancements
    
    def _update_profile_data(self, conn, media_item_id: str, enhanced_data: Dict):
        """Update profile with enhanced data"""
        if not enhanced_data:
            return
        
        update_fields = []
        values = []
        
        for field, value in enhanced_data.items():
            update_fields.append(f"{field} = %s")
            values.append(value)
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(media_item_id)
        
        query = f"""
        UPDATE media_profiles 
        SET {', '.join(update_fields)}
        WHERE media_item_id = %s
        """
        
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, values)
                conn.commit()
                logger.debug(f"Enhanced profile for {media_item_id}")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error enhancing profile for {media_item_id}: {e}")
    
    def _infer_target_audience(self, merged_data: Dict) -> Optional[str]:
        """Infer target audience from merged data"""
        content_rating = merged_data.get('content_rating', '').upper()
        
        if content_rating in ['G', 'TV-G']:
            return 'family'
        elif content_rating in ['PG', 'TV-PG']:
            return 'general'
        elif content_rating in ['PG-13', 'TV-14']:
            return 'teens_adults'
        elif content_rating in ['R', 'TV-MA', 'NC-17']:
            return 'adults'
        
        return 'general'
    
    def _infer_complexity_level(self, merged_data: Dict) -> int:
        """Infer complexity level from merged data"""
        # Base complexity on runtime and rating
        runtime = self._clean_numeric_value(merged_data.get('runtime_minutes', 0)) or 0
        rating = self._clean_numeric_value(merged_data.get('community_rating', 0)) or 0
        
        complexity = 3  # Default medium complexity
        
        # Longer runtime might indicate higher complexity
        if runtime > 150:
            complexity += 1
        elif runtime < 90:
            complexity -= 1
        
        # Higher ratings might indicate higher complexity
        if rating > 8.0:
            complexity += 1
        elif rating < 6.0:
            complexity -= 1
        
        return max(1, min(5, complexity))
    
    def _extract_content_warnings(self, merged_data: Dict) -> List[str]:
        """Extract content warnings from merged data"""
        warnings = []
        content_rating = merged_data.get('content_rating', '').upper()
        
        if content_rating in ['R', 'TV-MA', 'NC-17']:
            warnings.append('mature_content')
        
        if content_rating in ['PG-13', 'TV-14', 'R', 'TV-MA', 'NC-17']:
            warnings.append('parental_guidance')
        
        return warnings
    
    def _map_content_rating_to_audience(self, content_rating: str) -> str:
        """Map content rating to target audience"""
        rating_map = {
            'G': 'family',
            'TV-G': 'family',
            'PG': 'general',
            'TV-PG': 'general',
            'PG-13': 'teens_adults',
            'TV-14': 'teens_adults',
            'R': 'adults',
            'TV-MA': 'adults',
            'NC-17': 'adults'
        }
        
        return rating_map.get(content_rating.upper(), 'general')
    
    def _log_final_stats(self):
        """Log final merge statistics"""
        duration = self.merge_stats['end_time'] - self.merge_stats['start_time']
        
        logger.info("=== Data Merge Complete ===")
        logger.info(f"Duration: {duration}")
        logger.info(f"Items processed: {self.merge_stats['items_processed']}")
        logger.info(f"Items merged: {self.merge_stats['items_merged']}")
        logger.info(f"Items updated: {self.merge_stats['items_updated']}")
        logger.info(f"Errors: {self.merge_stats['errors']}")
        logger.info(f"API sources merged: {self.merge_stats['api_sources_merged']}")
        
        # Save stats to config
        stats_config = {
            'last_merge_run': self.merge_stats['end_time'].isoformat(),
            'last_merge_stats': self.merge_stats
        }
        ConfigManager.update_config(stats_config)

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge external API data with media profiles')
    parser.add_argument('--limit', type=int, help='Limit number of items to process')
    parser.add_argument('--offset', type=int, default=0, help='Offset for processing')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port')
    parser.add_argument('--db-name', default='media_rec', help='Database name')
    parser.add_argument('--db-user', default='postgres', help='Database user')
    parser.add_argument('--db-password', required=True, help='Database password')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Create merger and process
    merger = DataMerger(db_config)
    
    try:
        merger.connect()
        stats = merger.process_media_items(args.limit, args.offset)
        
        print("\n=== Merge Results ===")
        print(f"Items processed: {stats['items_processed']}")
        print(f"Items merged: {stats['items_merged']}")
        print(f"Items updated: {stats['items_updated']}")
        print(f"Errors: {stats['errors']}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        merger.disconnect()

if __name__ == "__main__":
    main()