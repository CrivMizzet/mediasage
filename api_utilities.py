#!/usr/bin/env python3
"""
API Utilities Module
Helper functions and utilities for external API integration
Updated to support TMDB, TVDB, and OMDB APIs with the orchestrator
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    tmdb_api_key: str
    tvdb_api_key: str
    omdb_api_key: str
    cache_ttl_hours: int = 168

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str

class MediaMatcher:
    """Utilities for matching media items with external API results"""
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for better matching"""
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove common prefixes/suffixes
        prefixes = ['the ', 'a ', 'an ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @staticmethod
    def calculate_title_similarity(title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        norm1 = MediaMatcher.normalize_title(title1)
        norm2 = MediaMatcher.normalize_title(title2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Check if one is contained in the other
        if norm1 in norm2 or norm2 in norm1:
            return 0.9
        
        # Simple word overlap calculation
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def find_best_match(target_title: str, candidates: List[Dict], title_key: str = 'title') -> Optional[Dict]:
        """Find the best matching candidate from a list"""
        if not candidates:
            return None
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            candidate_title = candidate.get(title_key, '')
            # Handle TVDB API response structure
            if not candidate_title and 'name' in candidate:
                candidate_title = candidate.get('name', '')
            
            if not candidate_title:
                continue
            
            score = MediaMatcher.calculate_title_similarity(target_title, candidate_title)
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Only return match if score is above threshold
        return best_match if best_score >= 0.7 else None
    
    @staticmethod
    def match_by_year_and_title(target_title: str, target_year: Optional[int], 
                               candidates: List[Dict], title_key: str = 'title', 
                               year_key: str = 'release_date') -> Optional[Dict]:
        """Enhanced matching that considers both title and year"""
        if not candidates:
            return None
        
        scored_candidates = []
        
        for candidate in candidates:
            candidate_title = candidate.get(title_key, '') or candidate.get('name', '')
            if not candidate_title:
                continue
            
            title_score = MediaMatcher.calculate_title_similarity(target_title, candidate_title)
            
            # Calculate year score
            year_score = 0.0
            if target_year:
                candidate_date = candidate.get(year_key, '') or candidate.get('first_air_date', '')
                candidate_year = extract_year_from_date(candidate_date)
                
                if candidate_year:
                    year_diff = abs(target_year - candidate_year)
                    if year_diff == 0:
                        year_score = 1.0
                    elif year_diff <= 1:
                        year_score = 0.8
                    elif year_diff <= 3:
                        year_score = 0.5
                    else:
                        year_score = 0.1
            
            # Combined score with title being more important
            combined_score = (title_score * 0.7) + (year_score * 0.3)
            
            if combined_score > 0.0:
                scored_candidates.append((candidate, combined_score))
        
        if not scored_candidates:
            return None
        
        # Sort by score and return best match if above threshold
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_score = scored_candidates[0]
        
        return best_candidate if best_score >= 0.6 else None

class DataValidator:
    """Validates and cleans external API data"""
    
    @staticmethod
    def validate_tmdb_data(data: Dict) -> bool:
        """Validate TMDB API response data"""
        if not isinstance(data, dict):
            return False
        
        # Check for required fields
        required_fields = ['id']
        for field in required_fields:
            if field not in data:
                logger.warning(f"TMDB data missing required field: {field}")
                return False
        
        return True
    
    @staticmethod
    def validate_tvdb_data(data: Dict) -> bool:
        """Validate TVDB API response data"""
        if not isinstance(data, dict):
            return False
        
        # TVDB API can return different structures, be flexible
        if 'data' in data:
            # Check if data contains the actual content
            content = data['data']
            if isinstance(content, list) and len(content) > 0:
                return True
            elif isinstance(content, dict) and content.get('id'):
                return True
        elif data.get('id'):
            # Direct data structure
            return True
        
        logger.warning("TVDB data does not contain valid structure")
        return False
    
    @staticmethod
    def validate_omdb_data(data: Dict) -> bool:
        """Validate OMDB API response data"""
        if not isinstance(data, dict):
            return False
        
        # Check response status
        if data.get('Response') != 'True':
            logger.warning(f"OMDB API returned error: {data.get('Error', 'Unknown error')}")
            return FALSE
        
        # Check for required fields
        required_fields = ['Title', 'Type']
        for field in required_fields:
            if field not in data:
                logger.warning(f"OMDB data missing required field: {field}")
                return False
        
        return True
    
    @staticmethod
    def clean_numeric_value(value: Any) -> Optional[float]:
        """Clean and convert numeric values"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common non-numeric characters
            cleaned = re.sub(r'[^\d.-]', '', value)
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None
        
        return None
    
    @staticmethod
    def clean_date_value(value: Any) -> Optional[str]:
        """Clean and standardize date values"""
        if not value:
            return None
        
        if isinstance(value, str):
            # Try to parse and reformat date
            date_formats = ['%Y-%m-%d', '%Y', '%d %b %Y', '%b %d, %Y', '%Y-%m-%d %H:%M:%S']
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(value, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        
        return str(value) if value else None
    
    @staticmethod
    def validate_api_response(api_source: str, data: Any) -> bool:
        """Validate API response based on source"""
        if api_source.lower() == 'tmdb':
            return DataValidator.validate_tmdb_data(data)
        elif api_source.lower() == 'tvdb':
            return DataValidator.validate_tvdb_data(data)
        elif api_source.lower() == 'omdb':
            return DataValidator.validate_omdb_data(data)
        else:
            logger.warning(f"Unknown API source: {api_source}")
            return False

class ConfigManager:
    """Manages configuration files for API integration"""
    
    CONFIG_DIR = 'config'
    CONFIG_FILE = 'external_api_config.json'
    CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE)
    
    @staticmethod
    def ensure_config_dir():
        """Ensure config directory exists"""
        os.makedirs(ConfigManager.CONFIG_DIR, exist_ok=True)
    
    @staticmethod
    def load_config() -> Dict:
        """Load configuration from file"""
        ConfigManager.ensure_config_dir()
        
        try:
            with open(ConfigManager.CONFIG_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info(f"Config file {ConfigManager.CONFIG_PATH} not found, using defaults")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict):
        """Save configuration to file"""
        ConfigManager.ensure_config_dir()
        
        try:
            with open(ConfigManager.CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {ConfigManager.CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    @staticmethod
    def update_config(updates: Dict):
        """Update specific configuration values"""
        config = ConfigManager.load_config()
        config.update(updates)
        ConfigManager.save_config(config)
    
    @staticmethod
    def get_config_value(key: str, default: Any = None) -> Any:
        """Get a specific configuration value"""
        config = ConfigManager.load_config()
        return config.get(key, default)
    
    @staticmethod
    def load_api_config() -> Optional[APIConfig]:
        """Load API configuration from file"""
        config = ConfigManager.load_config()
        
        try:
            return APIConfig(
                tmdb_api_key=config.get('tmdb_api_key', ''),
                tvdb_api_key=config.get('tvdb_api_key', ''),
                omdb_api_key=config.get('omdb_api_key', ''),
                cache_ttl_hours=config.get('cache_ttl_hours', 168)
            )
        except Exception as e:
            logger.error(f"Error loading API config: {e}")
            return None
    
    @staticmethod
    def load_database_config() -> Optional[DatabaseConfig]:
        """Load database configuration from file"""
        config = ConfigManager.load_config()
        
        try:
            return DatabaseConfig(
                host=config.get('db_host', 'localhost'),
                port=config.get('db_port', 5432),
                database=config.get('db_name', 'media_rec'),
                username=config.get('db_user', 'postgres'),
                password=config.get('db_password', '')
            )
        except Exception as e:
            logger.error(f"Error loading database config: {e}")
            return None

class RateLimitTracker:
    """Tracks API rate limit usage for TMDB, TVDB, and OMDB"""
    
    LOG_DIR = 'logs'
    USAGE_FILE = 'api_usage.json'
    USAGE_PATH = os.path.join(LOG_DIR, USAGE_FILE)
    
    # Rate limits per API (calls per day)
    RATE_LIMITS = {
        'tmdb': 10000,    # TMDB allows 1000 requests per day for free tier
        'tvdb': 4000,     # TVDB allows 4000 requests per day
        'omdb': 1000      # OMDB allows 1000 requests per day for free tier
    }
    
    @staticmethod
    def ensure_log_dir():
        """Ensure log directory exists"""
        os.makedirs(RateLimitTracker.LOG_DIR, exist_ok=True)
    
    @staticmethod
    def load_usage() -> Dict:
        """Load API usage data"""
        RateLimitTracker.ensure_log_dir()
        
        try:
            with open(RateLimitTracker.USAGE_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            logger.error("Error parsing usage file, starting fresh")
            return {}
    
    @staticmethod
    def save_usage(usage: Dict):
        """Save API usage data"""
        RateLimitTracker.ensure_log_dir()
        
        try:
            with open(RateLimitTracker.USAGE_PATH, 'w') as f:
                json.dump(usage, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving usage data: {e}")
    
    @staticmethod
    def record_api_call(api_source: str, endpoint: str):
        """Record an API call for tracking"""
        usage = RateLimitTracker.load_usage()
        
        today = datetime.now().strftime('%Y-%m-%d')
        api_source = api_source.lower()
        
        if api_source not in usage:
            usage[api_source] = {}
        
        if today not in usage[api_source]:
            usage[api_source][today] = {}
        
        if endpoint not in usage[api_source][today]:
            usage[api_source][today][endpoint] = 0
        
        usage[api_source][today][endpoint] += 1
        
        RateLimitTracker.save_usage(usage)
        
        # Log warning if approaching rate limit
        daily_usage = RateLimitTracker.get_daily_usage(api_source, today)
        rate_limit = RateLimitTracker.RATE_LIMITS.get(api_source, 1000)
        
        if daily_usage >= rate_limit * 0.9:  # 90% of limit
            logger.warning(f"{api_source.upper()} API usage at {daily_usage}/{rate_limit} calls today")
        elif daily_usage >= rate_limit * 0.8:  # 80% of limit
            logger.info(f"{api_source.upper()} API usage at {daily_usage}/{rate_limit} calls today")
    
    @staticmethod
    def get_daily_usage(api_source: str, date: Optional[str] = None) -> int:
        """Get daily usage for an API source"""
        usage = RateLimitTracker.load_usage()
        
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        api_source = api_source.lower()
        
        if api_source not in usage or date not in usage[api_source]:
            return 0
        
        return sum(usage[api_source][date].values())
    
    @staticmethod
    def check_rate_limit(api_source: str) -> bool:
        """Check if API calls are within rate limit"""
        daily_usage = RateLimitTracker.get_daily_usage(api_source)
        rate_limit = RateLimitTracker.RATE_LIMITS.get(api_source.lower(), 1000)
        
        return daily_usage < rate_limit
    
    @staticmethod
    def get_remaining_calls(api_source: str) -> int:
        """Get remaining calls for today"""
        daily_usage = RateLimitTracker.get_daily_usage(api_source)
        rate_limit = RateLimitTracker.RATE_LIMITS.get(api_source.lower(), 1000)
        
        return max(0, rate_limit - daily_usage)
    
    @staticmethod
    def cleanup_old_usage(days_to_keep: int = 30):
        """Clean up old usage data"""
        usage = RateLimitTracker.load_usage()
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_usage = {}
        
        for api_source, dates in usage.items():
            cleaned_usage[api_source] = {}
            
            for date, endpoints in dates.items():
                try:
                    date_obj = datetime.strptime(date, '%Y-%m-%d')
                    if date_obj >= cutoff_date:
                        cleaned_usage[api_source][date] = endpoints
                except ValueError:
                    # Skip invalid date entries
                    continue
        
        RateLimitTracker.save_usage(cleaned_usage)
        logger.info(f"Cleaned up usage data older than {days_to_keep} days")

class ErrorHandler:
    """Handles and logs API errors"""
    
    LOG_DIR = 'logs'
    ERROR_FILE = 'api_errors.json'
    ERROR_PATH = os.path.join(LOG_DIR, ERROR_FILE)
    
    @staticmethod
    def ensure_log_dir():
        """Ensure log directory exists"""
        os.makedirs(ErrorHandler.LOG_DIR, exist_ok=True)
    
    @staticmethod
    def log_error(api_source: str, endpoint: str, error: str, context: Dict = None):
        """Log an API error"""
        ErrorHandler.ensure_log_dir()
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'api_source': api_source.lower(),
            'endpoint': endpoint,
            'error': str(error),
            'context': context or {}
        }
        
        try:
            # Load existing errors
            try:
                with open(ErrorHandler.ERROR_PATH, 'r') as f:
                    errors = json.load(f)
            except FileNotFoundError:
                errors = []
            
            # Add new error
            errors.append(error_entry)
            
            # Keep only last 1000 errors
            if len(errors) > 1000:
                errors = errors[-1000:]
            
            # Save updated errors
            with open(ErrorHandler.ERROR_PATH, 'w') as f:
                json.dump(errors, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to log API error: {e}")
        
        # Also log to standard logger
        logger.error(f"API Error - {api_source.upper()}:{endpoint} - {error}")
    
    @staticmethod
    def get_recent_errors(hours: int = 24, api_source: Optional[str] = None) -> List[Dict]:
        """Get recent API errors"""
        try:
            with open(ErrorHandler.ERROR_PATH, 'r') as f:
                errors = json.load(f)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_errors = []
            for error in errors:
                try:
                    error_time = datetime.fromisoformat(error['timestamp'])
                    if error_time >= cutoff_time:
                        if api_source is None or error.get('api_source') == api_source.lower():
                            recent_errors.append(error)
                except ValueError:
                    continue
            
            return recent_errors
        
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []
    
    @staticmethod
    def get_error_summary(hours: int = 24) -> Dict[str, int]:
        """Get error summary by API source"""
        recent_errors = ErrorHandler.get_recent_errors(hours)
        
        summary = {}
        for error in recent_errors:
            api_source = error.get('api_source', 'unknown')
            summary[api_source] = summary.get(api_source, 0) + 1
        
        return summary

# Utility Functions

def format_runtime(minutes: Optional[int]) -> Optional[str]:
    """Format runtime in minutes to human-readable string"""
    if not minutes:
        return None
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours > 0:
        return f"{hours}h {remaining_minutes}m"
    else:
        return f"{remaining_minutes}m"

def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string"""
    if not date_str:
        return None
    
    # Try to extract 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
    if year_match:
        return int(year_match.group())
    
    return None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove/replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized[:255]  # Limit length

def calculate_confidence_score(scores: List[float]) -> float:
    """Calculate overall confidence score from multiple sources"""
    if not scores:
        return 0.0
    
    # Remove None values
    valid_scores = [score for score in scores if score is not None]
    
    if not valid_scores:
        return 0.0
    
    # Calculate weighted average (more sources = higher confidence)
    base_score = sum(valid_scores) / len(valid_scores)
    source_bonus = min(0.1 * len(valid_scores), 0.3)  # Up to 30% bonus for multiple sources
    
    return min(1.0, base_score + source_bonus)

def merge_arrays(arrays: List[List], deduplicate: bool = True) -> List:
    """Merge multiple arrays into one"""
    merged = []
    
    for array in arrays:
        if isinstance(array, list):
            merged.extend(array)
    
    if deduplicate:
        # Preserve order while removing duplicates
        seen = set()
        result = []
        for item in merged:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    return merged

def safe_get_nested(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary values"""
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def normalize_media_type(media_type: str, api_source: str = '') -> str:
    """Normalize media type across different APIs"""
    if not media_type:
        return 'unknown'
    
    media_type = media_type.lower()
    
    # TMDB mappings
    if api_source.lower() == 'tmdb':
        if media_type == 'movie':
            return 'Movie'
        elif media_type == 'tv':
            return 'Series'
    
    # TVDB mappings
    elif api_source.lower() == 'tvdb':
        if media_type in ['movie', 'film']:
            return 'Movie'
        elif media_type in ['series', 'tv', 'television']:
            return 'Series'
    
    # OMDB mappings
    elif api_source.lower() == 'omdb':
        if media_type == 'movie':
            return 'Movie'
        elif media_type == 'series':
            return 'Series'
    
    # Generic mappings
    if media_type in ['movie', 'film']:
        return 'Movie'
    elif media_type in ['series', 'tv', 'television', 'show']:
        return 'Series'
    
    return media_type.title()

def extract_genres_from_response(data: Dict, api_source: str) -> List[str]:
    """Extract genres from API response based on source"""
    genres = []
    
    if api_source.lower() == 'tmdb':
        genre_list = data.get('genres', [])
        if isinstance(genre_list, list):
            genres = [g.get('name', '') for g in genre_list if isinstance(g, dict) and g.get('name')]
    
    elif api_source.lower() == 'tvdb':
        # TVDB can have genres in different formats
        genre_data = data.get('genres', []) or data.get('genre', [])
        if isinstance(genre_data, list):
            for genre in genre_data:
                if isinstance(genre, str):
                    genres.append(genre)
                elif isinstance(genre, dict):
                    genres.append(genre.get('name', ''))
    
    elif api_source.lower() == 'omdb':
        genre_str = data.get('Genre', '')
        if genre_str:
            genres = [g.strip() for g in genre_str.split(',')]
    
    return [g for g in genres if g]  # Remove empty strings

class APIResponseCache:
    """Simple in-memory cache for API responses during processing"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache:
            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                if self.access_order:
                    oldest_key = self.access_order.pop(0)
                    if oldest_key in self.cache:
                        del self.cache[oldest_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached values"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate()
        }

def batch_process_items(items: List[Any], batch_size: int = 10):
    """Generator to process items in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def estimate_processing_time(total_items: int, items_per_minute: int = 30) -> str:
    """Estimate processing time for given number of items"""
    if items_per_minute <= 0:
        return "Unknown"
    
    total_minutes = total_items / items_per_minute
    
    if total_minutes < 1:
        return f"{int(total_minutes * 60)} seconds"
    elif total_minutes < 60:
        return f"{int(total_minutes)} minutes"
    else:
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        return f"{hours}h {minutes}m"

class ProgressTracker:
    """Track and report progress of long-running operations"""
    
    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed_items = 0
        self.start_time = datetime.now()
        self.last_report_time = self.start_time
        self.report_interval = 60  # Report every 60 seconds
        self.api_stats = {'tmdb': 0, 'tvdb': 0, 'omdb': 0}
    
    def update(self, items_processed: int = 1, api_source: Optional[str] = None):
        """Update progress"""
        self.processed_items += items_processed
        
        if api_source:
            api_source = api_source.lower()
            if api_source in self.api_stats:
                self.api_stats[api_source] += 1
        
        current_time = datetime.now()
        
        # Report progress periodically
        if (current_time - self.last_report_time).seconds >= self.report_interval:
            self.report_progress()
            self.last_report_time = current_time
    
    def report_progress(self):
        """Report current progress"""
        if self.total_items == 0:
            return
        
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        
        percentage = (self.processed_items / self.total_items) * 100
        
        # Estimate remaining time
        if self.processed_items > 0:
            avg_time_per_item = elapsed_time.total_seconds() / self.processed_items
            remaining_items = self.total_items - self.processed_items
            estimated_remaining = timedelta(seconds=avg_time_per_item * remaining_items)
            
            api_summary = ", ".join([f"{api.upper()}: {count}" for api, count in self.api_stats.items() if count > 0])
            
            logger.info(
                f"{self.operation_name}: {self.processed_items}/{self.total_items} "
                f"({percentage:.1f}%) - Elapsed: {str(elapsed_time).split('.')[0]} - "
                f"Est. remaining: {str(estimated_remaining).split('.')[0]} - "
                f"API calls: {api_summary}"
            )
        else:
            logger.info(f"{self.operation_name}: {self.processed_items}/{self.total_items} ({percentage:.1f}%)")
    
    def finish(self):
        """Report completion"""
        elapsed_time = datetime.now() - self.start_time
        api_summary = ", ".join([f"{api.upper()}: {count}" for api, count in self.api_stats.items() if count > 0])
        
        logger.info(
            f"{self.operation_name} completed: {self.processed_items} items in "
            f"{str(elapsed_time).split('.')[0]} - Total API calls: {api_summary}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        
        return {
            'operation_name': self.operation_name,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'percentage_complete': (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0,
            'elapsed_time_seconds': elapsed_time.total_seconds(),
            'api_stats': self.api_stats.copy(),
            'items_per_second': self.processed_items / elapsed_time.total_seconds() if elapsed_time.total_seconds() > 0 else 0
        }

class APIHealthChecker:
    """Monitor API health and availability"""
    
    def __init__(self):
        self.last_check_time = {}
        self.check_interval = 300  # 5 minutes
        self.api_status = {
            'tmdb': {'available': True, 'last_error': None, 'error_count': 0},
            'tvdb': {'available': True, 'last_error': None, 'error_count': 0},
            'omdb': {'available': True, 'last_error': None, 'error_count': 0}
        }
    
    def record_success(self, api_source: str):
        """Record successful API call"""
        api_source = api_source.lower()
        if api_source in self.api_status:
            self.api_status[api_source]['available'] = True
            self.api_status[api_source]['error_count'] = 0
            self.api_status[api_source]['last_error'] = None
    
    def record_error(self, api_source: str, error: str):
        """Record API error"""
        api_source = api_source.lower()
        if api_source in self.api_status:
            self.api_status[api_source]['error_count'] += 1
            self.api_status[api_source]['last_error'] = error
            
            # Mark as unavailable after 5 consecutive errors
            if self.api_status[api_source]['error_count'] >= 5:
                self.api_status[api_source]['available'] = False
                logger.warning(f"{api_source.upper()} API marked as unavailable after 5 consecutive errors")
    
    def is_api_available(self, api_source: str) -> bool:
        """Check if API is currently available"""
        api_source = api_source.lower()
        return self.api_status.get(api_source, {}).get('available', False)
    
    def get_api_status(self, api_source: str) -> Dict[str, Any]:
        """Get detailed API status"""
        api_source = api_source.lower()
        return self.api_status.get(api_source, {
            'available': False, 
            'last_error': 'Unknown API', 
            'error_count': 0
        })
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all APIs"""
        return self.api_status.copy()

def create_cache_key(api_source: str, endpoint: str, params: Dict[str, Any]) -> str:
    """Create a consistent cache key for API requests"""
    # Sort parameters for consistent key generation
    sorted_params = sorted(params.items())
    param_str = "&".join([f"{k}={v}" for k, v in sorted_params])
    
    # Create key from api_source, endpoint, and parameters
    key_parts = [api_source.lower(), endpoint, param_str]
    cache_key = "|".join(key_parts)
    
    # Hash long keys to avoid filesystem issues
    if len(cache_key) > 200:
        import hashlib
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
    
    return cache_key

def validate_external_ids(data: Dict) -> Dict[str, Optional[str]]:
    """Extract and validate external IDs from API response"""
    external_ids = {
        'imdb_id': None,
        'tmdb_id': None,
        'tvdb_id': None
    }
    
    # Direct ID fields
    if 'imdb_id' in data:
        external_ids['imdb_id'] = str(data['imdb_id']) if data['imdb_id'] else None
    if 'id' in data:
        external_ids['tmdb_id'] = str(data['id']) if data['id'] else None
    
    # External IDs object (common in TMDB responses)
    if 'external_ids' in data:
        ext_ids = data['external_ids']
        if isinstance(ext_ids, dict):
            if 'imdb_id' in ext_ids:
                external_ids['imdb_id'] = str(ext_ids['imdb_id']) if ext_ids['imdb_id'] else None
            if 'tvdb_id' in ext_ids:
                external_ids['tvdb_id'] = str(ext_ids['tvdb_id']) if ext_ids['tvdb_id'] else None
    
    # TVDB specific fields
    if 'tvdb' in data:
        external_ids['tvdb_id'] = str(data['tvdb']) if data['tvdb'] else None
    
    # Validate IMDB ID format
    if external_ids['imdb_id'] and not external_ids['imdb_id'].startswith('tt'):
        external_ids['imdb_id'] = f"tt{external_ids['imdb_id']}"
    
    return external_ids

def clean_overview_text(text: Optional[str]) -> Optional[str]:
    """Clean and standardize overview/plot text"""
    if not text:
        return None
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common prefixes that don't add value
    prefixes_to_remove = [
        'Plot Summary:',
        'Plot:',
        'Summary:',
        'Overview:',
        'Description:'
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    return text if text else None

def parse_content_rating(rating: Optional[str], media_type: str = 'movie') -> Optional[str]:
    """Parse and standardize content rating"""
    if not rating:
        return None
    
    rating = rating.strip().upper()
    
    # Handle different rating systems
    movie_ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'NR', 'UNRATED']
    tv_ratings = ['TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA']
    
    # Direct match
    if rating in movie_ratings or rating in tv_ratings:
        return rating
    
    # Handle variations
    rating_mappings = {
        'NOT RATED': 'NR',
        'APPROVED': 'NR',
        'PASSED': 'NR',
        'M': 'R',  # Old MPAA rating
        'X': 'NC-17',  # Old MPAA rating
        'GP': 'PG',  # Old MPAA rating
        'PG13': 'PG-13',
        'TV14': 'TV-14',
        'TVMA': 'TV-MA',
        'TVY7': 'TV-Y7'
    }
    
    return rating_mappings.get(rating, rating)

def extract_production_companies(data: Dict, api_source: str) -> List[str]:
    """Extract production companies from API response"""
    companies = []
    
    if api_source.lower() == 'tmdb':
        company_list = data.get('production_companies', [])
        if isinstance(company_list, list):
            companies = [c.get('name', '') for c in company_list if isinstance(c, dict) and c.get('name')]
    
    elif api_source.lower() == 'omdb':
        production = data.get('Production', '')
        if production and production != 'N/A':
            companies = [p.strip() for p in production.split(',')]
    
    elif api_source.lower() == 'tvdb':
        # TVDB might have network or production info
        network = data.get('network', '')
        if network:
            companies.append(network)
    
    return [c for c in companies if c]  # Remove empty strings

def merge_external_data(sources: List[Dict[str, Any]], priority_order: List[str] = None) -> Dict[str, Any]:
    """Merge data from multiple external API sources"""
    if not sources:
        return {}
    
    if priority_order is None:
        priority_order = ['tmdb', 'tvdb', 'omdb']  # Default priority
    
    merged_data = {}
    
    # Define field mappings and priorities
    field_mappings = {
        'title': ['title', 'name', 'Title'],
        'original_title': ['original_title', 'original_name'],
        'overview': ['overview', 'plot', 'Plot'],
        'release_date': ['release_date', 'first_air_date', 'Released'],
        'runtime': ['runtime', 'episode_run_time', 'Runtime'],
        'genres': ['genres', 'Genre'],
        'rating': ['vote_average', 'imdbRating'],
        'content_rating': ['content_rating', 'Rated'],
        'poster_path': ['poster_path', 'Poster'],
        'backdrop_path': ['backdrop_path'],
        'popularity': ['popularity'],
        'vote_count': ['vote_count', 'imdbVotes']
    }
    
    # Organize sources by API type
    sources_by_api = {}
    for source in sources:
        api_type = source.get('api_source', '').lower()
        if api_type:
            sources_by_api[api_type] = source.get('data', {})
    
    # Merge fields according to priority
    for field, possible_keys in field_mappings.items():
        for api in priority_order:
            if api in sources_by_api:
                data = sources_by_api[api]
                for key in possible_keys:
                    if key in data and data[key] is not None:
                        merged_data[field] = data[key]
                        break
                if field in merged_data:
                    break
    
    # Handle special cases
    if 'runtime' in merged_data:
        # Convert runtime to minutes if it's in different format
        runtime = merged_data['runtime']
        if isinstance(runtime, str):
            # Handle formats like "120 min" or "2h 30m"
            runtime_match = re.search(r'(\d+)', runtime)
            if runtime_match:
                merged_data['runtime'] = int(runtime_match.group(1))
        elif isinstance(runtime, list) and runtime:
            # Take first runtime for TV shows
            merged_data['runtime'] = runtime[0]
    
    return merged_data

# Example usage and testing functions
if __name__ == "__main__":
    # Test media matching
    matcher = MediaMatcher()
    
    test_cases = [
        ("The Matrix", "Matrix"),
        ("Star Wars: A New Hope", "Star Wars"),
        ("The Lord of the Rings: The Fellowship of the Ring", "Lord of the Rings Fellowship"),
    ]
    
    print("=== Media Matching Tests ===")
    for title1, title2 in test_cases:
        similarity = matcher.calculate_title_similarity(title1, title2)
        print(f"'{title1}' vs '{title2}': {similarity:.2f}")
    
    # Test data validation
    validator = DataValidator()
    
    print("\n=== Data Validation Tests ===")
    
    # Test TMDB data
    tmdb_test_data = {"id": 123, "title": "Test Movie"}
    print(f"TMDB validation: {validator.validate_tmdb_data(tmdb_test_data)}")
    
    # Test TVDB data
    tvdb_test_data = {"data": {"id": 456, "name": "Test Series"}}
    print(f"TVDB validation: {validator.validate_tvdb_data(tvdb_test_data)}")
    
    # Test OMDB data
    omdb_test_data = {"Response": "True", "Title": "Test Movie", "Type": "movie"}
    print(f"OMDB validation: {validator.validate_omdb_data(omdb_test_data)}")
    
    # Test numeric cleaning
    print(f"Clean '$123,456': {validator.clean_numeric_value('$123,456')}")
    print(f"Clean '7.5/10': {validator.clean_numeric_value('7.5/10')}")
    
    # Test rate limiting
    print("\n=== Rate Limiting Tests ===")
    rate_tracker = RateLimitTracker()
    
    # Simulate API calls
    for i in range(5):
        rate_tracker.record_api_call('tmdb', 'search')
        rate_tracker.record_api_call('tvdb', 'series')
        rate_tracker.record_api_call('omdb', 'search')
    
    print(f"TMDB daily usage: {rate_tracker.get_daily_usage('tmdb')}")
    print(f"TVDB daily usage: {rate_tracker.get_daily_usage('tvdb')}")
    print(f"OMDB daily usage: {rate_tracker.get_daily_usage('omdb')}")
    
    # Test progress tracker
    print("\n=== Progress Tracking Test ===")
    tracker = ProgressTracker(100, "Test Operation")
    for i in range(25):
        api_source = ['tmdb', 'tvdb', 'omdb'][i % 3]
        tracker.update(api_source=api_source)
    
    tracker.report_progress()
    stats = tracker.get_stats()
    print(f"Processing stats: {stats}")
    
    # Test cache
    print("\n=== Cache Test ===")
    cache = APIResponseCache(max_size=5)
    
    # Add some items
    for i in range(7):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Test retrieval
    print(f"Retrieved key_0: {cache.get('key_0')}")  # Should be None (evicted)
    print(f"Retrieved key_6: {cache.get('key_6')}")  # Should be value_6
    print(f"Cache stats: {cache.get_stats()}")
    
    # Test API health checker
    print("\n=== API Health Test ===")
    health_checker = APIHealthChecker()
    
    # Simulate some errors
    health_checker.record_error('tmdb', 'Connection timeout')
    health_checker.record_error('tmdb', 'Rate limit exceeded')
    health_checker.record_success('tvdb')
    
    print(f"TMDB status: {health_checker.get_api_status('tmdb')}")
    print(f"TVDB status: {health_checker.get_api_status('tvdb')}")
    print(f"All status: {health_checker.get_all_status()}")
    
    print("\n=== Utility Function Tests ===")
    
    # Test genre extraction
    tmdb_data = {
        "genres": [
            {"id": 28, "name": "Action"},
            {"id": 12, "name": "Adventure"}
        ]
    }
    genres = extract_genres_from_response(tmdb_data, 'tmdb')
    print(f"TMDB genres: {genres}")
    
    # Test external ID validation
    test_ids = {"imdb_id": "123456", "id": 789, "external_ids": {"tvdb_id": 101112}}
    validated_ids = validate_external_ids(test_ids)
    print(f"Validated IDs: {validated_ids}")
    
    # Test content rating parsing
    ratings = ['PG-13', 'TV-MA', 'R', 'UNRATED', 'PG13']
    for rating in ratings:
        parsed = parse_content_rating(rating)
        print(f"Rating '{rating}' -> '{parsed}'")
    
    print("\nAll tests completed!")
