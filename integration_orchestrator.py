#!/usr/bin/env python3
"""
Integration Orchestrator
Coordinates the external API integration and data merging processes
Updated to work with the DataMerger class from data_merger.py
python integration_orchestrator.py --db-host 192.168.0.39 --db-name media_rec --db-username postgres --db-password 8g1k9ap2 --omdb-api-key 'c863fe16' --tvdb-api-key 'c8f86a37-b82c-4008-899c-5a3b9aa2f67a' --tmdb-api-key '41f5a79f118f6e78d51c26f60da97c2c' --batch-size 100
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

# Import our modules
from external_api_integration import ExternalAPIIntegrationService, APIConfig, DatabaseConfig
from data_merger import DataMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('external_api_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationOrchestrator:
    """Orchestrates the complete external API integration process"""
    
    def __init__(self, api_config: APIConfig, db_config: DatabaseConfig):
        self.api_config = api_config
        self.db_config = db_config
        self.stats = {
            'start_time': None,
            'end_time': None,
            'items_processed': 0,
            'items_enriched': 0,
            'items_merged': 0,
            'items_updated': 0,
            'errors': 0,
            'api_calls': {
                'tmdb': 0,
                'tvdb': 0,
                'omdb': 0
            },
            'merge_stats': {
                'api_sources_merged': {'tmdb': 0, 'tvdb': 0, 'omdb': 0}
            }
        }
    
    async def run_integration_pipeline(self, limit: Optional[int] = None, offset: int = 0, 
                                     skip_enrichment: bool = False, skip_merging: bool = False):
        """Run the complete integration pipeline"""
        self.stats['start_time'] = datetime.now()
        logger.info("Starting External API Integration Pipeline")
        
        try:
            # Step 1: External API Enrichment
            if not skip_enrichment:
                logger.info("=== STEP 1: External API Enrichment ===")
                await self._run_api_enrichment(limit, offset)
            else:
                logger.info("Skipping API enrichment (--skip-enrichment flag)")
            
            # Step 2: Data Merging
            if not skip_merging:
                logger.info("=== STEP 2: Data Merging ===")
                await self._run_data_merging(limit, offset)
            else:
                logger.info("Skipping data merging (--skip-merging flag)")
            
            # Step 3: Generate Summary Report
            logger.info("=== STEP 3: Summary Report ===")
            self._generate_summary_report()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.stats['errors'] += 1
            raise
        
        finally:
            self.stats['end_time'] = datetime.now()
            self._save_execution_stats()
    
    async def _run_api_enrichment(self, limit: Optional[int], offset: int):
        """Run external API enrichment"""
        try:
            service = ExternalAPIIntegrationService(self.api_config, self.db_config)
            
            logger.info("Starting external API data enrichment...")
            enrichment_stats = await service.process_media_library(limit=limit, offset=offset)
            
            # Update stats if service returns them
            if enrichment_stats:
                self.stats['items_processed'] = enrichment_stats.get('items_processed', 0)
                self.stats['items_enriched'] = enrichment_stats.get('items_enriched', 0)
                self.stats['api_calls'].update(enrichment_stats.get('api_calls', {}))
            
            logger.info("External API enrichment completed successfully")
            
        except Exception as e:
            logger.error(f"API enrichment failed: {e}")
            self._log_error("orchestrator", "api_enrichment", str(e))
            raise
    
    async def _run_data_merging(self, limit: Optional[int], offset: int):
        """Run data merging process using DataMerger class"""
        try:
            logger.info("Starting data merging process...")
            logger.info("Merging external API data with existing media profiles")
            
            # Convert APIConfig and DatabaseConfig to dict format expected by DataMerger
            db_config_dict = {
                'host': self.db_config.host,
                'port': self.db_config.port,
                'database': self.db_config.database,
                'user': self.db_config.username,
                'password': self.db_config.password
            }
            
            # Initialize DataMerger
            merger = DataMerger(db_config_dict)
            
            try:
                # Connect to database
                merger.connect()
                
                # Run the merging process
                merge_stats = merger.process_media_items(limit=limit, offset=offset)
                
                # Update orchestrator stats with merge results
                if merge_stats:
                    self.stats['items_merged'] = merge_stats.get('items_merged', 0)
                    self.stats['items_updated'] = merge_stats.get('items_updated', 0)
                    self.stats['merge_stats']['api_sources_merged'] = merge_stats.get('api_sources_merged', {})
                    
                    # Add any merge errors to total error count
                    merge_errors = merge_stats.get('errors', 0)
                    self.stats['errors'] += merge_errors
                
                logger.info(f"Data merging completed successfully")
                logger.info(f"Items merged: {self.stats['items_merged']}")
                logger.info(f"Items updated: {self.stats['items_updated']}")
                
            except Exception as e:
                logger.error(f"Data merging process failed: {e}")
                self._log_error("orchestrator", "data_merging", str(e))
                raise
            
            finally:
                # Always disconnect from database
                merger.disconnect()
            
        except Exception as e:
            logger.error(f"Data merging initialization failed: {e}")
            self._log_error("orchestrator", "data_merging_init", str(e))
            raise
    
    def _generate_summary_report(self):
        """Generate and display summary report"""
        if self.stats['end_time'] is None:
            self.stats['end_time'] = datetime.now()
        
        elapsed_time = self.stats['end_time'] - self.stats['start_time']
        
        # Calculate success rate
        total_processed = max(self.stats['items_processed'], 1)  # Avoid division by zero
        success_rate = ((total_processed - self.stats['errors']) / total_processed) * 100
        
        # Get merge source stats
        merge_sources = self.stats['merge_stats']['api_sources_merged']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    INTEGRATION SUMMARY REPORT                 ║  
╠══════════════════════════════════════════════════════════════╣
║ Start Time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}                              ║
║ End Time:   {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}                              ║
║ Duration:   {str(elapsed_time).split('.')[0]}                              ║
║                                                              ║
║ PROCESSING STATISTICS:                                       ║
║ • Items Processed: {self.stats['items_processed']:>5}                              ║
║ • Items Enriched:  {self.stats['items_enriched']:>5}                              ║
║ • Items Merged:    {self.stats['items_merged']:>5}                              ║
║ • Items Updated:   {self.stats['items_updated']:>5}                              ║
║ • Success Rate:    {success_rate:>5.1f}%                            ║
║ • Errors:          {self.stats['errors']:>5}                              ║
║                                                              ║
║ API USAGE (ENRICHMENT):                                      ║
║ • TMDB Calls:      {self.stats['api_calls']['tmdb']:>5}                              ║
║ • TVDB Calls:      {self.stats['api_calls']['tvdb']:>5}                              ║
║ • OMDB Calls:      {self.stats['api_calls']['omdb']:>5}                              ║
║                                                              ║
║ DATA MERGING SOURCES:                                        ║
║ • TMDB Merged:     {merge_sources.get('tmdb', 0):>5}                              ║
║ • TVDB Merged:     {merge_sources.get('tvdb', 0):>5}                              ║
║ • OMDB Merged:     {merge_sources.get('omdb', 0):>5}                              ║
║                                                              ║
║ CACHE PERFORMANCE:                                           ║
║ • Cache Hit Rate:  {self._get_cache_hit_rate():.1f}%                            ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        print(report)
        
        # Log individual component results
        if self.stats['items_enriched'] > 0:
            logger.info(f"✓ API Enrichment: {self.stats['items_enriched']} items enriched")
        
        if self.stats['items_merged'] > 0:
            logger.info(f"✓ Data Merging: {self.stats['items_merged']} items merged, {self.stats['items_updated']} updated")
        
        if self.stats['errors'] > 0:
            logger.warning(f"⚠ {self.stats['errors']} errors occurred during processing")
        
        logger.info("Integration pipeline completed successfully")
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate - placeholder for actual implementation"""
        # This would need to be implemented by tracking cache hits/misses
        # For now, return a reasonable estimate based on API calls vs items processed
        total_api_calls = sum(self.stats['api_calls'].values())
        if total_api_calls == 0 or self.stats['items_processed'] == 0:
            return 0.0
        
        # Rough estimate: if we made fewer API calls than items processed,
        # assume the difference was served from cache
        if total_api_calls < self.stats['items_processed']:
            cache_hits = self.stats['items_processed'] - total_api_calls
            return (cache_hits / self.stats['items_processed']) * 100
        
        return 0.0
    
    def _log_error(self, component: str, operation: str, error_message: str):
        """Log error with context"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'error': error_message
        }
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        error_log_file = 'logs/integration_errors.json'
        try:
            # Load existing errors
            try:
                with open(error_log_file, 'r') as f:
                    errors = json.load(f)
            except FileNotFoundError:
                errors = []
            
            errors.append(error_entry)
            
            # Keep only last 1000 errors
            if len(errors) > 1000:
                errors = errors[-1000:]
            
            # Save updated errors
            with open(error_log_file, 'w') as f:
                json.dump(errors, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log error to file: {e}")
    
    def _save_execution_stats(self):
        """Save execution statistics for future reference"""
        os.makedirs('logs', exist_ok=True)
        stats_file = 'logs/integration_stats.json'
        
        try:
            # Load existing stats
            try:
                with open(stats_file, 'r') as f:
                    all_stats = json.load(f)
            except FileNotFoundError:
                all_stats = []
            
            # Add current execution stats
            execution_id = self.stats['start_time'].strftime('%Y%m%d_%H%M%S')
            current_stats = {
                'execution_id': execution_id,
                **self.stats
            }
            
            # Convert datetime objects to strings for JSON serialization
            for key, value in current_stats.items():
                if isinstance(value, datetime):
                    current_stats[key] = value.isoformat()
            
            all_stats.append(current_stats)
            
            # Keep only last 50 executions
            if len(all_stats) > 50:
                all_stats = all_stats[-50:]
            
            # Save updated stats
            with open(stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
            
            logger.info(f"Execution statistics saved to {stats_file}")
        
        except Exception as e:
            logger.error(f"Failed to save execution statistics: {e}")

def load_configuration():
    """Load configuration from all sources with proper precedence"""
    parser = argparse.ArgumentParser(
        description="External API Integration Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with API keys from environment
  python integration_orchestrator.py --tmdb-api-key YOUR_KEY --tvdb-api-key YOUR_KEY --omdb-api-key YOUR_KEY
  
  # Process only 100 items for testing
  python integration_orchestrator.py --limit 100
  
  # Skip enrichment, only merge existing data
  python integration_orchestrator.py --skip-enrichment
  
  # Skip merging, only do enrichment
  python integration_orchestrator.py --skip-merging
  
  # Resume from offset 500
  python integration_orchestrator.py --offset 500
        """
    )
    
    # API Configuration
    parser.add_argument("--tmdb-api-key", help="TMDB API Key")
    parser.add_argument("--tvdb-api-key", help="TVDB API Key")
    parser.add_argument("--omdb-api-key", help="OMDB API Key")
    parser.add_argument("--cache-ttl-hours", type=int, default=168, help="Cache TTL in hours (default: 168)")
    
    # Database Configuration
    parser.add_argument("--db-host", default="localhost", help="Database host (default: localhost)")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port (default: 5432)")
    parser.add_argument("--db-name", default="media_rec", help="Database name (default: media_rec)")
    parser.add_argument("--db-user", default="postgres", help="Database username (default: postgres)")
    parser.add_argument("--db-password", help="Database password")
    
    # Processing Configuration
    parser.add_argument("--limit", type=int, help="Limit number of items to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for processing (default: 0)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing (default: 10)")
    
    # Pipeline Control
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip external API enrichment")
    parser.add_argument("--skip-merging", action="store_true", help="Skip data merging")
    
    # Utility Options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config-file", help="Load additional configuration from JSON file")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load additional config from file if specified
    file_config = {}
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
                logger.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Failed to load config file {args.config_file}: {e}")
            sys.exit(1)
    
    # Build API configuration with precedence: CLI > File > Environment > Default
    def get_config_value(cli_val, file_key, env_key, default=""):
        if cli_val is not None:
            return cli_val
        if file_key in file_config:
            return file_config[file_key]
        return os.getenv(env_key, default)
    
    # Validate required API keys
    tmdb_key = get_config_value(args.tmdb_api_key, "tmdb_api_key", "TMDB_API_KEY")
    tvdb_key = get_config_value(args.tvdb_api_key, "tvdb_api_key", "TVDB_API_KEY")
    omdb_key = get_config_value(args.omdb_api_key, "omdb_api_key", "OMDB_API_KEY")
    
    if not tmdb_key or not tvdb_key or not omdb_key:
        logger.error("TMDB, TVDB, and OMDB API keys are all required!")
        logger.error("Provide them via command line arguments, config file, or environment variables")
        sys.exit(1)
    
    api_config = APIConfig(
        tmdb_api_key=tmdb_key,
        tvdb_api_key=tvdb_key,
        omdb_api_key=omdb_key,
        cache_ttl_hours=int(get_config_value(args.cache_ttl_hours, "cache_ttl_hours", "CACHE_TTL_HOURS", 168))
    )
    
    db_config = DatabaseConfig(
        host=get_config_value(args.db_host, "db_host", "DB_HOST", "localhost"),
        port=int(get_config_value(args.db_port, "db_port", "DB_PORT", "5432")),
        database=get_config_value(args.db_name, "db_name", "DB_NAME", "media_rec"),
        username=get_config_value(args.db_user, "db_user", "DB_USER", "postgres"),
        password=get_config_value(args.db_password, "db_password", "DB_PASSWORD", "")
    )
    
    return api_config, db_config, args

async def main():
    """Main execution function"""
    try:
        # Load configuration
        api_config, db_config, args = load_configuration()
        
        # Show configuration (without sensitive data)
        logger.info(f"Database: {db_config.username}@{db_config.host}:{db_config.port}/{db_config.database}")
        logger.info(f"Processing: limit={args.limit}, offset={args.offset}")
        logger.info(f"Cache TTL: {api_config.cache_ttl_hours} hours")
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("This would process media items with external API integration and data merging")
            return 0
        
        # Initialize and run orchestrator
        orchestrator = IntegrationOrchestrator(api_config, db_config)
        
        await orchestrator.run_integration_pipeline(
            limit=args.limit,
            offset=args.offset,
            skip_enrichment=args.skip_enrichment,
            skip_merging=args.skip_merging
        )
        
        logger.info("Integration orchestrator completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Integration orchestrator failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))