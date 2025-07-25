#!/usr/bin/env python3
"""
Database Initialization Script for Media Recommendation System with Ollama Integration

This script initializes the PostgreSQL database schema for the media recommendation system
including Ollama analysis capabilities and processing queue management.
It supports configuration via environment variables or command line arguments.

Usage:
    python db_init_new.py --host localhost --username postgres --password mypass --database media_rec
    
    Or using environment variables:
    export POSTGRES_HOST=localhost
    export POSTGRES_USERNAME=postgres
    export POSTGRES_PASSWORD=mypass
    export POSTGRES_DATABASE=media_rec
    python db_init_new.py
"""

import argparse
import logging
import os
import sys
import re
from typing import Optional, List
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('db_init.log')
    ]
)
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Handles PostgreSQL database initialization for the media recommendation system with Ollama integration."""
    
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        """
        Initialize the database initializer.
        
        Args:
            host: PostgreSQL server host
            port: PostgreSQL server port
            username: Database username
            password: Database password
            database: Database name to create/initialize
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.connection = None
        
    def connect_to_postgres(self) -> bool:
        """
        Connect to PostgreSQL server (not to specific database).
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database='postgres'  # Connect to default postgres database first
            )
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info(f"Connected to PostgreSQL server at {self.host}:{self.port}")
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL server: {e}")
            return False
    
    def connect_to_database(self) -> bool:
        """
        Connect to the target database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.connection:
                self.connection.close()
                
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info(f"Connected to database '{self.database}'")
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to database '{self.database}': {e}")
            return False
    
    def database_exists(self) -> bool:
        """
        Check if the target database exists.
        
        Returns:
            True if database exists, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (self.database,)
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists
        except psycopg2.Error as e:
            logger.error(f"Error checking if database exists: {e}")
            return False
    
    def create_database(self) -> bool:
        """
        Create the target database if it doesn't exist.
        
        Returns:
            True if database created or already exists, False otherwise
        """
        try:
            if self.database_exists():
                logger.info(f"Database '{self.database}' already exists")
                return True
                
            cursor = self.connection.cursor()
            cursor.execute(f'CREATE DATABASE "{self.database}"')
            cursor.close()
            logger.info(f"Created database '{self.database}'")
            return True
        except psycopg2.Error as e:
            logger.error(f"Failed to create database '{self.database}': {e}")
            return False
    
    def split_sql_statements(self, sql_content: str) -> List[str]:
        """
        Properly split SQL content into individual statements, handling dollar-quoted strings.
        
        Args:
            sql_content: Raw SQL content
            
        Returns:
            List of individual SQL statements
        """
        statements = []
        current_statement = ""
        in_dollar_quote = False
        dollar_tag = ""
        i = 0
        
        while i < len(sql_content):
            char = sql_content[i]
            
            # Check for dollar-quoted string start/end
            if char == '$' and not in_dollar_quote:
                # Look for dollar tag
                tag_end = sql_content.find('$', i + 1)
                if tag_end != -1:
                    potential_tag = sql_content[i:tag_end + 1]
                    # Check if this looks like a dollar quote tag
                    if re.match(r'\$[a-zA-Z_][a-zA-Z0-9_]*\$', potential_tag) or potential_tag == '$$':
                        in_dollar_quote = True
                        dollar_tag = potential_tag
                        current_statement += sql_content[i:tag_end + 1]
                        i = tag_end + 1
                        continue
            elif char == '$' and in_dollar_quote:
                # Check if this is the closing dollar tag
                if sql_content[i:].startswith(dollar_tag):
                    in_dollar_quote = False
                    current_statement += dollar_tag
                    i += len(dollar_tag)
                    dollar_tag = ""
                    continue
            
            current_statement += char
            
            # Check for statement end (semicolon not in dollar quotes)
            if char == ';' and not in_dollar_quote:
                stmt = current_statement.strip()
                if stmt:
                    statements.append(stmt)
                current_statement = ""
            
            i += 1
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements
    
    def execute_sql_file(self, sql_content: str) -> bool:
        """
        Execute SQL content in the database with proper statement splitting.
        
        Args:
            sql_content: SQL commands to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            
            # Use improved SQL statement splitting
            statements = self.split_sql_statements(sql_content)
            
            successful_statements = 0
            failed_statements = 0
            
            for i, statement in enumerate(statements):
                if statement and not statement.isspace():
                    try:
                        cursor.execute(statement)
                        successful_statements += 1
                        logger.debug(f"Successfully executed statement {i + 1}/{len(statements)}")
                    except psycopg2.Error as e:
                        failed_statements += 1
                        logger.error(f"Error executing statement {i + 1}: {e}")
                        logger.debug(f"Problematic statement: {statement[:500]}...")
                        # For critical errors, we might want to stop
                        if "syntax error" in str(e).lower() and "function" in statement.lower():
                            logger.error("Critical function creation error - stopping execution")
                            cursor.close()
                            return False
            
            cursor.close()
            
            logger.info(f"Schema execution completed: {successful_statements} successful, {failed_statements} failed")
            
            # Consider it successful if most statements worked
            return failed_statements == 0 or (successful_statements > failed_statements * 3)
            
        except psycopg2.Error as e:
            logger.error(f"Failed to execute SQL schema: {e}")
            return False
    
    def verify_schema(self) -> bool:
        """
        Verify that the schema was created correctly by checking for key tables and functions.
        
        Returns:
            True if schema appears to be correctly installed, False otherwise
        """
        expected_tables = [
            'media_items', 'genres', 'media_genres', 'studios', 'people',
            'media_credits', 'users', 'user_activity', 'ollama_analysis',
            'analysis_tags', 'media_analysis_tags', 'external_api_cache',
            'recommendations', 'recommendation_feedback', 'sync_operations',
            'system_config', 'analysis_queue', 'media_profiles', 'media_embeddings'
        ]
        
        try:
            cursor = self.connection.cursor()
            
            # Check tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [table for table in expected_tables if table not in existing_tables]
            
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                cursor.close()
                return False
            
            # Check for the trigger function
            cursor.execute("""
                SELECT routine_name 
                FROM information_schema.routines 
                WHERE routine_schema = 'public' AND routine_name = 'update_updated_at_column'
            """)
            
            functions = cursor.fetchall()
            if not functions:
                logger.warning("Missing update_updated_at_column function")
            else:
                logger.info("Trigger function found")
            
            # Check for triggers
            cursor.execute("""
                SELECT trigger_name, event_object_table
                FROM information_schema.triggers
                WHERE trigger_schema = 'public' AND trigger_name LIKE '%updated_at%'
            """)
            
            triggers = cursor.fetchall()
            logger.info(f"Found {len(triggers)} update triggers")
            
            # Check for jellyfin_etag column
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'media_items' AND column_name = 'jellyfin_etag'
            """)
            
            etag_column = cursor.fetchone()
            if etag_column:
                logger.info("jellyfin_etag column found in media_items table")
            else:
                logger.warning("jellyfin_etag column missing from media_items table")
            
            cursor.close()
            
            logger.info(f"Schema verification passed. Found {len(existing_tables)} tables, {len(functions)} functions, {len(triggers)} triggers.")
            return True
                
        except psycopg2.Error as e:
            logger.error(f"Failed to verify schema: {e}")
            return False
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def get_sql_schema() -> str:
    """
    Return the complete SQL schema as a string, including Ollama integration features.
    """
    return """
-- Media Recommendation System Database Schema with Ollama Integration
-- PostgreSQL Database Design

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE MEDIA TABLES
-- ============================================================================

-- Main media items table (movies, TV shows, etc.)
CREATE TABLE media_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    jellyfin_id VARCHAR(255) UNIQUE NOT NULL,
    jellyfin_etag VARCHAR(255), -- Added for Jellyfin sync optimization
    title VARCHAR(255) NOT NULL,
    original_title VARCHAR(255),
    media_type VARCHAR(50) NOT NULL CHECK (media_type IN ('movie', 'series', 'season', 'episode', 'music', 'book')),
    release_date DATE,
    runtime_minutes INTEGER,
    overview TEXT,
    tagline TEXT,
    content_rating VARCHAR(20),
    community_rating DECIMAL(3,1),
    critic_rating DECIMAL(3,1),
    imdb_id VARCHAR(20),
    tmdb_id INTEGER,
    tvdb_id INTEGER,
    file_path TEXT,
    file_size_bytes BIGINT,
    date_added TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    date_modified TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_analyzed TIMESTAMP WITH TIME ZONE,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Genres lookup table
CREATE TABLE genres (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    jellyfin_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many relationship between media and genres
CREATE TABLE media_genres (
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    genre_id INTEGER REFERENCES genres(id) ON DELETE CASCADE,
    PRIMARY KEY (media_item_id, genre_id)
);

-- Studios/Production companies
CREATE TABLE studios (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    jellyfin_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE media_studios (
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    studio_id INTEGER REFERENCES studios(id) ON DELETE CASCADE,
    PRIMARY KEY (media_item_id, studio_id)
);

-- People (actors, directors, writers, etc.)
CREATE TABLE people (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    jellyfin_id VARCHAR(50),
    imdb_id VARCHAR(20),
    tmdb_id INTEGER,
    birth_date DATE,
    death_date DATE,
    biography TEXT,
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, jellyfin_id)
);

-- Roles/Credits (actor, director, writer, etc.)
CREATE TABLE media_credits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
    role_type VARCHAR(50) NOT NULL CHECK (role_type IN ('actor', 'director', 'writer', 'producer', 'composer', 'other')),
    character_name VARCHAR(255), -- For actors
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- USER ACTIVITY & PREFERENCES
-- ============================================================================

-- User profiles (if supporting multiple users)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    jellyfin_user_id VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    display_name VARCHAR(100),
    email VARCHAR(255),
    last_active TIMESTAMP WITH TIME ZONE,
    preferences JSONB, -- Store user preferences as JSON
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User viewing history and ratings
CREATE TABLE user_activity (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN ('watched', 'partially_watched', 'rated', 'favorited', 'added_to_watchlist')),
    rating DECIMAL(3,1), -- User's personal rating
    watch_progress_percent DECIMAL(5,2), -- 0-100
    watch_count INTEGER DEFAULT 1,
    last_watched TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, media_item_id, activity_type)
);

-- ============================================================================
-- AI ANALYSIS TABLES (OLLAMA INTEGRATION)
-- ============================================================================

-- Ollama analysis results (enhanced)
CREATE TABLE ollama_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    analysis_type VARCHAR(100) NOT NULL, -- 'content_analysis', 'sentiment', 'themes', etc.
    model_used VARCHAR(100) NOT NULL, -- Which Ollama model was used
    prompt_version VARCHAR(50), -- Track prompt versions for consistency
    analysis_version VARCHAR(20), -- Version of analysis algorithm
    analysis_result JSONB NOT NULL, -- Store the full analysis as JSON
    confidence_score DECIMAL(4,3), -- 0-1 confidence in the analysis
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_item_id, analysis_type, model_used)
);

-- Analysis processing queue
CREATE TABLE analysis_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    priority INTEGER DEFAULT 5,
    attempts INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB, -- Store additional processing parameters
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_item_id, analysis_type)
);

-- Media profile summaries for quick access
CREATE TABLE media_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE UNIQUE,
    primary_themes TEXT[],
    mood_tags TEXT[],
    style_descriptors TEXT[],
    target_audience VARCHAR(100),
    complexity_level INTEGER CHECK (complexity_level >= 1 AND complexity_level <= 10),
    emotional_intensity INTEGER CHECK (emotional_intensity >= 1 AND emotional_intensity <= 10),
    recommended_viewing_context TEXT,
    content_warnings TEXT[],
    similar_to_ids UUID[], -- Array of similar media UUIDs
    profile_complete BOOLEAN DEFAULT FALSE,
    profile_version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings storage (temporary before moving to QDrant)
CREATE TABLE media_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    embedding_type VARCHAR(50) NOT NULL, -- 'content', 'style', 'theme', etc.
    embedding_vector FLOAT[], -- Temporary storage before QDrant
    embedding_model VARCHAR(100) NOT NULL,
    vector_dimension INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_item_id, embedding_type, embedding_model)
);

-- Extracted themes and tags from analysis
CREATE TABLE analysis_tags (
    id SERIAL PRIMARY KEY,
    tag_name VARCHAR(100) UNIQUE NOT NULL,
    tag_category VARCHAR(50), -- 'theme', 'mood', 'style', 'content_warning', etc.
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE media_analysis_tags (
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES analysis_tags(id) ON DELETE CASCADE,
    relevance_score DECIMAL(4,3), -- 0-1 how relevant this tag is to the media
    source VARCHAR(50) DEFAULT 'ollama', -- 'ollama', 'external_api', 'manual'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (media_item_id, tag_id)
);

-- ============================================================================
-- EXTERNAL API DATA
-- ============================================================================

-- Cache external API responses
CREATE TABLE external_api_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_source VARCHAR(50) NOT NULL, -- 'tmdb', 'imdb', 'omdb', etc.
    endpoint VARCHAR(255) NOT NULL,
    query_params JSONB,
    response_data JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_external_cache_lookup ON external_api_cache(api_source, endpoint, query_params);

-- Link external API data to media items
CREATE TABLE media_external_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    media_item_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    api_source VARCHAR(50) NOT NULL,
    external_id VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- 'metadata', 'reviews', 'recommendations', etc.
    raw_data JSONB NOT NULL,
    processed_data JSONB, -- Cleaned/normalized version
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(media_item_id, api_source, data_type)
);

-- ============================================================================
-- RECOMMENDATIONS
-- ============================================================================

-- Generated recommendations
CREATE TABLE recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    recommended_media_id UUID REFERENCES media_items(id) ON DELETE CASCADE,
    recommendation_type VARCHAR(50) NOT NULL, -- 'similar', 'trending', 'genre_based', etc.
    algorithm_version VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(4,3) NOT NULL, -- 0-1
    reasoning JSONB, -- Store explanation of why this was recommended
    base_media_ids UUID[], -- Array of media IDs this recommendation is based on
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'shown', 'dismissed', 'downloaded')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_recommendations_user_status ON recommendations(user_id, status);
CREATE INDEX idx_recommendations_confidence ON recommendations(confidence_score DESC);
CREATE INDEX idx_recommendations_created ON recommendations(created_at DESC);

-- User feedback on recommendations
CREATE TABLE recommendation_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    recommendation_id UUID REFERENCES recommendations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    feedback_type VARCHAR(20) NOT NULL CHECK (feedback_type IN ('like', 'dislike', 'not_interested', 'downloaded')),
    feedback_reason VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SYSTEM TABLES
-- ============================================================================

-- Track sync operations with Jellyfin
CREATE TABLE sync_operations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_type VARCHAR(50) NOT NULL, -- 'full_sync', 'incremental_sync', 'analysis_run'
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    items_processed INTEGER DEFAULT 0,
    items_total INTEGER,
    error_message TEXT,
    metadata JSONB -- Store additional operation details
);

-- Configuration and feature flags
CREATE TABLE system_config (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Media items indexes
CREATE INDEX idx_media_items_type ON media_items(media_type);
CREATE INDEX idx_media_items_date_added ON media_items(date_added);
CREATE INDEX idx_media_items_rating ON media_items(community_rating DESC);
CREATE INDEX idx_media_items_jellyfin_id ON media_items(jellyfin_id);
CREATE INDEX idx_media_items_jellyfin_etag ON media_items(jellyfin_etag);
CREATE INDEX idx_media_items_available ON media_items(is_available) WHERE is_available = true;

-- User activity indexes
CREATE INDEX idx_user_activity_user_media ON user_activity(user_id, media_item_id);
CREATE INDEX idx_user_activity_type ON user_activity(activity_type);
CREATE INDEX idx_user_activity_last_watched ON user_activity(last_watched DESC);

-- Analysis indexes
CREATE INDEX idx_ollama_analysis_media ON ollama_analysis(media_item_id);
CREATE INDEX idx_ollama_analysis_type ON ollama_analysis(analysis_type);
CREATE INDEX idx_ollama_analysis_media_type ON ollama_analysis(media_item_id, analysis_type);
CREATE INDEX idx_ollama_analysis_confidence ON ollama_analysis(confidence_score DESC);
CREATE INDEX idx_media_tags_relevance ON media_analysis_tags(relevance_score DESC);

-- Analysis queue indexes
CREATE INDEX idx_analysis_queue_status ON analysis_queue(status);
CREATE INDEX idx_analysis_queue_priority ON analysis_queue(priority DESC, created_at ASC);
CREATE INDEX idx_analysis_queue_media ON analysis_queue(media_item_id);

-- Media profiles indexes
CREATE INDEX idx_media_profiles_complete ON media_profiles(profile_complete);
CREATE INDEX idx_media_profiles_complexity ON media_profiles(complexity_level);
CREATE INDEX idx_media_profiles_intensity ON media_profiles(emotional_intensity);
CREATE INDEX idx_media_profiles_themes ON media_profiles USING GIN(primary_themes);
CREATE INDEX idx_media_profiles_moods ON media_profiles USING GIN(mood_tags);

-- Embeddings indexes
CREATE INDEX idx_media_embeddings_type ON media_embeddings(embedding_type);
CREATE INDEX idx_media_embeddings_media ON media_embeddings(media_item_id);

-- External API cache index
CREATE INDEX idx_external_cache_expires ON external_api_cache(expires_at);
CREATE INDEX idx_external_cache_source ON external_api_cache(api_source);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply the trigger to relevant tables
CREATE TRIGGER update_media_items_updated_at BEFORE UPDATE ON media_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_people_updated_at BEFORE UPDATE ON people
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_activity_updated_at BEFORE UPDATE ON user_activity
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analysis_queue_updated_at 
    BEFORE UPDATE ON analysis_queue
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_media_profiles_updated_at 
    BEFORE UPDATE ON media_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- UTILITY FUNCTIONS FOR OLLAMA PROCESSING
-- ============================================================================

-- Function to get media items ready for analysis
CREATE OR REPLACE FUNCTION get_media_for_analysis(
    batch_size INTEGER DEFAULT 10,
    analysis_type VARCHAR DEFAULT 'thematic_analysis'
) RETURNS TABLE (
    media_id UUID,
    title VARCHAR,
    media_type VARCHAR,
    overview TEXT,
    genres TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mi.id,
        mi.title,
        mi.media_type,
        mi.overview,
        ARRAY_AGG(g.name) as genres
    FROM media_items mi
    LEFT JOIN media_genres mg ON mi.id = mg.media_item_id
    LEFT JOIN genres g ON mg.genre_id = g.id
    WHERE mi.is_available = true
    AND mi.id NOT IN (
        SELECT media_item_id 
        FROM ollama_analysis 
        WHERE analysis_type = get_media_for_analysis.analysis_type
    )
    AND mi.id NOT IN (
        SELECT media_item_id 
        FROM analysis_queue 
        WHERE analysis_type = get_media_for_analysis.analysis_type 
        AND status IN ('pending', 'processing')
    )
    GROUP BY mi.id, mi.title, mi.media_type, mi.overview
    ORDER BY mi.date_added DESC
    LIMIT batch_size;
END;
$$ LANGUAGE plpgsql;

-- Function to add items to analysis queue
CREATE OR REPLACE FUNCTION queue_media_for_analysis(
    media_ids UUID[],
    analysis_type VARCHAR DEFAULT 'thematic_analysis',
    priority INTEGER DEFAULT 5
) RETURNS INTEGER AS $$
DECLARE
    media_id UUID;
    inserted_count INTEGER := 0;
BEGIN
    FOREACH media_id IN ARRAY media_ids
    LOOP
        INSERT INTO analysis_queue (media_item_id, analysis_type, priority)
        VALUES (media_id, analysis_type, priority)
        ON CONFLICT (media_item_id, analysis_type) 
        DO UPDATE SET 
            status = 'pending',
            priority = EXCLUDED.priority,
            updated_at = CURRENT_TIMESTAMP;
        
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR EASY DATA ACCESS
-- ============================================================================

-- View for media with complete profiles
CREATE OR REPLACE VIEW media_with_profiles AS
SELECT 
    mi.id,
    mi.title,
    mi.media_type,
    mi.release_date,
    mi.overview,
    mp.primary_themes,
    mp.mood_tags,
    mp.style_descriptors,
    mp.target_audience,
    mp.complexity_level,
    mp.emotional_intensity,
    mp.profile_complete,
    mp.updated_at as profile_updated_at,
    ARRAY_AGG(DISTINCT g.name) as genres
FROM media_items mi
LEFT JOIN media_profiles mp ON mi.id = mp.media_item_id
LEFT JOIN media_genres mg ON mi.id = mg.media_item_id
LEFT JOIN genres g ON mg.genre_id = g.id
WHERE mi.is_available = true
GROUP BY mi.id, mi.title, mi.media_type, mi.release_date, mi.overview,
         mp.primary_themes, mp.mood_tags, mp.style_descriptors, mp.target_audience,
         mp.complexity_level, mp.emotional_intensity, mp.profile_complete, mp.updated_at;

-- View for analysis queue status
CREATE OR REPLACE VIEW analysis_queue_status AS
SELECT 
    analysis_type,
    status,
    COUNT(*) as count,
    AVG(attempts) as avg_attempts,
    MIN(created_at) as oldest_queued,
    MAX(updated_at) as last_updated
FROM analysis_queue
GROUP BY analysis_type, status
ORDER BY analysis_type, status;

-- ============================================================================
-- INITIAL CONFIGURATION DATA
-- ============================================================================

-- Insert system configuration with Ollama settings
INSERT INTO system_config (key, value, description) VALUES
('schema_version', '"2.0.0"', 'Database schema version with Ollama integration'),
('last_full_sync', 'null', 'Timestamp of last full Jellyfin sync'),
('ollama_host', '"http://localhost:11434"', 'Ollama server endpoint'),
('ollama_default_model', '"llama3.1"', 'Default Ollama model for analysis'),
('ollama_batch_size', '10', 'Number of media items to process in one batch'),
('ollama_max_retries', '3', 'Maximum retry attempts for failed analysis'),
('ollama_timeout_seconds', '120', 'Timeout for Ollama API calls'),
('analysis_prompt_version', '"1.0"', 'Current version of analysis prompts'),
('enable_background_processing', 'true', 'Enable automatic background analysis'),
('qdrant_host', '"http://localhost:6333"', 'QDrant vector database endpoint'),
('embedding_model', '"nomic-embed-text"', 'Model used for generating embeddings'),
('recommendation_refresh_hours', '24', 'Hours between recommendation refreshes'),
('max_recommendations_per_user', '50', 'Maximum recommendations to store per user')

ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = CURRENT_TIMESTAMP;

-- Add some common genres (you can expand this list)
INSERT INTO genres (name) VALUES
('Action'), ('Adventure'), ('Animation'), ('Comedy'), ('Crime'),
('Documentary'), ('Drama'), ('Family'), ('Fantasy'), ('History'),
('Horror'), ('Music'), ('Mystery'), ('Romance'), ('Science Fiction'),
('TV Movie'), ('Thriller'), ('War'), ('Western'), ('Biography'),
('Musical'), ('Sport'), ('Film-Noir'), ('News'), ('Reality-TV'),
('Talk-Show'), ('Game-Show'), ('Short')

ON CONFLICT (name) DO NOTHING;

-- Add analysis tags including both original and Ollama-specific ones
INSERT INTO analysis_tags (tag_name, tag_category, description) VALUES
('fast-paced', 'pacing', 'Content with quick scene changes and high energy'),
('slow-burn', 'pacing', 'Content that builds tension gradually'),
('character-driven', 'narrative', 'Story focuses on character development'),
('plot-driven', 'narrative', 'Story focuses on events and action'),
('dark-themes', 'mood', 'Contains mature or heavy subject matter'),
('lighthearted', 'mood', 'Upbeat and entertaining content'),
('violence', 'content_warning', 'Contains violent scenes'),
('strong-language', 'content_warning', 'Contains profanity or strong language'),
('complex-plot', 'complexity', 'Intricate storyline requiring attention'),
('ensemble-cast', 'structure', 'Features multiple main characters'),
('thematic_analysis', 'analysis_type', 'Deep analysis of themes and meaning'),
('style_analysis', 'analysis_type', 'Analysis of visual and narrative style'),
('audience_analysis', 'analysis_type', 'Target audience and viewing context analysis'),
('emotional_analysis', 'analysis_type', 'Emotional journey and intensity analysis'),
('similarity_analysis', 'analysis_type', 'Content similarity and recommendation markers'),
('high_concept', 'complexity', 'Abstract or philosophical themes'),
('accessible', 'complexity', 'Easy to understand and follow'),
('atmospheric', 'style', 'Strong mood and atmosphere'),
('dialogue_heavy', 'style', 'Conversation-focused storytelling'),
('visual_spectacle', 'style', 'Emphasis on visual elements'),
('cerebral', 'audience', 'Appeals to intellectual viewers'),
('mainstream', 'audience', 'Broad appeal audience'),
('niche', 'audience', 'Specialized or cult following'),
('binge_worthy', 'viewing_context', 'Good for extended viewing sessions'),
('casual_viewing', 'viewing_context', 'Good for background or casual watching'),
('focused_viewing', 'viewing_context', 'Requires full attention')

ON CONFLICT (tag_name) DO NOTHING;
"""

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL database schema for Media Recommendation System with Ollama Integration"
    )
    
    parser.add_argument(
        '--host',
        default=os.getenv('POSTGRES_HOST', 'localhost'),
        help='PostgreSQL server host (default: localhost, env: POSTGRES_HOST)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('POSTGRES_PORT', '5432')),
        help='PostgreSQL server port (default: 5432, env: POSTGRES_PORT)'
    )
    
    parser.add_argument(
        '--username',
        default=os.getenv('POSTGRES_USERNAME', 'postgres'),
        help='Database username (default: postgres, env: POSTGRES_USERNAME)'
    )
    
    parser.add_argument(
        '--password',
        default=os.getenv('POSTGRES_PASSWORD'),
        help='Database password (env: POSTGRES_PASSWORD)'
    )
    
    parser.add_argument(
        '--database',
        default=os.getenv('POSTGRES_DATABASE', 'media_recommendation'),
        help='Database name (default: media_recommendation, env: POSTGRES_DATABASE)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of database (WARNING: This will drop existing data)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify schema without creating anything'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main initialization function."""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate required arguments
    if not args.password:
        logger.error("PostgreSQL password is required. Set via --password or POSTGRES_PASSWORD environment variable.")
        sys.exit(1)
    
    logger.info("Starting database initialization with Ollama integration...")
    logger.info(f"Target: {args.username}@{args.host}:{args.port}/{args.database}")
    
    # Initialize database handler
    db_init = DatabaseInitializer(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        database=args.database
    )
    
    try:
        # Connect to PostgreSQL server
        if not db_init.connect_to_postgres():
            sys.exit(1)
        
        # Handle force recreation
        if args.force and db_init.database_exists():
            logger.warning(f"Force flag set. Dropping existing database '{args.database}'")
            cursor = db_init.connection.cursor()
            cursor.execute(f'DROP DATABASE IF EXISTS "{args.database}"')
            cursor.close()
            logger.info(f"Dropped database '{args.database}'")
        
        # Create database if needed
        if not args.verify_only:
            if not db_init.create_database():
                sys.exit(1)
        
        # Connect to the target database
        if not db_init.connect_to_database():
            sys.exit(1)
        
        if args.verify_only:
            # Only verify the schema
            if db_init.verify_schema():
                logger.info("Schema verification completed successfully")
            else:
                logger.error("Schema verification failed")
                sys.exit(1)
        else:
            # Execute the schema creation
            logger.info("Executing database schema with Ollama integration...")
            sql_schema = get_sql_schema()
            
            if not db_init.execute_sql_file(sql_schema):
                logger.error("Failed to create database schema")
                sys.exit(1)
            
            # Verify the schema was created correctly
            if not db_init.verify_schema():
                logger.error("Schema verification failed after creation")
                sys.exit(1)
            
            logger.info("Database initialization completed successfully!")
            logger.info(f"Database '{args.database}' is ready for use with Ollama integration.")
            logger.info("Schema includes:")
            logger.info("  - Core media tables with jellyfin_etag support")
            logger.info("  - Ollama analysis and processing queue tables")
            logger.info("  - Media profiles and embeddings storage")
            logger.info("  - Enhanced analysis tags and configuration")
    
    except KeyboardInterrupt:
        logger.info("Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        sys.exit(1)
    finally:
        db_init.close_connection()

if __name__ == "__main__":
    main()