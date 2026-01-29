"""
OPTIMIZED PHOTO ALBUM APPLICATION - STREAMLIT OPTIMIZED VERSION
Simplified for easier debugging and fewer dependencies
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageFilter, ImageEnhance, ImageDraw
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import hashlib
import csv
import io
import time
import math
from dataclasses import dataclass, asdict
from enum import Enum
import random
import string
from collections import defaultdict, deque
import re
import os
import sys
import traceback
from contextlib import contextmanager
import concurrent.futures
from functools import lru_cache, wraps
import pickle
import gc
import warnings
import logging
from logging.handlers import RotatingFileHandler
import inspect

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Simplified configuration"""
    APP_NAME = "PhotoAlbum Pro"
    VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Files
    DB_FILE = DB_DIR / "album.db"
    
    # Settings
    THUMBNAIL_SIZE = (300, 300)
    ITEMS_PER_PAGE = 24
    GRID_COLUMNS = 4
    CACHE_TTL = 3600
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    @classmethod
    def init_directories(cls):
        """Initialize all required directories"""
        directories = [
            cls.DATA_DIR,
            cls.THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.CACHE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create sample data if none exists
        cls._create_sample_data()
    
    @classmethod
    def _create_sample_data(cls):
        """Create sample data for demo"""
        sample_dir = cls.DATA_DIR / "sample-person"
        sample_dir.mkdir(exist_ok=True)
        
        # Create a sample image if none exists
        sample_image = sample_dir / "sample.jpg"
        if not sample_image.exists():
            try:
                img = Image.new('RGB', (400, 300), color='#667eea')
                draw = ImageDraw.Draw(img)
                draw.ellipse((150, 50, 250, 150), fill='#ffffff')
                draw.text((160, 160), "Sample Photo", fill='#ffffff')
                img.save(sample_image)
                print(f"Created sample image at {sample_image}")
            except Exception as e:
                print(f"Could not create sample image: {e}")

# ============================================================================
# SIMPLIFIED CACHE SYSTEM
# ============================================================================
class SimpleCache:
    """Simple in-memory cache"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key, default=None):
        """Get cached value if not expired"""
        if key in self.cache:
            # Check if expired (simplified - no TTL in this version)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return default
    
    def set(self, key, value):
        """Set cache value"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        # Enforce max size
        if len(self.cache) > self.max_size:
            # Remove oldest 10%
            sorted_items = sorted(self.timestamps.items(), key=lambda x: x[1])
            for k, _ in sorted_items[:self.max_size // 10]:
                if k in self.cache:
                    del self.cache[k]
                if k in self.timestamps:
                    del self.timestamps[k]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self):
        """Get cache statistics"""
        hit_ratio = self.hits / max(1, self.hits + self.misses)
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': hit_ratio
        }

# ============================================================================
# SIMPLIFIED DATABASE MANAGER
# ============================================================================
class SimpleDatabaseManager:
    """Simplified database manager"""
    
    def __init__(self):
        self.db_path = Config.DB_FILE
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_database(self):
        """Initialize database with basic tables"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Images table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    created_date TIMESTAMP,
                    checksum TEXT UNIQUE,
                    thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # People table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY,
                    folder_name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    bio TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Album entries
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS album_entries (
                    entry_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    caption TEXT,
                    description TEXT,
                    location TEXT,
                    tags TEXT,
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (image_id),
                    FOREIGN KEY (person_id) REFERENCES people (person_id)
                )
                ''')
                
                # Comments
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                # Ratings
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating_value INTEGER CHECK (rating_value BETWEEN 1 AND 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entry_id, user_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                # Indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_person ON album_entries(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_created ON album_entries(created_at)')
                
                conn.commit()
                
                # Add sample person if none exists
                cursor.execute('SELECT COUNT(*) as count FROM people')
                if cursor.fetchone()['count'] == 0:
                    cursor.execute('''
                    INSERT INTO people (person_id, folder_name, display_name, bio)
                    VALUES (?, ?, ?, ?)
                    ''', (str(uuid.uuid4()), 'sample-person', 'Sample Person', 'Sample person for demo'))
                    conn.commit()
                
                print("Database initialized successfully")
                
        except Exception as e:
            print(f"Database initialization error: {e}")
            raise
    
    def execute_query(self, query, params=()):
        """Execute query and return results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if query.strip().upper().startswith('SELECT'):
                    return [dict(row) for row in cursor.fetchall()]
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            st.error(f"Query execution error: {e}")
            return []
    
    def scan_directory(self):
        """Scan for images and add to database"""
        try:
            # Get all person directories
            person_dirs = [d for d in Config.DATA_DIR.iterdir() 
                          if d.is_dir() and not d.name.startswith('.')]
            
            results = {
                'total': 0,
                'added': 0,
                'errors': []
            }
            
            for person_dir in person_dirs:
                # Check if person exists in database
                person_query = '''
                SELECT person_id FROM people WHERE folder_name = ?
                '''
                person_result = self.execute_query(person_query, (person_dir.name,))
                
                if not person_result:
                    # Add new person
                    person_id = str(uuid.uuid4())
                    add_person_query = '''
                    INSERT INTO people (person_id, folder_name, display_name, bio)
                    VALUES (?, ?, ?, ?)
                    '''
                    display_name = ' '.join(word.capitalize() for word in person_dir.name.split('-'))
                    self.execute_query(add_person_query, 
                                     (person_id, person_dir.name, display_name, f"Photos of {display_name}"))
                else:
                    person_id = person_result[0]['person_id']
                
                # Scan for images
                image_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                
                for image_path in image_files:
                    try:
                        # Calculate checksum
                        checksum = self._calculate_checksum(image_path)
                        
                        # Check if image already exists
                        check_query = 'SELECT image_id FROM images WHERE checksum = ?'
                        existing = self.execute_query(check_query, (checksum,))
                        
                        if existing:
                            continue
                        
                        # Get image info
                        with Image.open(image_path) as img:
                            stats = image_path.stat()
                            
                            # Create thumbnail
                            thumbnail_path = self._create_thumbnail(image_path)
                            
                            # Add to images table
                            image_id = str(uuid.uuid4())
                            add_image_query = '''
                            INSERT INTO images (image_id, filename, filepath, width, height, 
                                              format, created_date, checksum, thumbnail_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            '''
                            self.execute_query(add_image_query, (
                                image_id,
                                image_path.name,
                                str(image_path.relative_to(Config.DATA_DIR)),
                                img.width,
                                img.height,
                                img.format,
                                datetime.datetime.fromtimestamp(stats.st_mtime),
                                checksum,
                                str(thumbnail_path) if thumbnail_path else None
                            ))
                            
                            # Add album entry
                            entry_id = str(uuid.uuid4())
                            add_entry_query = '''
                            INSERT INTO album_entries (entry_id, image_id, person_id, caption, 
                                                      description, tags, created_by)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            '''
                            caption = image_path.stem.replace('_', ' ').replace('-', ' ').title()
                            tags = f"{person_dir.name},photo,memory"
                            
                            self.execute_query(add_entry_query, (
                                entry_id,
                                image_id,
                                person_id,
                                caption,
                                f"Photo of {display_name}",
                                tags,
                                'system'
                            ))
                            
                            results['added'] += 1
                            results['total'] += 1
                            
                    except Exception as e:
                        error_msg = f"Error processing {image_path.name}: {e}"
                        results['errors'].append(error_msg)
                        print(error_msg)
            
            return results
            
        except Exception as e:
            st.error(f"Directory scan error: {e}")
            return {'total': 0, 'added': 0, 'errors': [str(e)]}
    
    def _calculate_checksum(self, file_path):
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _create_thumbnail(self, image_path):
        """Create thumbnail for image"""
        try:
            thumbnail_dir = Config.THUMBNAIL_DIR / image_path.parent.relative_to(Config.DATA_DIR)
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            
            thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
            
            with Image.open(image_path) as img:
                # Handle rotation
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                
                # Create thumbnail
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                
                # Save
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
            
            return thumbnail_path
            
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None
    
    def get_people(self):
        """Get all people"""
        query = '''
        SELECT p.*, 
               (SELECT COUNT(*) FROM album_entries ae WHERE ae.person_id = p.person_id) as image_count,
               (SELECT AVG(r.rating_value) FROM ratings r 
                JOIN album_entries ae ON r.entry_id = ae.entry_id 
                WHERE ae.person_id = p.person_id) as avg_rating
        FROM people p
        ORDER BY p.display_name
        '''
        return self.execute_query(query)
    
    def get_entries_by_person(self, person_id, page=1, per_page=None):
        """Get entries for a person"""
        if per_page is None:
            per_page = Config.ITEMS_PER_PAGE
        
        offset = (page - 1) * per_page
        
        query = '''
        SELECT ae.*, i.filename, i.filepath, i.thumbnail_path,
               (SELECT AVG(rating_value) FROM ratings WHERE entry_id = ae.entry_id) as avg_rating,
               (SELECT COUNT(*) FROM comments WHERE entry_id = ae.entry_id) as comment_count
        FROM album_entries ae
        JOIN images i ON ae.image_id = i.image_id
        WHERE ae.person_id = ?
        ORDER BY ae.created_at DESC
        LIMIT ? OFFSET ?
        '''
        
        entries = self.execute_query(query, (person_id, per_page, offset))
        
        # Get total count
        count_query = 'SELECT COUNT(*) as count FROM album_entries WHERE person_id = ?'
        total_count = self.execute_query(count_query, (person_id,))[0]['count']
        
        return {
            'entries': entries,
            'total_count': total_count,
            'page': page,
            'total_pages': max(1, math.ceil(total_count / per_page))
        }
    
    def get_entry_details(self, entry_id):
        """Get detailed entry information"""
        query = '''
        SELECT ae.*, i.*, p.display_name,
               (SELECT AVG(rating_value) FROM ratings WHERE entry_id = ae.entry_id) as avg_rating,
               (SELECT COUNT(*) FROM ratings WHERE entry_id = ae.entry_id) as rating_count,
               (SELECT COUNT(*) FROM comments WHERE entry_id = ae.entry_id) as comment_count
        FROM album_entries ae
        JOIN images i ON ae.image_id = i.image_id
        JOIN people p ON ae.person_id = p.person_id
        WHERE ae.entry_id = ?
        '''
        
        results = self.execute_query(query, (entry_id,))
        return results[0] if results else None
    
    def search_entries(self, search_term):
        """Search entries"""
        if not search_term:
            return []
        
        query = '''
        SELECT ae.*, i.filename, i.filepath, i.thumbnail_path, p.display_name
        FROM album_entries ae
        JOIN images i ON ae.image_id = i.image_id
        JOIN people p ON ae.person_id = p.person_id
        WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ? OR p.display_name LIKE ?
        ORDER BY ae.created_at DESC
        LIMIT 100
        '''
        
        search_pattern = f'%{search_term}%'
        return self.execute_query(query, (search_pattern, search_pattern, search_pattern, search_pattern))
    
    def add_comment(self, entry_id, user_id, username, content):
        """Add comment to entry"""
        comment_id = str(uuid.uuid4())
        
        query = '''
        INSERT INTO comments (comment_id, entry_id, user_id, username, content)
        VALUES (?, ?, ?, ?, ?)
        '''
        
        return self.execute_query(query, (comment_id, entry_id, user_id, username, content))
    
    def add_rating(self, entry_id, user_id, rating):
        """Add or update rating"""
        # Check if rating exists
        check_query = 'SELECT rating_id FROM ratings WHERE entry_id = ? AND user_id = ?'
        existing = self.execute_query(check_query, (entry_id, user_id))
        
        if existing:
            # Update existing rating
            update_query = '''
            UPDATE ratings SET rating_value = ?, created_at = CURRENT_TIMESTAMP
            WHERE entry_id = ? AND user_id = ?
            '''
            return self.execute_query(update_query, (rating, entry_id, user_id))
        else:
            # Insert new rating
            rating_id = str(uuid.uuid4())
            insert_query = '''
            INSERT INTO ratings (rating_id, entry_id, user_id, rating_value)
            VALUES (?, ?, ?, ?)
            '''
            return self.execute_query(insert_query, (rating_id, entry_id, user_id, rating))
    
    def get_entry_comments(self, entry_id):
        """Get comments for an entry"""
        query = '''
        SELECT * FROM comments 
        WHERE entry_id = ?
        ORDER BY created_at DESC
        '''
        return self.execute_query(query, (entry_id,))
    
    def get_user_rating(self, entry_id, user_id):
        """Get user's rating for an entry"""
        query = '''
        SELECT rating_value FROM ratings 
        WHERE entry_id = ? AND user_id = ?
        '''
        results = self.execute_query(query, (entry_id, user_id))
        return results[0]['rating_value'] if results else None

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
def init_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
            'user_id': str(uuid.uuid4()),
            'username': 'Guest',
            'current_page': 'dashboard',
            'selected_person': None,
            'selected_image': None,
            'search_query': '',
            'page_num': 1,
            'favorites': set(),
            'theme': 'light'
        })

# ============================================================================
# SIMPLE ALBUM MANAGER
# ============================================================================
class SimpleAlbumManager:
    """Simplified album manager"""
    
    def __init__(self):
        try:
            Config.init_directories()
            self.db = SimpleDatabaseManager()
            self.cache = SimpleCache()
            print("Album manager initialized successfully")
        except Exception as e:
            print(f"Error initializing album manager: {e}")
            raise
    
    def get_image_data_url(self, image_path):
        """Get image as data URL"""
        try:
            if not image_path or not Path(image_path).exists():
                return ""
            
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                mime_type = "image/jpeg" if str(image_path).lower().endswith(('.jpg', '.jpeg')) else "image/png"
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""

# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    """UI components for the application"""
    
    @staticmethod
    def rating_stars(rating, max_rating=5):
        """Generate rating stars"""
        if rating is None or rating <= 0:
            rating = 0
        
        stars = []
        full_stars = int(rating)
        has_half_star = rating - full_stars >= 0.5
        
        for i in range(max_rating):
            if i < full_stars:
                stars.append('⭐')
            elif i == full_stars and has_half_star:
                stars.append('⭐')
            else:
                stars.append('☆')
        
        return f"{''.join(stars)} ({rating:.1f})"
    
    @staticmethod
    def tag_badges(tags, max_display=5):
        """Generate tag badges"""
        if not tags:
            return ""
        
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        displayed_tags = tags[:max_display]
        extra_count = len(tags) - max_display if len(tags) > max_display else 0
        
        badges = []
        for tag in displayed_tags:
            badges.append(f'<span style="background: #667eea; color: white; padding: 2px 8px; '
                         f'border-radius: 12px; font-size: 12px; margin: 2px; display: inline-block;">{tag}</span>')
        
        html = ' '.join(badges)
        
        if extra_count > 0:
            html += f'<span style="background: #f0f0f0; color: #666; padding: 2px 8px; border-radius: 12px; '
            html += f'font-size: 12px; margin: 2px; display: inline-block;">+{extra_count} more</span>'
        
        return html
    
    @staticmethod
    def pagination(current_page, total_pages, total_items):
        """Generate pagination controls"""
        if total_pages <= 1:
            return ""
        
        html = '<div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">'
        
        # Previous button
        if current_page > 1:
            html += f'<button onclick="window.streamlitSetSessionState(\'page_num\', {current_page - 1})" '
            html += 'style="margin: 0 10px; padding: 5px 15px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer;">◀ Prev</button>'
        else:
            html += '<button disabled style="margin: 0 10px; padding: 5px 15px; border: 1px solid #eee; border-radius: 4px; background: #f9f9f9; color: #ccc;">◀ Prev</button>'
        
        # Page info
        html += f'<span style="margin: 0 15px;">Page {current_page} of {total_pages} ({total_items} items)</span>'
        
        # Next button
        if current_page < total_pages:
            html += f'<button onclick="window.streamlitSetSessionState(\'page_num\', {current_page + 1})" '
            html += 'style="margin: 0 10px; padding: 5px 15px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer;">Next ▶</button>'
        else:
            html += '<button disabled style="margin: 0 10px; padding: 5px 15px; border: 1px solid #eee; border-radius: 4px; background: #f9f9f9; color: #ccc;">Next ▶</button>'
        
        html += '</div>'
        return html

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoAlbumApp:
    """Main application class"""
    
    def __init__(self):
        self.manager = None
        self.ui = UIComponents()
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page"""
        st.set_page_config(
            page_title=Config.APP_NAME,
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_sidebar(self):
        """Render sidebar"""
        with st.sidebar:
            st.title(f"📸 {Config.APP_NAME}")
            st.caption(f"v{Config.VERSION}")
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            
            nav_items = {
                "🏠 Dashboard": "dashboard",
                "👥 People": "people",
                "🖼️ Gallery": "gallery",
                "⭐ Favorites": "favorites",
                "🔍 Search": "search",
                "⚙️ Settings": "settings"
            }
            
            for label, page in nav_items.items():
                if st.button(label, use_container_width=True, key=f"nav_{page}"):
                    st.session_state.current_page = page
                    st.session_state.page_num = 1  # Reset pagination
                    st.rerun()
            
            st.divider()
            
            # Quick Actions
            st.subheader("Quick Actions")
            
            if st.button("🔄 Scan Directory", use_container_width=True):
                with st.spinner("Scanning for images..."):
                    if self.manager:
                        results = self.manager.db.scan_directory()
                        if results['added'] > 0:
                            st.success(f"Added {results['added']} new images!")
                        else:
                            st.info("No new images found")
                        st.rerun()
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                if self.manager:
                    self.manager.cache.clear()
                    st.success("Cache cleared!")
            
            st.divider()
            
            # Stats
            st.subheader("Stats")
            
            if self.manager:
                people = self.manager.db.get_people()
                total_images = sum(p.get('image_count', 0) for p in people)
                
                st.metric("People", len(people))
                st.metric("Total Images", total_images)
                
                # Cache stats
                cache_stats = self.manager.cache.get_stats()
                st.metric("Cache Hit Rate", f"{cache_stats['hit_ratio']:.0%}")
            
            st.divider()
            
            # Theme
            st.subheader("Theme")
            theme = st.selectbox("Select Theme", ["light", "dark", "blue"], 
                               index=["light", "dark", "blue"].index(st.session_state.theme))
            
            if theme != st.session_state.theme:
                st.session_state.theme = theme
                self.apply_theme(theme)
    
    def apply_theme(self, theme):
        """Apply theme"""
        if theme == "dark":
            st.markdown("""
            <style>
            .stApp {
                background-color: #1a1a1a;
                color: white;
            }
            .st-bw, .st-cm, .st-d6 {
                background-color: #2d2d2d;
            }
            </style>
            """, unsafe_allow_html=True)
        elif theme == "blue":
            st.markdown("""
            <style>
            .stApp {
                background-color: #f0f8ff;
            }
            </style>
            """, unsafe_allow_html=True)
        # Light theme is default
    
    def render_dashboard(self):
        """Render dashboard"""
        st.title("📊 Dashboard")
        
        if not self.manager:
            st.error("Application not initialized")
            return
        
        # Welcome message
        st.markdown(f"Welcome to **{Config.APP_NAME}**!")
        st.markdown("Manage your photo collection with ease.")
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            people = self.manager.db.get_people()
            st.metric("People", len(people))
        
        with col2:
            total_images = sum(p.get('image_count', 0) for p in people)
            st.metric("Total Images", total_images)
        
        with col3:
            # Get recent activity count (simplified)
            st.metric("Recent Activity", "24")
        
        with col4:
            # Average rating
            avg_rating = 0
            count = 0
            for person in people:
                if person.get('avg_rating'):
                    avg_rating += person['avg_rating']
                    count += 1
            avg_rating = avg_rating / max(1, count)
            st.metric("Avg Rating", f"{avg_rating:.1f}")
        
        st.divider()
        
        # Recent photos
        st.subheader("Recent Photos")
        
        # Get some sample entries
        sample_entries = []
        for person in people[:3]:  # First 3 people
            entries = self.manager.db.get_entries_by_person(person['person_id'], 1, 3)
            sample_entries.extend(entries.get('entries', []))
        
        if sample_entries:
            cols = st.columns(min(4, len(sample_entries)))
            for idx, entry in enumerate(sample_entries[:4]):
                with cols[idx % len(cols)]:
                    self.render_gallery_item(entry)
        else:
            st.info("No photos found. Scan your directory to add photos.")
        
        # Quick actions
        st.divider()
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Add Photos", use_container_width=True):
                st.info("Upload feature coming soon!")
        
        with col2:
            if st.button("👥 Manage People", use_container_width=True):
                st.session_state.current_page = 'people'
                st.rerun()
        
        with col3:
            if st.button("📊 View Reports", use_container_width=True):
                st.info("Reports coming soon!")
    
    def render_people(self):
        """Render people page"""
        st.title("👥 People")
        
        if not self.manager:
            st.error("Application not initialized")
            return
        
        # Search
        search_query = st.text_input("Search people...", key="people_search")
        
        # Get people
        people = self.manager.db.get_people()
        
        # Filter by search
        if search_query:
            people = [p for p in people if search_query.lower() in p['display_name'].lower()]
        
        # Display people
        if not people:
            st.info("No people found. Add some photos to get started.")
            return
        
        # Grid layout
        cols = st.columns(3)
        for idx, person in enumerate(people):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Person info
                    st.subheader(person['display_name'])
                    st.caption(f"Folder: {person['folder_name']}")
                    
                    # Stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Photos", person.get('image_count', 0))
                    with col2:
                        st.metric("Rating", f"{person.get('avg_rating', 0):.1f}" if person.get('avg_rating') else "N/A")
                    with col3:
                        st.metric("Added", "Today" if person.get('image_count', 0) > 0 else "Never")
                    
                    # Actions
                    if st.button("View Photos", key=f"view_{person['person_id']}", use_container_width=True):
                        st.session_state.selected_person = person['person_id']
                        st.session_state.current_page = 'gallery'
                        st.rerun()
        
        # Add new person
        st.divider()
        with st.expander("➕ Add New Person"):
            with st.form("add_person_form"):
                col1, col2 = st.columns(2)
                with col1:
                    folder_name = st.text_input("Folder Name*")
                    display_name = st.text_input("Display Name*")
                with col2:
                    bio = st.text_area("Bio", height=100)
                
                if st.form_submit_button("Add Person"):
                    if folder_name and display_name:
                        try:
                            # Create folder
                            person_dir = Config.DATA_DIR / folder_name
                            person_dir.mkdir(exist_ok=True)
                            
                            # Add to database
                            query = '''
                            INSERT INTO people (person_id, folder_name, display_name, bio)
                            VALUES (?, ?, ?, ?)
                            '''
                            self.manager.db.execute_query(query, 
                                                         (str(uuid.uuid4()), folder_name, display_name, bio))
                            st.success(f"Person '{display_name}' added!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding person: {e}")
                    else:
                        st.warning("Please fill in required fields (*)")
    
    def render_gallery(self):
        """Render gallery page"""
        st.title("🖼️ Gallery")
        
        if not self.manager:
            st.error("Application not initialized")
            return
        
        # Check if viewing specific person
        selected_person_id = st.session_state.get('selected_person')
        
        if selected_person_id:
            # Get person info
            people = self.manager.db.get_people()
            selected_person = next((p for p in people if p['person_id'] == selected_person_id), None)
            
            if selected_person:
                st.subheader(f"Photos of {selected_person['display_name']}")
                if st.button("← Back to All People"):
                    st.session_state.selected_person = None
                    st.rerun()
        
        # Search and filters
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                search_query = st.text_input("Search photos...", key="gallery_search")
            
            with col2:
                view_mode = st.selectbox("View", ["Grid", "List"], key="view_mode")
            
            with col3:
                sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Rating"], key="sort_by")
        
        # Get entries
        if selected_person_id:
            data = self.manager.db.get_entries_by_person(selected_person_id, st.session_state.page_num)
            entries = data['entries']
            total_pages = data['total_pages']
            total_count = data['total_count']
        else:
            # Get entries from all people (simplified)
            entries = []
            people = self.manager.db.get_people()
            for person in people:
                person_entries = self.manager.db.get_entries_by_person(person['person_id'], 1, 10)
                entries.extend(person_entries.get('entries', []))
            
            # Apply search
            if search_query:
                entries = [e for e in entries if search_query.lower() in e.get('caption', '').lower()]
            
            total_count = len(entries)
            total_pages = max(1, math.ceil(total_count / Config.ITEMS_PER_PAGE))
            
            # Pagination
            start_idx = (st.session_state.page_num - 1) * Config.ITEMS_PER_PAGE
            end_idx = start_idx + Config.ITEMS_PER_PAGE
            entries = entries[start_idx:end_idx]
        
        # Apply sorting
        if sort_by == "Oldest":
            entries.sort(key=lambda x: x.get('created_at', ''))
        elif sort_by == "Rating":
            entries.sort(key=lambda x: x.get('avg_rating', 0), reverse=True)
        else:  # Newest
            entries.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Display entries
        if not entries:
            st.info("No photos found. Add some photos or try a different search.")
            return
        
        if view_mode == "Grid":
            cols = st.columns(Config.GRID_COLUMNS)
            for idx, entry in enumerate(entries):
                with cols[idx % Config.GRID_COLUMNS]:
                    self.render_gallery_item(entry)
        else:
            # List view
            for entry in entries:
                self.render_gallery_item_list(entry)
        
        # Pagination
        if total_pages > 1:
            st.markdown(self.ui.pagination(st.session_state.page_num, total_pages, total_count), 
                       unsafe_allow_html=True)
    
    def render_gallery_item(self, entry):
        """Render gallery item in grid view"""
        with st.container(border=True):
            # Thumbnail
            thumbnail_path = entry.get('thumbnail_path')
            if thumbnail_path:
                full_path = Config.BASE_DIR / thumbnail_path
                if full_path.exists():
                    data_url = self.manager.get_image_data_url(full_path)
                    if data_url:
                        st.image(data_url, use_column_width=True)
                else:
                    st.markdown("🖼️")
            else:
                st.markdown("🖼️")
            
            # Caption
            caption = entry.get('caption', 'Untitled')
            if len(caption) > 30:
                caption = caption[:27] + '...'
            st.markdown(f"**{caption}**")
            
            # Rating
            rating = entry.get('avg_rating')
            if rating:
                st.caption(self.ui.rating_stars(rating))
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View", key=f"view_{entry['entry_id']}", use_container_width=True):
                    st.session_state.selected_image = entry['entry_id']
                    st.session_state.current_page = 'image_detail'
                    st.rerun()
            with col2:
                is_favorited = entry['entry_id'] in st.session_state.favorites
                if is_favorited:
                    if st.button("★", key=f"unfav_{entry['entry_id']}", use_container_width=True):
                        st.session_state.favorites.remove(entry['entry_id'])
                        st.rerun()
                else:
                    if st.button("☆", key=f"fav_{entry['entry_id']}", use_container_width=True):
                        st.session_state.favorites.add(entry['entry_id'])
                        st.rerun()
    
    def render_gallery_item_list(self, entry):
        """Render gallery item in list view"""
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # Thumbnail
                thumbnail_path = entry.get('thumbnail_path')
                if thumbnail_path:
                    full_path = Config.BASE_DIR / thumbnail_path
                    if full_path.exists():
                        data_url = self.manager.get_image_data_url(full_path)
                        if data_url:
                            st.image(data_url, width=100)
                    else:
                        st.markdown("🖼️")
                else:
                    st.markdown("🖼️")
            
            with col2:
                st.markdown(f"### {entry.get('caption', 'Untitled')}")
                
                if entry.get('description'):
                    st.markdown(f"*{entry.get('description')[:100]}...*")
                
                # Tags
                if entry.get('tags'):
                    st.markdown(self.ui.tag_badges(entry['tags']), unsafe_allow_html=True)
            
            with col3:
                # Rating
                rating = entry.get('avg_rating')
                if rating:
                    st.markdown(f"**{rating:.1f}** ⭐")
                
                # Actions
                if st.button("Details", key=f"det_{entry['entry_id']}", use_container_width=True):
                    st.session_state.selected_image = entry['entry_id']
                    st.session_state.current_page = 'image_detail'
                    st.rerun()
    
    def render_image_detail(self):
        """Render image detail page"""
        entry_id = st.session_state.get('selected_image')
        
        if not entry_id:
            st.error("No image selected")
            if st.button("Back to Gallery"):
                st.session_state.current_page = 'gallery'
                st.rerun()
            return
        
        if not self.manager:
            st.error("Application not initialized")
            return
        
        # Get entry details
        entry = self.manager.db.get_entry_details(entry_id)
        
        if not entry:
            st.error("Image not found")
            if st.button("Back to Gallery"):
                st.session_state.current_page = 'gallery'
                st.rerun()
            return
        
        # Back button
        if st.button("← Back"):
            st.session_state.current_page = 'gallery'
            st.rerun()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            filepath = entry.get('filepath')
            if filepath:
                image_path = Config.DATA_DIR / filepath
                if image_path.exists():
                    data_url = self.manager.get_image_data_url(image_path)
                    st.image(data_url, use_column_width=True)
                else:
                    st.error("Image file not found")
            else:
                st.error("No image available")
        
        with col2:
            # Image info
            st.title(entry.get('caption', 'Untitled'))
            st.markdown(f"👤 **{entry.get('display_name', 'Unknown')}**")
            
            # Rating
            avg_rating = entry.get('avg_rating', 0)
            rating_count = entry.get('rating_count', 0)
            st.markdown(f"⭐ **{avg_rating:.1f}** ({rating_count} ratings)")
            
            # User rating
            st.subheader("Your Rating")
            user_rating = self.manager.db.get_user_rating(entry_id, st.session_state.user_id)
            
            rating_cols = st.columns(5)
            for i in range(1, 6):
                with rating_cols[i-1]:
                    if st.button(f"{i} ⭐", key=f"rate_{i}", use_container_width=True):
                        self.manager.db.add_rating(entry_id, st.session_state.user_id, i)
                        st.success(f"Rated {i} stars!")
                        st.rerun()
            
            # Favorite button
            is_favorited = entry_id in st.session_state.favorites
            if is_favorited:
                if st.button("★ Remove Favorite", use_container_width=True):
                    st.session_state.favorites.remove(entry_id)
                    st.success("Removed from favorites!")
                    st.rerun()
            else:
                if st.button("☆ Add Favorite", use_container_width=True):
                    st.session_state.favorites.add(entry_id)
                    st.success("Added to favorites!")
                    st.rerun()
            
            # Metadata
            with st.expander("📊 Metadata"):
                if entry.get('description'):
                    st.markdown(f"**Description:** {entry['description']}")
                
                if entry.get('location'):
                    st.markdown(f"**Location:** {entry['location']}")
                
                if entry.get('created_at'):
                    st.markdown(f"**Date Added:** {entry['created_at']}")
                
                if entry.get('tags'):
                    st.markdown("**Tags:**")
                    st.markdown(self.ui.tag_badges(entry['tags'], max_display=10), unsafe_allow_html=True)
                
                if entry.get('width') and entry.get('height'):
                    st.markdown(f"**Dimensions:** {entry['width']} x {entry['height']}")
                
                if entry.get('format'):
                    st.markdown(f"**Format:** {entry['format']}")
        
        # Comments section
        st.divider()
        st.subheader("💬 Comments")
        
        # Add comment form
        with st.form(key="add_comment_form"):
            comment_text = st.text_area("Add a comment...", height=100, max_chars=500)
            submitted = st.form_submit_button("Post Comment")
            
            if submitted and comment_text.strip():
                self.manager.db.add_comment(
                    entry_id, 
                    st.session_state.user_id, 
                    st.session_state.username, 
                    comment_text.strip()
                )
                st.success("Comment added!")
                st.rerun()
        
        # Display comments
        comments = self.manager.db.get_entry_comments(entry_id)
        
        if comments:
            for comment in comments:
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**{comment.get('username', 'Anonymous')}**")
                        st.caption(comment.get('created_at', ''))
                    with col2:
                        st.markdown(comment.get('content', ''))
        else:
            st.info("No comments yet. Be the first to comment!")
    
    def render_favorites(self):
        """Render favorites page"""
        st.title("⭐ Favorites")
        
        if not self.manager:
            st.error("Application not initialized")
            return
        
        favorites = list(st.session_state.favorites)
        
        if not favorites:
            st.info("You haven't added any favorites yet.")
            st.markdown("Browse the gallery and click the ☆ button to add favorites!")
            return
        
        # Get favorite entries
        entries = []
        for entry_id in favorites:
            entry = self.manager.db.get_entry_details(entry_id)
            if entry:
                entries.append(entry)
        
        # Display favorites
        if not entries:
            st.info("Favorite entries not found (may have been deleted)")
            return
        
        cols = st.columns(Config.GRID_COLUMNS)
        for idx, entry in enumerate(entries):
            with cols[idx % Config.GRID_COLUMNS]:
                self.render_gallery_item(entry)
    
    def render_search(self):
        """Render search page"""
        st.title("🔍 Search")
        
        if not self.manager:
            st.error("Application not initialized")
            return
        
        # Search input
        search_query = st.text_input("Search for photos...", key="search_input")
        
        if search_query:
            with st.spinner("Searching..."):
                results = self.manager.db.search_entries(search_query)
                
                if results:
                    st.subheader(f"Found {len(results)} results")
                    
                    # Group by person
                    results_by_person = {}
                    for result in results:
                        person_id = result['person_id']
                        if person_id not in results_by_person:
                            results_by_person[person_id] = []
                        results_by_person[person_id].append(result)
                    
                    # Display results
                    for person_id, person_results in results_by_person.items():
                        person_name = person_results[0]['display_name']
                        st.markdown(f"### 👤 {person_name}")
                        
                        cols = st.columns(min(4, len(person_results)))
                        for idx, result in enumerate(person_results[:4]):
                            with cols[idx % len(cols)]:
                                self.render_gallery_item(result)
                        
                        if len(person_results) > 4:
                            st.caption(f"... and {len(person_results) - 4} more")
                        
                        st.divider()
                else:
                    st.info("No results found")
        else:
            st.info("Enter a search term to find photos")
    
    def render_settings(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["General", "Performance", "About"])
        
        with tabs[0]:
            st.subheader("General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Username", value=st.session_state.username, key="username_input")
                st.selectbox("Language", ["English", "Spanish", "French"], key="language")
            
            with col2:
                st.selectbox("Theme", ["light", "dark", "blue"], key="theme_select")
                st.checkbox("Show notifications", value=True, key="notifications")
                st.checkbox("Auto-refresh gallery", value=True, key="auto_refresh")
            
            if st.button("Save Settings", type="primary"):
                st.session_state.username = st.session_state.username_input
                st.session_state.theme = st.session_state.theme_select
                self.apply_theme(st.session_state.theme)
                st.success("Settings saved!")
        
        with tabs[1]:
            st.subheader("Performance Settings")
            
            if self.manager:
                cache_stats = self.manager.cache.get_stats()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Cache Size", cache_stats['size'])
                    st.metric("Cache Hits", cache_stats['hits'])
                
                with col2:
                    st.metric("Cache Misses", cache_stats['misses'])
                    st.metric("Hit Rate", f"{cache_stats['hit_ratio']:.1%}")
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Clear Cache", use_container_width=True):
                        self.manager.cache.clear()
                        st.success("Cache cleared!")
                        st.rerun()
                
                with col2:
                    if st.button("Optimize DB", use_container_width=True):
                        st.info("Database optimization would run here")
                
                with col3:
                    if st.button("Refresh Stats", use_container_width=True):
                        st.rerun()
            else:
                st.info("Performance stats not available")
        
        with tabs[2]:
            st.subheader("About")
            
            st.markdown(f"### {Config.APP_NAME}")
            st.markdown(f"**Version:** {Config.VERSION}")
            st.markdown("A simple yet powerful photo album management application.")
            
            st.divider()
            
            st.markdown("### System Information")
            st.markdown(f"**Python:** {sys.version.split()[0]}")
            st.markdown(f"**Streamlit:** {st.__version__}")
            st.markdown(f"**Platform:** {sys.platform}")
            
            if self.manager:
                db_size = os.path.getsize(self.manager.db.db_path) if self.manager.db.db_path.exists() else 0
                st.markdown(f"**Database Size:** {db_size / (1024*1024):.2f} MB")
    
    def render_main(self):
        """Main application renderer"""
        # Initialize session state
        init_session_state()
        
        # Initialize manager
        if self.manager is None:
            try:
                self.manager = SimpleAlbumManager()
                st.success("✅ Application initialized successfully!")
            except Exception as e:
                st.error(f"❌ Could not initialize application manager: {e}")
                st.info("""
                **Troubleshooting steps:**
                1. Make sure you have write permissions in the application directory
                2. Check that required directories can be created
                3. Try restarting the application
                
                **Quick fix:** Run in a directory where you have write permissions
                """)
                return
        
        # Render sidebar
        self.render_sidebar()
        
        # Render current page
        current_page = st.session_state.current_page
        
        try:
            if current_page == 'dashboard':
                self.render_dashboard()
            elif current_page == 'people':
                self.render_people()
            elif current_page == 'gallery':
                self.render_gallery()
            elif current_page == 'image_detail':
                self.render_image_detail()
            elif current_page == 'favorites':
                self.render_favorites()
            elif current_page == 'search':
                self.render_search()
            elif current_page == 'settings':
                self.render_settings()
            else:
                self.render_dashboard()
        
        except Exception as e:
            st.error(f"Error rendering page: {e}")
            with st.expander("Error Details"):
                st.exception(e)
            
            if st.button("Restart Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Footer
        st.divider()
        st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} • v{Config.VERSION}")

# ============================================================================
# RUN APPLICATION
# ============================================================================
def main():
    """Main entry point"""
    try:
        # Create and run app
        app = PhotoAlbumApp()
        app.render_main()
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        
        # Show detailed error
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())
        
        # Recovery options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restart Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📋 Copy Error Report"):
                error_report = f"""
                Application Error Report
                Time: {datetime.datetime.now()}
                Error: {str(e)}
                Traceback: {traceback.format_exc()}
                """
                st.code(error_report, language="text")

# Run the application
if __name__ == "__main__":
    main()
