"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH AI ENHANCEMENTS
Version: 3.1.0 - Optimized for Streamlit Performance
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search,
          AI Image Analysis, AI Style Transfer, AI Editing, Batch Processing
          WITH PERFORMANCE OPTIMIZATIONS
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageFilter, ImageEnhance
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import csv
import io
import time
import math
from dataclasses import dataclass, asdict
from enum import Enum
import random
import string
from collections import defaultdict
import re
import os
from contextlib import contextmanager
import functools
import pickle
import hashlib
import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import lru_cache
import asyncio
import warnings

# ============================================================================
# PERFORMANCE OPTIMIZATION IMPORTS
# ============================================================================
try:
    # Use numba for performance-critical numerical operations
    from numba import jit, njit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using fallback methods")

# Try to use joblib for caching heavy computations
try:
    from joblib import Memory
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Joblib not available for caching")

# ============================================================================
# AI IMPORTS WITH LAZY LOADING
# ============================================================================
AI_AVAILABLE = False
AI_MODULES = {}

def lazy_load_ai():
    """Lazy load AI modules only when needed"""
    global AI_AVAILABLE, AI_MODULES
    
    if not AI_AVAILABLE:
        try:
            # Import only when actually needed
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
            from torchvision import models
            
            # Store in dictionary for easy access
            AI_MODULES['torch'] = torch
            AI_MODULES['nn'] = nn
            AI_MODULES['transforms'] = transforms
            AI_MODULES['models'] = models
            
            from transformers import (
                AutoModelForSeq2SeqLM, 
                AutoTokenizer,
                BlipProcessor, 
                BlipForConditionalGeneration,
                CLIPProcessor, 
                CLIPModel,
                pipeline
            )
            
            AI_MODULES['AutoModelForSeq2SeqLM'] = AutoModelForSeq2SeqLM
            AI_MODULES['AutoTokenizer'] = AutoTokenizer
            AI_MODULES['BlipProcessor'] = BlipProcessor
            AI_MODULES['BlipForConditionalGeneration'] = BlipForConditionalGeneration
            AI_MODULES['CLIPProcessor'] = CLIPProcessor
            AI_MODULES['CLIPModel'] = CLIPModel
            AI_MODULES['pipeline'] = pipeline
            
            from diffusers import (
                StableDiffusionPipeline,
                StableDiffusionImg2ImgPipeline,
                StableDiffusionInpaintPipeline,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler
            )
            
            AI_MODULES['StableDiffusionPipeline'] = StableDiffusionPipeline
            AI_MODULES['StableDiffusionImg2ImgPipeline'] = StableDiffusionImg2ImgPipeline
            AI_MODULES['StableDiffusionInpaintPipeline'] = StableDiffusionInpaintPipeline
            AI_MODULES['EulerAncestralDiscreteScheduler'] = EulerAncestralDiscreteScheduler
            AI_MODULES['DPMSolverMultistepScheduler'] = DPMSolverMultistepScheduler
            
            import cv2
            from scipy import ndimage
            
            AI_MODULES['cv2'] = cv2
            AI_MODULES['ndimage'] = ndimage
            
            AI_AVAILABLE = True
            print("AI libraries loaded successfully")
        except ImportError:
            print("AI libraries not installed. Running in basic mode.")
            AI_AVAILABLE = False

# ============================================================================
# PERFORMANCE OPTIMIZATION DECORATORS AND UTILITIES
# ============================================================================

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def st_cache_data(ttl=3600, max_entries=100, persist="session"):
        """
        Enhanced caching decorator for Streamlit with TTL and size limits
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = PerformanceOptimizer._create_cache_key(func, args, kwargs)
                
                # Check if in session state
                cache_dict_name = f"_cache_{func.__name__}"
                if cache_dict_name not in st.session_state:
                    st.session_state[cache_dict_name] = {}
                
                cache_dict = st.session_state[cache_dict_name]
                
                # Check cache
                if cache_key in cache_dict:
                    cached_data, timestamp = cache_dict[cache_key]
                    if time.time() - timestamp < ttl:
                        return cached_data
                    else:
                        # Remove expired entry
                        del cache_dict[cache_key]
                
                # Generate new data
                result = func(*args, **kwargs)
                
                # Store in cache with timestamp
                cache_dict[cache_key] = (result, time.time())
                
                # Enforce max entries
                if len(cache_dict) > max_entries:
                    # Remove oldest entry
                    oldest_key = min(cache_dict.keys(), 
                                   key=lambda k: cache_dict[k][1])
                    del cache_dict[oldest_key]
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def _create_cache_key(func, args, kwargs):
        """Create a unique cache key"""
        # Use function name
        key_parts = [func.__module__, func.__name__]
        
        # Add args (skip self for methods)
        if args and hasattr(args[0], '__class__') and args[0].__class__.__name__ == func.__qualname__.split('.')[0]:
            args = args[1:]  # Skip self
        
        for arg in args:
            try:
                if isinstance(arg, (int, float, str, bool, type(None))):
                    key_parts.append(str(arg))
                elif isinstance(arg, (list, tuple)):
                    key_parts.append(str(len(arg)))
                elif isinstance(arg, dict):
                    key_parts.append(str(sorted(arg.keys())))
                else:
                    # For objects, use their id
                    key_parts.append(str(id(arg)))
            except:
                key_parts.append("unhashable")
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            try:
                if isinstance(v, (int, float, str, bool, type(None))):
                    key_parts.append(f"{k}:{v}")
                else:
                    key_parts.append(f"{k}:{type(v).__name__}")
            except:
                key_parts.append(f"{k}:unhashable")
        
        # Create hash
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    @njit(parallel=True, fastmath=True) if NUMBA_AVAILABLE else lambda x: x
    def fast_image_processing(image_array):
        """Optimized image processing using Numba"""
        if not NUMBA_AVAILABLE:
            return image_array
        
        # This is a placeholder for numba-optimized operations
        # In practice, you'd implement specific image processing operations
        result = np.empty_like(image_array)
        for i in prange(image_array.shape[0]):
            for j in prange(image_array.shape[1]):
                for k in prange(image_array.shape[2]):
                    result[i, j, k] = image_array[i, j, k] * 1.0
        return result

# ============================================================================
# SESSION STATE MANAGEMENT WITH TTL
# ============================================================================

class SessionStateManager:
    """Enhanced session state management with TTL and automatic cleanup"""
    
    def __init__(self):
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize session state with TTL tracking"""
        if '_state_ttl' not in st.session_state:
            st.session_state['_state_ttl'] = {}
        
        if '_state_last_access' not in st.session_state:
            st.session_state['_state_last_access'] = {}
        
        # Cleanup expired states periodically
        self._cleanup_expired()
    
    def set(self, key, value, ttl=None):
        """Set a value in session state with optional TTL"""
        st.session_state[key] = value
        
        if ttl:
            st.session_state['_state_ttl'][key] = ttl
            st.session_state['_state_last_access'][key] = time.time()
    
    def get(self, key, default=None):
        """Get a value from session state, checking TTL"""
        if key not in st.session_state:
            return default
        
        # Check TTL if exists
        if key in st.session_state['_state_ttl']:
            ttl = st.session_state['_state_ttl'][key]
            last_access = st.session_state['_state_last_access'].get(key, 0)
            
            if time.time() - last_access > ttl:
                # TTL expired, remove the key
                self.delete(key)
                return default
            
            # Update last access time
            st.session_state['_state_last_access'][key] = time.time()
        
        return st.session_state[key]
    
    def delete(self, key):
        """Delete a key from session state and TTL tracking"""
        if key in st.session_state:
            del st.session_state[key]
        
        if key in st.session_state['_state_ttl']:
            del st.session_state['_state_ttl'][key]
        
        if key in st.session_state['_state_last_access']:
            del st.session_state['_state_last_access'][key]
    
    def clear_expired(self):
        """Clear all expired session state entries"""
        keys_to_delete = []
        
        for key, ttl in st.session_state['_state_ttl'].items():
            if key in st.session_state['_state_last_access']:
                last_access = st.session_state['_state_last_access'][key]
                if time.time() - last_access > ttl:
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.delete(key)
    
    def _cleanup_expired(self):
        """Cleanup expired states (called periodically)"""
        # Run cleanup every 10 minutes
        if '_last_cleanup' not in st.session_state:
            st.session_state['_last_cleanup'] = 0
        
        if time.time() - st.session_state['_last_cleanup'] > 600:  # 10 minutes
            self.clear_expired()
            st.session_state['_last_cleanup'] = time.time()

# ============================================================================
# ASYNCHRONOUS TASK MANAGER
# ============================================================================

class AsyncTaskManager:
    """Manage asynchronous tasks to prevent UI blocking"""
    
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = {}
        self._results = {}
        self._lock = threading.Lock()
    
    def submit_task(self, task_id, func, *args, **kwargs):
        """Submit a task for background execution"""
        future = self.executor.submit(func, *args, **kwargs)
        
        with self._lock:
            self._futures[task_id] = {
                'future': future,
                'submitted_at': time.time(),
                'status': 'pending'
            }
        
        return task_id
    
    def check_result(self, task_id, timeout=None):
        """Check if a task is complete and get result"""
        with self._lock:
            if task_id not in self._futures:
                return None
            
            task_info = self._futures[task_id]
            
            if task_info['status'] == 'completed':
                return self._results.get(task_id)
            
            if task_info['future'].done():
                try:
                    result = task_info['future'].result(timeout=timeout)
                    task_info['status'] = 'completed'
                    self._results[task_id] = result
                    return result
                except Exception as e:
                    task_info['status'] = 'failed'
                    task_info['error'] = str(e)
                    return None
            
            return None
    
    def cleanup_old_tasks(self, max_age=3600):
        """Clean up old completed tasks"""
        current_time = time.time()
        tasks_to_remove = []
        
        with self._lock:
            for task_id, task_info in self._futures.items():
                if task_info['status'] == 'completed':
                    if current_time - task_info['submitted_at'] > max_age:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                if task_id in self._results:
                    del self._results[task_id]
                del self._futures[task_id]
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=False)

# ============================================================================
# OPTIMIZED CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Application configuration constants with performance optimizations"""
    APP_NAME = "MemoryVault Pro AI"
    VERSION = "3.1.0"
    
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    AI_MODELS_DIR = BASE_DIR / "ai_models"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Files and paths
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    CACHE_FILE = CACHE_DIR / "app_cache.pkl"
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Performance settings
    MAX_CACHE_SIZE = 1000
    CACHE_TTL = 3600  # 1 hour
    DB_CONNECTION_POOL_SIZE = 5
    IMAGE_PROCESSING_THREADS = 2
    LAZY_LOADING_ENABLED = True
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # AI Settings (lazy loaded)
    AI_ENABLED = False  # Will be set when AI is loaded
    
    @classmethod
    def init_directories(cls):
        """Create all necessary directories with optimized permissions"""
        directories = [
            cls.DATA_DIR,
            cls.THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.EXPORT_DIR,
            cls.AI_MODELS_DIR,
            cls.CACHE_DIR
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {str(e)}")
        
        # Initialize joblib cache if available
        if JOBLIB_AVAILABLE:
            global memory
            memory = Memory(cls.CACHE_DIR, verbose=0)
        
        # Check for AI availability
        lazy_load_ai()
        cls.AI_ENABLED = AI_AVAILABLE

# ============================================================================
# OPTIMIZED DATA MODELS
# ============================================================================

@dataclass(frozen=True)  # Frozen dataclasses are hashable for caching
class ImageMetadata:
    """Metadata for a single image - optimized with slots"""
    __slots__ = ['image_id', 'filename', 'filepath', 'file_size', 'dimensions', 
                 'format', 'created_date', 'modified_date', 'exif_data', 'checksum']
    
    image_id: str
    filename: str
    filepath: str
    file_size: int
    dimensions: Tuple[int, int]
    format: str
    created_date: datetime.datetime
    modified_date: datetime.datetime
    exif_data: Optional[Dict]
    checksum: str
    
    @classmethod
    @PerformanceOptimizer.st_cache_data(ttl=3600)
    def from_image(cls, image_path: Path) -> 'ImageMetadata':
        """Create metadata from image file with caching"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with Image.open(image_path) as img:
            stats = image_path.stat()
            return cls(
                image_id=str(uuid.uuid4()),
                filename=image_path.name,
                filepath=str(image_path.relative_to(Config.DATA_DIR)),
                file_size=stats.st_size,
                dimensions=img.size,
                format=img.format,
                created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
                modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
                exif_data=cls._extract_exif_fast(img),
                checksum=cls._calculate_checksum_fast(image_path)
            )
    
    @staticmethod
    def _extract_exif_fast(img: Image.Image) -> Optional[Dict]:
        """Fast EXIF extraction with limited fields"""
        try:
            exif = {}
            if hasattr(img, '_getexif') and img._getexif():
                raw_exif = img._getexif()
                # Only extract commonly used EXIF fields for performance
                important_tags = {
                    271: 'Make',
                    272: 'Model',
                    274: 'Orientation',
                    306: 'DateTime',
                    36867: 'DateTimeOriginal',
                    33434: 'ExposureTime',
                    33437: 'FNumber',
                    34855: 'ISOSpeedRatings',
                    37378: 'Flash',
                    37386: 'FocalLength'
                }
                
                for tag_id, value in raw_exif.items():
                    tag_name = important_tags.get(tag_id)
                    if tag_name:
                        try:
                            exif[tag_name] = str(value)
                        except:
                            pass
                
                return exif if exif else None
        except Exception:
            return None
    
    @staticmethod
    def _calculate_checksum_fast(image_path: Path) -> str:
        """Fast checksum calculation using file metadata"""
        try:
            # For performance, use a combination of file stats
            stats = image_path.stat()
            checksum_str = f"{stats.st_size}-{stats.st_mtime}-{stats.st_ctime}"
            return hashlib.md5(checksum_str.encode()).hexdigest()
        except Exception:
            # Fallback to file content hash
            hash_md5 = hashlib.md5()
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

# ============================================================================
# OPTIMIZED DATABASE MANAGEMENT WITH CONNECTION POOLING
# ============================================================================

class DatabaseConnectionPool:
    """Connection pool for database to reduce connection overhead"""
    
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections = []
        self._lock = threading.Lock()
    
    def get_connection(self):
        """Get a connection from the pool"""
        with self._lock:
            if self._connections:
                return self._connections.pop()
            elif len(self._connections) < self.pool_size:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for performance
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = -2000")  # 2MB cache
                return conn
            else:
                # Wait for a connection to be returned
                # In production, implement proper waiting logic
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode = WAL")
                return conn
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        with self._lock:
            if len(self._connections) < self.pool_size:
                self._connections.append(conn)
            else:
                conn.close()
    
    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._connections:
                conn.close()
            self._connections.clear()

class OptimizedDatabaseManager(DatabaseManager):
    """Optimized database manager with connection pooling"""
    
    def __init__(self, db_path: Path = None):
        super().__init__(db_path)
        self.connection_pool = DatabaseConnectionPool(
            self.db_path, 
            Config.DB_CONNECTION_POOL_SIZE
        )
    
    @contextmanager
    def get_connection(self):
        """Get database connection with pooling"""
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            st.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    conn.commit()
                    self.connection_pool.return_connection(conn)
                except:
                    conn.close()
    
    def _init_database(self):
        """Initialize database with performance optimizations"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable performance optimizations
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = -2000")  # 2MB cache
                cursor.execute("PRAGMA temp_store = MEMORY")
                cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
                
                # Create tables with optimized schema
                # ... (rest of table creation as before, but consider adding WITHOUT ROWID for performance)
                
                conn.commit()
                st.success("✅ Database initialized successfully!")
                
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise

# ============================================================================
# OPTIMIZED IMAGE PROCESSING
# ============================================================================

class OptimizedImageProcessor(ImageProcessor):
    """Optimized image processor with caching and performance improvements"""
    
    # Cache for thumbnails
    _thumbnail_cache = {}
    _thumbnail_lock = threading.Lock()
    
    @classmethod
    @PerformanceOptimizer.st_cache_data(ttl=86400)  # Cache thumbnails for 24 hours
    def create_thumbnail(cls, image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail with caching and optimization"""
        if not image_path.exists():
            return None
            
        thumbnail_dir = thumbnail_dir or Config.THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
        
        # Check cache first
        cache_key = f"thumb_{image_path.stat().st_mtime}_{image_path.stat().st_size}"
        with cls._thumbnail_lock:
            if cache_key in cls._thumbnail_cache:
                cached_path = cls._thumbnail_cache[cache_key]
                if cached_path.exists():
                    return cached_path
        
        try:
            # Use low-memory approach for large images
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Open with minimal loading
                with Image.open(image_path) as img:
                    # Get image info without loading full image
                    img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                    
                    # Convert mode efficiently
                    if img.mode not in ('RGB', 'L'):
                        if img.mode == 'RGBA':
                            # Create white background for RGBA
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                            img = background
                        else:
                            img = img.convert('RGB')
                    
                    # Optimize save parameters
                    save_kwargs = {
                        'format': 'JPEG',
                        'quality': 85,
                        'optimize': True,
                        'progressive': True,  # Progressive JPEG for faster loading
                        'subsampling': -1  # Keep all color information
                    }
                    
                    img.save(thumbnail_path, **save_kwargs)
            
            # Cache the thumbnail path
            with cls._thumbnail_lock:
                cls._thumbnail_cache[cache_key] = thumbnail_path
                # Limit cache size
                if len(cls._thumbnail_cache) > 100:
                    # Remove oldest (simplified)
                    del cls._thumbnail_cache[next(iter(cls._thumbnail_cache))]
            
            return thumbnail_path
            
        except Exception as e:
            st.error(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_image_data_url_cached(image_path: Path) -> str:
        """Get image data URL with caching"""
        if not image_path.exists():
            return ""
        
        # Create cache key
        cache_key = f"dataurl_{image_path.stat().st_mtime}_{image_path.stat().st_size}"
        
        # Check session cache first
        if 'image_data_url_cache' not in st.session_state:
            st.session_state.image_data_url_cache = {}
        
        if cache_key in st.session_state.image_data_url_cache:
            return st.session_state.image_data_url_cache[cache_key]
        
        try:
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                mime_type = Image.MIME.get(image_path.suffix.lower(), 'image/jpeg')
                data_url = f"data:{mime_type};base64,{encoded}"
                
                # Cache in session state (limited size)
                cache = st.session_state.image_data_url_cache
                cache[cache_key] = data_url
                
                # Limit cache size
                if len(cache) > 50:
                    # Remove first key (FIFO)
                    del cache[next(iter(cache))]
                
                return data_url
        except Exception as e:
            return ""

# ============================================================================
# OPTIMIZED ALBUM MANAGER WITH PERFORMANCE IMPROVEMENTS
# ============================================================================

class OptimizedAlbumManager(AlbumManager):
    """Optimized album manager with performance improvements"""
    
    def __init__(self):
        super().__init__()
        self.db = OptimizedDatabaseManager()
        self.cache = CacheManager()
        self.image_processor = OptimizedImageProcessor()
        self.session_manager = SessionStateManager()
        self.async_manager = AsyncTaskManager(max_workers=Config.IMAGE_PROCESSING_THREADS)
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize optimized session state"""
        # Use session manager for state with TTL
        defaults = {
            'current_page': ('dashboard', 3600),  # 1 hour TTL
            'selected_person': (None, 1800),      # 30 minutes TTL
            'selected_image': (None, 1800),       # 30 minutes TTL
            'search_query': ('', 900),            # 15 minutes TTL
            'view_mode': ('grid', 3600),
            'sort_by': ('date', 3600),
            'sort_order': ('desc', 3600),
            'selected_tags': ([], 1800),
            'user_id': (str(uuid.uuid4()), 86400),  # 24 hours TTL
            'username': ('Guest', 86400),
            'user_role': (UserRoles.VIEWER.value, 86400),
            'favorites': (set(), 3600),
            'recently_viewed': ([], 1800),
            'toc_page': (1, 900),
            'gallery_page': (1, 900),
            'show_directory_info': (True, 3600)
        }
        
        for key, (default_value, ttl) in defaults.items():
            if key not in st.session_state:
                self.session_manager.set(key, default_value, ttl)
        
        # Initialize performance-related states
        if '_performance_cache' not in st.session_state:
            st.session_state._performance_cache = {}
    
    @PerformanceOptimizer.st_cache_data(ttl=300)  # Cache for 5 minutes
    def get_person_stats(self, person_id: str) -> Dict:
        """Get statistics for a person with enhanced caching"""
        return super().get_person_stats(person_id)
    
    @PerformanceOptimizer.st_cache_data(ttl=60)  # Cache for 1 minute
    def get_entries_by_person(self, person_id: str, page: int = 1, search_query: str = None) -> Dict:
        """Get entries with pagination and caching"""
        return super().get_entries_by_person(person_id, page, search_query)
    
    def scan_directory_async(self, data_dir: Path = None):
        """Scan directory asynchronously to prevent UI blocking"""
        task_id = f"scan_{datetime.datetime.now().timestamp()}"
        
        def scan_task():
            return super().scan_directory(data_dir)
        
        self.async_manager.submit_task(task_id, scan_task)
        return task_id
    
    def batch_process_images(self, image_paths: List[Path], operation: str, **kwargs):
        """Batch process images asynchronously"""
        task_id = f"batch_{operation}_{datetime.datetime.now().timestamp()}"
        
        def batch_task():
            results = []
            for img_path in image_paths:
                if operation == 'analyze':
                    result = self.analyze_image_with_ai(img_path) if hasattr(self, 'analyze_image_with_ai') else {}
                elif operation == 'enhance':
                    result = self.apply_ai_modification(img_path, 'auto_enhance') if hasattr(self, 'apply_ai_modification') else None
                results.append((img_path, result))
            return results
        
        self.async_manager.submit_task(task_id, batch_task)
        return task_id

# ============================================================================
# OPTIMIZED UI COMPONENTS WITH LAZY RENDERING
# ============================================================================

class OptimizedUIComponents(UIComponents):
    """Optimized UI components with lazy rendering and performance improvements"""
    
    @staticmethod
    def lazy_image_display(image_url, width=None, caption="", use_column_width=False):
        """Lazy load images to improve page load time"""
        # Use Streamlit's native lazy loading with optimized parameters
        return st.image(
            image_url,
            width=width,
            caption=caption,
            use_column_width=use_column_width,
            output_format="JPEG",
            clamp=False
        )
    
    @staticmethod
    def virtual_scroll_container(items, render_item_callback, batch_size=20, height=600):
        """
        Virtual scroll container for large lists
        Only renders visible items to improve performance
        """
        if not items:
            st.info("No items to display")
            return
        
        # Create a session state for scroll position
        scroll_key = f"scroll_{hash(str(items[:10]))}"
        if scroll_key not in st.session_state:
            st.session_state[scroll_key] = 0
        
        # Calculate visible range
        start_idx = st.session_state[scroll_key]
        end_idx = min(start_idx + batch_size, len(items))
        
        # Display current batch
        for i in range(start_idx, end_idx):
            render_item_callback(items[i], i)
        
        # Navigation controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if start_idx > 0 and st.button("◀ Previous", key=f"prev_{scroll_key}"):
                st.session_state[scroll_key] = max(0, start_idx - batch_size)
                st.rerun()
        
        with col2:
            st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(items)}")
        
        with col3:
            if end_idx < len(items) and st.button("Next ▶", key=f"next_{scroll_key}"):
                st.session_state[scroll_key] = end_idx
                st.rerun()
    
    @staticmethod
    def optimized_dataframe(data, key=None):
        """Display DataFrame with optimizations"""
        if isinstance(data, list) and len(data) > 1000:
            # For large datasets, show a sample first
            st.warning(f"Showing first 1000 of {len(data)} rows. Use filters to reduce data.")
            data = data[:1000]
        
        # Use Streamlit's optimized dataframe with specified height
        return st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
            column_config=None,
            column_order=None,
            height=400
        )

# ============================================================================
# MAIN APPLICATION WITH PERFORMANCE OPTIMIZATIONS
# ============================================================================

class OptimizedPhotoAlbumApp(PhotoAlbumApp):
    """Optimized photo album application with performance improvements"""
    
    def __init__(self):
        self.performance_mode = st.sidebar.checkbox("Enable Performance Mode", value=True)
        
        if self.performance_mode:
            # Use optimized components
            self.manager = OptimizedAlbumManager()
            self.ui_components = OptimizedUIComponents()
            
            # Set performance-related configurations
            Config.ITEMS_PER_PAGE = 12  # Reduced for better performance
            Config.GRID_COLUMNS = 3     # Reduced for better rendering
            Config.LAZY_LOADING_ENABLED = True
        else:
            # Use original components
            super().__init__()
        
        self.setup_page_config()
        self.check_initialization()
    
    def setup_page_config(self):
        """Configure Streamlit page with performance optimizations"""
        st.set_page_config(
            page_title=Config.APP_NAME,
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-album',
                'Report a bug': 'https://github.com/yourusername/photo-album/issues',
                'About': f"# {Config.APP_NAME} v{Config.VERSION}\nOptimized for performance"
            }
        )
        
        # Add performance CSS
        st.markdown("""
        <style>
        /* Performance optimizations */
        .stImage img {
            max-width: 100%;
            height: auto;
        }
        
        /* Reduce animations for performance */
        * {
            animation-duration: 0.01s !important;
            transition-duration: 0.01s !important;
        }
        
        /* Optimize scrolling */
        .element-container {
            contain: content;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_gallery_page_optimized(self):
        """Optimized gallery page with virtual scrolling"""
        st.title("📁 Gallery (Optimized)")
        
        # Performance mode toggle
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Clear Cache", help="Clear all caches for fresh data"):
                self.manager.cache.clear()
                if hasattr(self.manager, 'session_manager'):
                    self.manager.session_manager.clear_expired()
                st.success("Cache cleared!")
                st.rerun()
        
        # Get entries with pagination
        selected_person_id = st.session_state.get('selected_person')
        page = st.session_state.get('gallery_page', 1)
        
        with st.spinner("Loading gallery..."):
            if selected_person_id:
                data = self.manager.get_entries_by_person(selected_person_id, page)
            else:
                # Get all entries with pagination
                data = self._get_all_entries_paginated(page)
        
        # Use virtual scrolling for large datasets
        if data['total_count'] > 50:
            st.info(f"Using virtual scrolling for {data['total_count']} items")
            
            def render_item(entry, idx):
                with st.container(border=True):
                    self.ui_components._render_gallery_item_optimized(entry)
            
            self.ui_components.virtual_scroll_container(
                data['entries'],
                render_item,
                batch_size=Config.ITEMS_PER_PAGE,
                height=800
            )
        else:
            # Regular grid for small datasets
            cols = st.columns(Config.GRID_COLUMNS)
            for idx, entry in enumerate(data['entries']):
                with cols[idx % Config.GRID_COLUMNS]:
                    self.ui_components._render_gallery_item_optimized(entry)
    
    def _get_all_entries_paginated(self, page=1):
        """Get paginated entries with caching"""
        cache_key = f"all_entries_page_{page}"
        
        @PerformanceOptimizer.st_cache_data(ttl=60)
        def fetch_entries(page):
            offset = (page - 1) * Config.ITEMS_PER_PAGE
            
            with self.manager.db.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(f'''
                SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                       (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                ORDER BY ae.created_at DESC
                LIMIT {Config.ITEMS_PER_PAGE} OFFSET {offset}
                ''')
                
                entries = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute('SELECT COUNT(*) FROM album_entries')
                total_count = cursor.fetchone()[0]
                total_pages = max(1, math.ceil(total_count / Config.ITEMS_PER_PAGE))
                
                return {
                    'entries': entries,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'current_page': page
                }
        
        return fetch_entries(page)

# ============================================================================
# ADDITIONAL PERFORMANCE OPTIMIZATIONS
# ============================================================================

class ImageProcessingOptimizer:
    """Optimize image processing operations"""
    
    @staticmethod
    @lru_cache(maxsize=100)
    def get_image_dimensions_cached(image_path: str) -> Tuple[int, int]:
        """Get image dimensions with caching"""
        try:
            with Image.open(image_path) as img:
                return img.size
        except:
            return (0, 0)
    
    @staticmethod
    def process_images_batch(image_paths: List[Path], operation: str):
        """Process multiple images in batch for efficiency"""
        if JOBLIB_AVAILABLE and len(image_paths) > 5:
            # Use parallel processing for large batches
            results = Parallel(n_jobs=Config.IMAGE_PROCESSING_THREADS)(
                delayed(ImageProcessingOptimizer._process_single_image)(path, operation)
                for path in image_paths
            )
            return results
        else:
            # Sequential processing for small batches
            return [ImageProcessingOptimizer._process_single_image(path, operation) 
                   for path in image_paths]
    
    @staticmethod
    def _process_single_image(image_path: Path, operation: str):
        """Process a single image"""
        try:
            with Image.open(image_path) as img:
                if operation == 'thumbnail':
                    return OptimizedImageProcessor.create_thumbnail(image_path)
                elif operation == 'analyze':
                    # Basic analysis
                    return {
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format
                    }
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

class MemoryManager:
    """Manage memory usage and cleanup"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage"""
        # Clear unused caches
        if 'image_data_url_cache' in st.session_state:
            if len(st.session_state.image_data_url_cache) > 100:
                # Clear half of the cache
                keys = list(st.session_state.image_data_url_cache.keys())[:50]
                for key in keys:
                    del st.session_state.image_data_url_cache[key]
        
        # Clear large DataFrames from session state
        for key in list(st.session_state.keys()):
            if isinstance(st.session_state[key], pd.DataFrame):
                if st.session_state[key].memory_usage().sum() > 100 * 1024 * 1024:  # 100MB
                    del st.session_state[key]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if hasattr(st, 'cache_data'):
            # Clear old Streamlit caches
            pass
    
    @staticmethod
    def monitor_memory(threshold_mb=500):
        """Monitor memory and warn if usage is high"""
        memory_mb = MemoryManager.get_memory_usage()
        
        if memory_mb > threshold_mb:
            st.warning(f"High memory usage: {memory_mb:.1f}MB")
            
            if st.button("Optimize Memory"):
                MemoryManager.optimize_memory()
                st.success("Memory optimized!")
                st.rerun()

# ============================================================================
# MAIN APPLICATION ENTRY POINT WITH PERFORMANCE MONITORING
# ============================================================================

def main_optimized():
    """Optimized main application entry point"""
    try:
        # Initialize performance monitoring
        if 'performance_start_time' not in st.session_state:
            st.session_state.performance_start_time = time.time()
        
        # Show performance stats in sidebar
        with st.sidebar.expander("📊 Performance Stats"):
            elapsed_time = time.time() - st.session_state.performance_start_time
            st.metric("Uptime", f"{elapsed_time:.0f}s")
            
            # Memory usage
            memory_mb = MemoryManager.get_memory_usage()
            st.metric("Memory", f"{memory_mb:.1f}MB")
            
            if st.button("Optimize Memory"):
                MemoryManager.optimize_memory()
                st.rerun()
        
        # Initialize optimized app
        app = OptimizedPhotoAlbumApp()
        
        # Check initialization
        if not app.initialized:
            st.error("Application failed to initialize. Check directory permissions.")
            return
        
        # Render main app
        app.render_main()
        
        # Periodically optimize memory
        if 'last_memory_optimization' not in st.session_state:
            st.session_state.last_memory_optimization = time.time()
        
        if time.time() - st.session_state.last_memory_optimization > 300:  # Every 5 minutes
            MemoryManager.optimize_memory()
            st.session_state.last_memory_optimization = time.time()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
        with st.expander("Error Details"):
            st.exception(e)
        
        if st.button("Try Again"):
            st.rerun()

# ============================================================================
# RUN OPTIMIZED APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Performance mode selection
    st.sidebar.title("⚡ Performance Mode")
    
    performance_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Optimized", "Balanced", "Compatibility"],
        help="Optimized: Best performance, Balanced: Good performance, Compatibility: All features"
    )
    
    # Configure based on mode
    if performance_mode == "Optimized":
        Config.ITEMS_PER_PAGE = 12
        Config.GRID_COLUMNS = 3
        Config.LAZY_LOADING_ENABLED = True
        main_optimized()
    elif performance_mode == "Balanced":
        Config.ITEMS_PER_PAGE = 20
        Config.GRID_COLUMNS = 4
        Config.LAZY_LOADING_ENABLED = True
        main_optimized()
    else:  # Compatibility
        # Run original app
        Config.ITEMS_PER_PAGE = 20
        Config.GRID_COLUMNS = 4
        Config.LAZY_LOADING_ENABLED = False
        
        # Import and run original main
        from original_app import main
        main()
