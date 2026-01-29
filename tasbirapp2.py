"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH AI ENHANCEMENTS
Version: 4.0.0 - Optimized & Enhanced
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search,
          AI Image Analysis, AI Style Transfer, AI Editing, Batch Processing,
          Optimized Caching, Lazy Loading, Resource Management, Background Processing
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
import sys
import gc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from functools import wraps, lru_cache
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================
try:
    import numba
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not installed. Install with: pip install numba")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not installed. Install with: pip install psutil")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("joblib not installed. Install with: pip install joblib")

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Dask not installed. Install with: pip install dask")

# ============================================================================
# AI IMPORTS (LAZY LOADING)
# ============================================================================
class LazyImport:
    """Lazy import wrapper to reduce startup time"""
    def __init__(self, module_name, class_name=None):
        self.module_name = module_name
        self.class_name = class_name
        self.module = None
    
    def __getattr__(self, name):
        if self.module is None:
            self.module = __import__(self.module_name)
            if self.class_name:
                self.module = getattr(self.module, self.class_name)
        return getattr(self.module, name)

# Lazy load heavy AI libraries
torch = LazyImport('torch')
torchvision = LazyImport('torchvision')
transformers = LazyImport('transformers')
diffusers = LazyImport('diffusers')

AI_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BlipProcessor,
        BlipForConditionalGeneration,
        CLIPProcessor,
        CLIPModel,
        pipeline
    )
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler
    )
    import cv2
    from scipy import ndimage
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI libraries not installed. Running in basic mode.")

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
class Config:
    """Application configuration constants with optimized defaults"""
    APP_NAME = "MemoryVault Pro AI"
    VERSION = "4.0.0"
    
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
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # Performance settings
    MAX_WORKERS = 4  # Thread pool workers
    BATCH_SIZE = 50  # Database batch operations
    IMAGE_PREFETCH_COUNT = 10  # Prefetch images
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    SESSION_STATE_TTL = 7200  # 2 hours
    CACHE_MAX_SIZE = 1000  # Max cache entries
    IMAGE_CACHE_MAX_SIZE = 50  # Max images in memory
    
    # AI Settings
    AI_ENABLED = AI_AVAILABLE
    AI_MODEL_PRELOAD = False  # Lazy load AI models
    AI_BATCH_SIZE = 4  # Batch size for AI processing
    
    # Resource limits
    MAX_MEMORY_USAGE = 2 * 1024 * 1024 * 1024  # 2GB
    MAX_CPU_PERCENT = 80  # 80% CPU usage limit
    
    # Background processing
    BACKGROUND_PROCESSING_ENABLED = True
    BACKGROUND_QUEUE_MAX_SIZE = 100
    
    # Lazy loading thresholds
    LAZY_LOAD_THRESHOLD = 100  # Number of items before lazy loading
    
    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
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
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create sample structure if no data exists
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()
    
    @classmethod
    def create_sample_structure(cls):
        """Create sample directory structure with example photos"""
        sample_people = ["john-smith", "sarah-johnson", "michael-brown"]
        for person in sample_people:
            person_dir = cls.DATA_DIR / person
            person_dir.mkdir(exist_ok=True)
            # Create a sample README file
            readme_file = person_dir / "README.txt"
            readme_file.write_text(f"Photos of {person.replace('-', ' ').title()}\nAdd your photos here!")
            # Add a simple sample image
            sample_image_path = person_dir / "sample.jpg"
            if not sample_image_path.exists():
                try:
                    from PIL import Image, ImageDraw
                    # Create image with gradient background
                    img = Image.new('RGB', (400, 300), color='#667eea')
                    draw = ImageDraw.Draw(img)
                    # Draw a simple person icon
                    draw.ellipse((150, 50, 250, 150), fill='#ffffff', outline='#4a5568')
                    draw.rectangle((150, 150, 250, 280), fill='#ffffff', outline='#4a5568')
                    # Add text
                    draw.text((120, 250), person.replace('-', ' ').title(), fill='#2d3748')
                    img.save(sample_image_path)
                    print(f"Created sample image for {person}")
                except Exception as e:
                    print(f"Could not create sample image for {person}: {str(e)}")

class UserRoles(Enum):
    """User permission levels"""
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"

# ============================================================================
# PERFORMANCE OPTIMIZED UTILITY FUNCTIONS
# ============================================================================
if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def fast_image_stats(img_array):
        """Fast image statistics calculation using numba"""
        mean_brightness = np.mean(img_array)
        std_contrast = np.std(img_array)
        return mean_brightness, std_contrast
    
    @njit(parallel=True)
    def fast_color_balance(img_array):
        """Fast color balance analysis"""
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        return r_mean, g_mean, b_mean
    
    @njit(parallel=True)
    def fast_histogram_equalization(img_array):
        """Fast histogram equalization"""
        result = np.zeros_like(img_array, dtype=np.uint8)
        for c in range(img_array.shape[2]):
            channel = img_array[:, :, c]
            hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            result[:, :, c] = np.interp(channel.flatten(), np.arange(256), cdf_normalized).reshape(channel.shape)
        return result
else:
    def fast_image_stats(img_array):
        """Fallback image statistics without numba"""
        mean_brightness = np.mean(img_array)
        std_contrast = np.std(img_array)
        return mean_brightness, std_contrast
    
    def fast_color_balance(img_array):
        """Fallback color balance without numba"""
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        return r_mean, g_mean, b_mean
    
    def fast_histogram_equalization(img_array):
        """Fallback histogram equalization"""
        from PIL import ImageOps
        img = Image.fromarray(img_array)
        img_eq = ImageOps.equalize(img)
        return np.array(img_eq)

def memory_efficient_image_loader(image_path, max_size=None):
    """Memory-efficient image loading with optional resizing"""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img.copy()
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def batch_process(func, items, max_workers=None):
    """Efficient batch processing with thread pool"""
    max_workers = max_workers or Config.MAX_WORKERS
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in as_completed(future_to_item):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing item: {str(e)}")
    return results

# ============================================================================
# SESSION STATE MANAGER WITH TTL
# ============================================================================
class SessionStateManager:
    """Enhanced session state manager with TTL and cleanup"""
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state with timestamps"""
        if 'session_state_timestamps' not in st.session_state:
            st.session_state.session_state_timestamps = {}
        if 'session_state_ttl' not in st.session_state:
            st.session_state.session_state_ttl = {}
    
    def set(self, key, value, ttl=None):
        """Set session state with optional TTL"""
        st.session_state[key] = value
        st.session_state.session_state_timestamps[key] = time.time()
        if ttl:
            st.session_state.session_state_ttl[key] = ttl
    
    def get(self, key, default=None):
        """Get session state with TTL check"""
        if key not in st.session_state:
            return default
        
        # Check TTL
        if key in st.session_state.session_state_timestamps:
            timestamp = st.session_state.session_state_timestamps[key]
            ttl = st.session_state.session_state_ttl.get(key, Config.SESSION_STATE_TTL)
            if time.time() - timestamp > ttl:
                # Expired
                del st.session_state[key]
                del st.session_state.session_state_timestamps[key]
                if key in st.session_state.session_state_ttl:
                    del st.session_state.session_state_ttl[key]
                return default
        
        return st.session_state[key]
    
    def clear_expired(self):
        """Clear all expired session state entries"""
        current_time = time.time()
        expired_keys = []
        for key, timestamp in st.session_state.session_state_timestamps.items():
            ttl = st.session_state.session_state_ttl.get(key, Config.SESSION_STATE_TTL)
            if current_time - timestamp > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in st.session_state:
                del st.session_state[key]
            del st.session_state.session_state_timestamps[key]
            if key in st.session_state.session_state_ttl:
                del st.session_state.session_state_ttl[key]
    
    def clear_all(self):
        """Clear all session state"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._init_session_state()

# ============================================================================
# BACKGROUND PROCESSING QUEUE
# ============================================================================
class BackgroundProcessor:
    """Background processing queue for heavy operations"""
    def __init__(self):
        self.queue = queue.Queue(maxsize=Config.BACKGROUND_QUEUE_MAX_SIZE)
        self.results = {}
        self.lock = threading.Lock()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start background processing thread"""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        """Stop background processing"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _process_queue(self):
        """Process items in queue"""
        while self.running:
            try:
                task_id, func, args, kwargs = self.queue.get(timeout=1)
                try:
                    result = func(*args, **kwargs)
                    with self.lock:
                        self.results[task_id] = result
                except Exception as e:
                    with self.lock:
                        self.results[task_id] = {'error': str(e)}
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue
    
    def submit(self, func, *args, **kwargs):
        """Submit task to background queue"""
        task_id = str(uuid.uuid4())
        try:
            self.queue.put((task_id, func, args, kwargs), block=False)
            return task_id
        except queue.Full:
            # Queue full, process synchronously
            try:
                result = func(*args, **kwargs)
                with self.lock:
                    self.results[task_id] = result
                return task_id
            except Exception as e:
                with self.lock:
                    self.results[task_id] = {'error': str(e)}
                return task_id
    
    def get_result(self, task_id, timeout=None):
        """Get result for task ID"""
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.lock:
                if task_id in self.results:
                    return self.results.pop(task_id)
            time.sleep(0.1)
        return None
    
    def is_complete(self, task_id):
        """Check if task is complete"""
        with self.lock:
            return task_id in self.results

# ============================================================================
# RESOURCE MONITOR
# ============================================================================
class ResourceMonitor:
    """Monitor system resources and optimize accordingly"""
    def __init__(self):
        self.last_check = 0
        self.check_interval = 5  # seconds
    
    def should_process(self):
        """Check if system resources allow processing"""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return True
        
        self.last_check = current_time
        
        if not PSUTIL_AVAILABLE:
            return True
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > Config.MAX_CPU_PERCENT:
                print(f"High CPU usage: {cpu_percent}%. Delaying processing.")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                print(f"High memory usage: {memory.percent}%. Delaying processing.")
                return False
            
            return True
        except Exception as e:
            print(f"Error checking resources: {str(e)}")
            return True
    
    def get_memory_usage(self):
        """Get current memory usage"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        return 0
    
    def cleanup_memory(self):
        """Force garbage collection"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class ImageMetadata:
    """Metadata for a single image"""
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
    def from_image(cls, image_path: Path) -> 'ImageMetadata':
        """Create metadata from image file"""
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
                exif_data=cls._extract_exif(img),
                checksum=cls._calculate_checksum(image_path)
            )
    
    @staticmethod
    def _extract_exif(img: Image.Image) -> Optional[Dict]:
        """Extract EXIF data from image"""
        try:
            exif = {}
            if hasattr(img, '_getexif') and img._getexif():
                raw_exif = img._getexif()
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    # Skip large binary data
                    if not isinstance(value, (bytes, np.ndarray)):
                        exif[tag] = str(value)
            return exif if exif else None
        except Exception as e:
            print(f"Error extracting EXIF data: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_checksum(image_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        if not image_path.exists():
            raise FileNotFoundError(f"File not found for checksum: {image_path}")
        
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

@dataclass
class AlbumEntry:
    """Album entry with user content"""
    entry_id: str
    image_id: str
    person_id: str
    caption: str
    description: str
    location: str
    date_taken: Optional[datetime.datetime]
    tags: List[str]
    privacy_level: str
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['date_taken'] = self.date_taken.isoformat() if self.date_taken else None
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class Comment:
    """Comment on an album entry"""
    comment_id: str
    entry_id: str
    user_id: str
    username: str
    content: str
    created_at: datetime.datetime
    is_edited: bool
    parent_comment_id: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class Rating:
    """Rating for an album entry"""
    rating_id: str
    entry_id: str
    user_id: str
    rating_value: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class PersonProfile:
    """Person profile information"""
    person_id: str
    folder_name: str
    display_name: str
    bio: str
    birth_date: Optional[datetime.date]
    relationship: str
    contact_info: str
    social_links: Dict[str, str]
    profile_image: Optional[str]
    created_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['birth_date'] = self.birth_date.isoformat() if self.birth_date else None
        data['created_at'] = self.created_at.isoformat()
        return data

# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================
class DatabaseConnectionPool:
    """Database connection pool for efficient database access"""
    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = []
        self.lock = threading.Lock()
        
        # Initialize pool
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self.connections.append(conn)
    
    def get_connection(self):
        """Get connection from pool"""
        with self.lock:
            if self.connections:
                return self.connections.pop()
            else:
                # Create new connection if pool is empty
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-64000")
                return conn
    
    def return_connection(self, conn):
        """Return connection to pool"""
        with self.lock:
            if len(self.connections) < self.pool_size:
                self.connections.append(conn)
            else:
                conn.close()
    
    def close_all(self):
        """Close all connections"""
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections = []

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================
class DatabaseManager:
    """Manage SQLite database operations with connection pooling"""
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE
        self.pool = DatabaseConnectionPool(self.db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with tables"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    created_date TIMESTAMP,
                    modified_date TIMESTAMP,
                    exif_data TEXT,
                    checksum TEXT UNIQUE,
                    thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY,
                    folder_name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    bio TEXT,
                    birth_date DATE,
                    relationship TEXT,
                    contact_info TEXT,
                    social_links TEXT,
                    profile_image TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS album_entries (
                    entry_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    caption TEXT,
                    description TEXT,
                    location TEXT,
                    date_taken TIMESTAMP,
                    tags TEXT,
                    privacy_level TEXT DEFAULT 'public',
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (image_id),
                    FOREIGN KEY (person_id) REFERENCES people (person_id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_edited BOOLEAN DEFAULT 0,
                    parent_comment_id TEXT,
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating_value INTEGER CHECK (rating_value BETWEEN 1 AND 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entry_id, user_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_favorites (
                    user_id TEXT,
                    entry_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, entry_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_tags ON album_entries(tags)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)')
                
                conn.commit()
                print("✅ Database initialized successfully!")
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Database initialization error: {str(e)}")
            raise
    
    def add_image(self, metadata: ImageMetadata, thumbnail_path: str = None):
        """Add image metadata to database"""
        if not isinstance(metadata, ImageMetadata):
            raise TypeError("metadata must be an ImageMetadata instance")
        
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO images
                (image_id, filename, filepath, file_size, width, height, format,
                 created_date, modified_date, exif_data, checksum, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.image_id,
                    metadata.filename,
                    metadata.filepath,
                    metadata.file_size,
                    metadata.dimensions[0],
                    metadata.dimensions[1],
                    metadata.format,
                    metadata.created_date,
                    metadata.modified_date,
                    json.dumps(metadata.exif_data) if metadata.exif_data else None,
                    metadata.checksum,
                    thumbnail_path
                ))
                conn.commit()
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error adding image to database: {str(e)}")
            raise
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve image metadata"""
        try:
            conn = self.pool.get_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM images WHERE image_id = ?', (image_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error retrieving image: {str(e)}")
            return None
    
    def add_person(self, person: PersonProfile):
        """Add person to database"""
        if not isinstance(person, PersonProfile):
            raise TypeError("person must be a PersonProfile instance")
        
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO people
                (person_id, folder_name, display_name, bio, birth_date,
                 relationship, contact_info, social_links, profile_image)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    person.person_id,
                    person.folder_name,
                    person.display_name,
                    person.bio,
                    person.birth_date.isoformat() if person.birth_date else None,
                    person.relationship,
                    person.contact_info,
                    json.dumps(person.social_links),
                    person.profile_image
                ))
                conn.commit()
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error adding person: {str(e)}")
            raise
    
    def get_all_people(self) -> List[Dict]:
        """Get all people from database"""
        try:
            conn = self.pool.get_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM people ORDER BY display_name')
                return [dict(row) for row in cursor.fetchall()]
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error retrieving people: {str(e)}")
            return []
    
    def add_album_entry(self, entry: AlbumEntry):
        """Add album entry to database"""
        if not isinstance(entry, AlbumEntry):
            raise TypeError("entry must be an AlbumEntry instance")
        
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO album_entries
                (entry_id, image_id, person_id, caption, description, location,
                 date_taken, tags, privacy_level, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id,
                    entry.image_id,
                    entry.person_id,
                    entry.caption,
                    entry.description,
                    entry.location,
                    entry.date_taken,
                    ','.join(entry.tags) if entry.tags else None,
                    entry.privacy_level,
                    entry.created_by
                ))
                conn.commit()
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error adding album entry: {str(e)}")
            raise
    
    def add_comment(self, comment: Comment):
        """Add comment to database"""
        if not isinstance(comment, Comment):
            raise TypeError("comment must be a Comment instance")
        
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO comments
                (comment_id, entry_id, user_id, username, content, parent_comment_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    comment.comment_id,
                    comment.entry_id,
                    comment.user_id,
                    comment.username,
                    comment.content,
                    comment.parent_comment_id
                ))
                conn.commit()
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error adding comment: {str(e)}")
            raise
    
    def add_rating(self, rating: Rating):
        """Add or update rating"""
        if not isinstance(rating, Rating):
            raise TypeError("rating must be a Rating instance")
        
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO ratings
                (rating_id, entry_id, user_id, rating_value)
                VALUES (?, ?, ?, ?)
                ''', (
                    rating.rating_id,
                    rating.entry_id,
                    rating.user_id,
                    rating.rating_value
                ))
                conn.commit()
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error adding rating: {str(e)}")
            raise
    
    def get_entry_comments(self, entry_id: str) -> List[Dict]:
        """Get comments for an entry"""
        try:
            conn = self.pool.get_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM comments
                WHERE entry_id = ?
                ORDER BY created_at DESC
                ''', (entry_id,))
                return [dict(row) for row in cursor.fetchall()]
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error retrieving comments: {str(e)}")
            return []
    
    def get_entry_ratings(self, entry_id: str) -> Tuple[float, int]:
        """Get average rating and count for an entry"""
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT AVG(rating_value), COUNT(*)
                FROM ratings
                WHERE entry_id = ?
                ''', (entry_id,))
                result = cursor.fetchone()
                avg_rating = result[0] if result and result[0] is not None else 0.0
                count = result[1] if result else 0
                return (float(avg_rating), int(count))
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error retrieving ratings: {str(e)}")
            return (0.0, 0)
    
    def search_entries(self, query: str, person_id: str = None) -> List[Dict]:
        """Search album entries"""
        try:
            conn = self.pool.get_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                search_pattern = f'%{query}%'
                
                if person_id:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.person_id = ? AND
                    (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                    ORDER BY ae.created_at DESC
                    ''', (person_id, search_pattern, search_pattern, search_pattern))
                else:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ORDER BY ae.created_at DESC
                    ''', (search_pattern, search_pattern, search_pattern))
                
                return [dict(row) for row in cursor.fetchall()]
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error searching entries: {str(e)}")
            return []
    
    def get_entry_details(self, entry_id: str) -> Optional[Dict]:
        """Get detailed information for an entry"""
        try:
            conn = self.pool.get_connection()
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT
                    ae.entry_id,
                    ae.image_id,
                    ae.person_id,
                    ae.caption,
                    ae.description,
                    ae.location,
                    ae.date_taken,
                    ae.tags,
                    ae.privacy_level,
                    ae.created_by,
                    ae.created_at,
                    ae.updated_at,
                    p.display_name,
                    p.folder_name,
                    i.filename,
                    i.filepath,
                    i.file_size,
                    i.format,
                    i.created_date,
                    i.exif_data,
                    i.thumbnail_path,
                    (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                    (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                WHERE ae.entry_id = ?
                ''', (entry_id,))
                row = cursor.fetchone()
                
                if row:
                    result = dict(row)
                    # Parse JSON fields
                    if result.get('exif_data'):
                        try:
                            result['exif_data'] = json.loads(result['exif_data'])
                        except Exception as e:
                            result['exif_data'] = {}
                            print(f"Error parsing exif_data: {str(e)}")
                    
                    # Parse tags
                    if result.get('tags'):
                        result['tags'] = [tag.strip() for tag in result['tags'].split(',') if tag.strip()]
                    else:
                        result['tags'] = []
                    
                    return result
                return None
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Database error retrieving entry details: {str(e)}")
            return None
    
    def batch_add_images(self, metadata_list: List[ImageMetadata], thumbnail_paths: List[Optional[str]] = None):
        """Batch add multiple images to database"""
        if not metadata_list:
            return
        
        try:
            conn = self.pool.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('BEGIN TRANSACTION')
                
                for i, metadata in enumerate(metadata_list):
                    thumbnail_path = thumbnail_paths[i] if thumbnail_paths and i < len(thumbnail_paths) else None
                    cursor.execute('''
                    INSERT OR REPLACE INTO images
                    (image_id, filename, filepath, file_size, width, height, format,
                     created_date, modified_date, exif_data, checksum, thumbnail_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metadata.image_id,
                        metadata.filename,
                        metadata.filepath,
                        metadata.file_size,
                        metadata.dimensions[0],
                        metadata.dimensions[1],
                        metadata.format,
                        metadata.created_date,
                        metadata.modified_date,
                        json.dumps(metadata.exif_data) if metadata.exif_data else None,
                        metadata.checksum,
                        thumbnail_path
                    ))
                
                conn.commit()
            finally:
                self.pool.return_connection(conn)
        except sqlite3.Error as e:
            print(f"Error batch adding images: {str(e)}")
            raise

# ============================================================================
# IMAGE PROCESSING AND THUMBNAIL MANAGEMENT
# ============================================================================
class ImageProcessor:
    """Handle image processing operations with optimization"""
    def __init__(self):
        self.thumbnail_cache = {}
        self.max_cache_size = Config.IMAGE_CACHE_MAX_SIZE
    
    def create_thumbnail(self, image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for image"""
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return None
        
        thumbnail_dir = thumbnail_dir or Config.THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
        
        try:
            # Check if thumbnail already exists and is up to date
            if thumbnail_path.exists():
                thumb_mtime = thumbnail_path.stat().st_mtime
                img_mtime = image_path.stat().st_mtime
                if thumb_mtime >= img_mtime:
                    return thumbnail_path
            
            with Image.open(image_path) as img:
                # Handle image rotation based on EXIF
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
                
                # Save thumbnail as JPG
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
            
            return thumbnail_path
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None
    
    def get_image_data_url(self, image_path: Path) -> str:
        """Convert image to data URL for embedding"""
        try:
            if not image_path.exists():
                return ""
            
            # Check cache
            cache_key = str(image_path)
            if cache_key in self.thumbnail_cache:
                return self.thumbnail_cache[cache_key]
            
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                mime_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else f"image/{image_path.suffix[1:]}"
                data_url = f"data:{mime_type};base64,{encoded}"
            
            # Cache result if cache not full
            if len(self.thumbnail_cache) < self.max_cache_size:
                self.thumbnail_cache[cache_key] = data_url
            
            return data_url
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return ""
    
    def clear_cache(self):
        """Clear thumbnail cache"""
        self.thumbnail_cache.clear()

# ============================================================================
# OPTIMIZED CACHE MANAGEMENT
# ============================================================================
class OptimizedCacheManager:
    """Enhanced cache manager with size limits and TTL"""
    def __init__(self, max_size=Config.CACHE_MAX_SIZE):
        self._cache = {}
        self._timestamps = {}
        self._access_count = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str, default=None):
        """Get cached value if not expired"""
        with self.lock:
            if key in self._cache:
                if time.time() - self._timestamps[key] < Config.CACHE_TTL:
                    # Update access count for LRU
                    self._access_count[key] = self._access_count.get(key, 0) + 1
                    return self._cache[key]
                else:
                    # Expired, remove from cache
                    del self._cache[key]
                    del self._timestamps[key]
                    if key in self._access_count:
                        del self._access_count[key]
            return default
    
    def set(self, key: str, value):
        """Set cache value with size management"""
        with self.lock:
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                if key in self._access_count:
                    del self._access_count[key]
            
            # Check if cache is full
            if len(self._cache) >= self.max_size:
                # Remove least recently used item
                if self._access_count:
                    lru_key = min(self._access_count.keys(), key=lambda k: self._access_count[k])
                    del self._cache[lru_key]
                    del self._timestamps[lru_key]
                    del self._access_count[lru_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_count[key] = 1
    
    def clear(self, key: str = None):
        """Clear cache or specific key"""
        with self.lock:
            if key:
                if key in self._cache:
                    del self._cache[key]
                    del self._timestamps[key]
                    if key in self._access_count:
                        del self._access_count[key]
            else:
                self._cache.clear()
                self._timestamps.clear()
                self._access_count.clear()
    
    def get_or_set(self, key: str, generator_func):
        """Get cached value or generate and cache it"""
        cached = self.get(key)
        if cached is not None:
            return cached
        
        try:
            value = generator_func()
            self.set(key, value)
            return value
        except Exception as e:
            print(f"Error generating cached value for {key}: {str(e)}")
            return None
    
    def cleanup_expired(self):
        """Remove all expired cache entries"""
        with self.lock:
            current_time = time.time()
            expired_keys = [k for k, t in self._timestamps.items() if current_time - t > Config.CACHE_TTL]
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
                if key in self._access_count:
                    del self._access_count[key]

# ============================================================================
# ALBUM MANAGER WITH OPTIMIZATIONS
# ============================================================================
class AlbumManager:
    """Main album management class with robust error handling and optimizations"""
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = OptimizedCacheManager()
        self.image_processor = ImageProcessor()
        self.session_manager = SessionStateManager()
        self.background_processor = BackgroundProcessor()
        self.resource_monitor = ResourceMonitor()
        self._init_session_state()
        
        # Start background processor if enabled
        if Config.BACKGROUND_PROCESSING_ENABLED:
            self.background_processor.start()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        session_keys = [
            'initialized', 'current_page', 'selected_person', 'selected_image',
            'search_query', 'view_mode', 'sort_by', 'sort_order', 'selected_tags',
            'user_id', 'username', 'user_role', 'favorites', 'recently_viewed',
            'toc_page', 'gallery_page', 'show_directory_info'
        ]
        
        for key in session_keys:
            if key not in st.session_state:
                if key == 'favorites':
                    st.session_state[key] = set()
                elif key == 'recently_viewed':
                    st.session_state[key] = []
                elif key == 'user_id':
                    st.session_state[key] = str(uuid.uuid4())
                elif key == 'username':
                    st.session_state[key] = 'Guest'
                elif key == 'user_role':
                    st.session_state[key] = UserRoles.VIEWER.value
                elif key == 'initialized':
                    st.session_state[key] = True
                else:
                    st.session_state[key] = None
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        """Scan directory for images and update database with optimizations"""
        data_dir = data_dir or Config.DATA_DIR
        results = {
            'total_images': 0,
            'new_images': 0,
            'updated_images': 0,
            'people_found': 0,
            'errors': []
        }
        
        try:
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
                print(f"Created directory: {data_dir}")
                return results
            
            # Find all person directories
            person_dirs = [d for d in data_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.')
                          and d.name != 'thumbnails']
            
            if not person_dirs:
                print("No person folders found. Creating sample structure...")
                Config.create_sample_structure()
                person_dirs = [d for d in data_dir.iterdir()
                              if d.is_dir() and not d.name.startswith('.')
                              and d.name != 'thumbnails']
            
            results['people_found'] = len(person_dirs)
            
            # Process each person directory
            for person_dir in person_dirs:
                # Handle person name formatting
                if '-' in person_dir.name:
                    display_name = ' '.join(part.capitalize() for part in person_dir.name.split('-'))
                elif '_' in person_dir.name:
                    display_name = ' '.join(part.capitalize() for part in person_dir.name.split('_'))
                else:
                    display_name = ' '.join(word.capitalize() for word in person_dir.name.split())
                
                # Create or update person profile
                existing_person = None
                for p in self.db.get_all_people():
                    if p['folder_name'] == person_dir.name:
                        existing_person = p
                        break
                
                if not existing_person:
                    person_profile = PersonProfile(
                        person_id=str(uuid.uuid4()),
                        folder_name=person_dir.name,
                        display_name=display_name,
                        bio=f"Photos of {display_name}",
                        birth_date=None,
                        relationship="Family/Friend",
                        contact_info="",
                        social_links={},
                        profile_image=None,
                        created_at=datetime.datetime.now()
                    )
                    self.db.add_person(person_profile)
                    person_id = person_profile.person_id
                else:
                    person_id = existing_person['person_id']
                
                # Process images in batches
                image_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                
                # Process in batches for better performance
                for i in range(0, len(image_files), Config.BATCH_SIZE):
                    batch = image_files[i:i + Config.BATCH_SIZE]
                    batch_results = self._process_image_batch(batch, person_id, display_name)
                    results['new_images'] += batch_results['new_images']
                    results['updated_images'] += batch_results['updated_images']
                    results['errors'].extend(batch_results['errors'])
            
            results['total_images'] = results['new_images'] + results['updated_images']
            self.cache.clear()
            
            return results
        except Exception as e:
            print(f"Critical error during directory scan: {str(e)}")
            results['errors'].append(f"Critical error: {str(e)}")
            return results
    
    def _process_image_batch(self, image_files: List[Path], person_id: str, display_name: str) -> Dict:
        """Process a batch of images efficiently"""
        results = {'new_images': 0, 'updated_images': 0, 'errors': []}
        
        for img_path in image_files:
            try:
                # Calculate checksum
                checksum = ImageMetadata._calculate_checksum(img_path)
                
                # Check if image exists in database
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT image_id FROM images WHERE checksum = ?', (checksum,))
                    existing = cursor.fetchone()
                
                if existing:
                    results['updated_images'] += 1
                    continue
                
                # Create thumbnail
                thumbnail_path = self.image_processor.create_thumbnail(img_path)
                thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                
                # Extract metadata
                try:
                    metadata = ImageMetadata.from_image(img_path)
                except Exception as e:
                    results['errors'].append(f"Error processing {img_path.name}: {str(e)}")
                    continue
                
                # Add to database
                self.db.add_image(metadata, thumbnail_path_str)
                
                # Create album entry
                album_entry = AlbumEntry(
                    entry_id=str(uuid.uuid4()),
                    image_id=metadata.image_id,
                    person_id=person_id,
                    caption=img_path.stem.replace('_', ' ').title(),
                    description=f"Photo of {display_name}",
                    location="",
                    date_taken=metadata.created_date,
                    tags=[display_name.lower().replace(' ', '-'), 'photo', 'memory'],
                    privacy_level='public',
                    created_by='system',
                    created_at=datetime.datetime.now(),
                    updated_at=datetime.datetime.now()
                )
                self.db.add_album_entry(album_entry)
                
                results['new_images'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {img_path}: {str(e)}"
                results['errors'].append(error_msg)
                print(error_msg)
        
        return results
    
    def get_person_stats(self, person_id: str) -> Dict:
        """Get statistics for a person with error handling"""
        cache_key = f"person_stats_{person_id}"
        
        def generate_stats():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM album_entries WHERE person_id = ?', (person_id,))
                    image_count = cursor.fetchone()[0]
                    
                    cursor.execute('''
                    SELECT COUNT(DISTINCT c.comment_id)
                    FROM comments c
                    JOIN album_entries ae ON c.entry_id = ae.entry_id
                    WHERE ae.person_id = ?
                    ''', (person_id,))
                    comment_count = cursor.fetchone()[0]
                    
                    cursor.execute('''
                    SELECT AVG(r.rating_value)
                    FROM ratings r
                    JOIN album_entries ae ON r.entry_id = ae.entry_id
                    WHERE ae.person_id = ?
                    ''', (person_id,))
                    avg_rating = cursor.fetchone()[0] or 0.0
                    
                    cursor.execute('SELECT MAX(ae.created_at) FROM album_entries ae WHERE ae.person_id = ?', (person_id,))
                    last_activity = cursor.fetchone()[0]
                    
                    return {
                        'image_count': image_count,
                        'comment_count': comment_count,
                        'avg_rating': float(avg_rating),
                        'last_activity': last_activity
                    }
            except sqlite3.Error as e:
                print(f"Error getting person stats: {str(e)}")
                return {
                    'image_count': 0,
                    'comment_count': 0,
                    'avg_rating': 0.0,
                    'last_activity': None
                }
        
        return self.cache.get_or_set(cache_key, generate_stats)
    
    def add_to_favorites(self, entry_id: str):
        """Add entry to user favorites"""
        user_id = st.session_state['user_id']
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR IGNORE INTO user_favorites (user_id, entry_id)
                VALUES (?, ?)
                ''', (user_id, entry_id))
                conn.commit()
            
            if 'favorites' not in st.session_state:
                st.session_state.favorites = set()
            st.session_state.favorites.add(entry_id)
        except sqlite3.Error as e:
            print(f"Error adding to favorites: {str(e)}")
    
    def remove_from_favorites(self, entry_id: str):
        """Remove entry from user favorites"""
        user_id = st.session_state['user_id']
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                DELETE FROM user_favorites WHERE user_id = ? AND entry_id = ?
                ''', (user_id, entry_id))
                conn.commit()
            
            if 'favorites' in st.session_state and entry_id in st.session_state.favorites:
                st.session_state.favorites.remove(entry_id)
        except sqlite3.Error as e:
            print(f"Error removing from favorites: {str(e)}")
    
    def get_user_favorites(self) -> List[Dict]:
        """Get user's favorite entries"""
        user_id = st.session_state['user_id']
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT ae.*, p.display_name, i.filename, i.thumbnail_path
                FROM user_favorites uf
                JOIN album_entries ae ON uf.entry_id = ae.entry_id
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                WHERE uf.user_id = ?
                ORDER BY uf.created_at DESC
                ''', (user_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error retrieving favorites: {str(e)}")
            return []
    
    def add_comment_to_entry(self, entry_id: str, content: str, parent_comment_id: str = None):
        """Add comment to an entry"""
        if not content or len(content.strip()) == 0:
            print("Comment cannot be empty")
            return False
        
        if len(content) > Config.MAX_COMMENT_LENGTH:
            print(f"Comment is too long. Maximum length is {Config.MAX_COMMENT_LENGTH} characters.")
            return False
        
        try:
            comment = Comment(
                comment_id=str(uuid.uuid4()),
                entry_id=entry_id,
                user_id=st.session_state['user_id'],
                username=st.session_state['username'],
                content=content.strip(),
                created_at=datetime.datetime.now(),
                is_edited=False,
                parent_comment_id=parent_comment_id
            )
            self.db.add_comment(comment)
            
            # Clear cache for this entry's comments
            cache_key = f"comments_{entry_id}"
            self.cache.clear(cache_key)
            
            print("Comment added successfully!")
            return True
        except Exception as e:
            print(f"Error adding comment: {str(e)}")
            return False
    
    def add_rating_to_entry(self, entry_id: str, rating_value: int):
        """Add or update rating for an entry"""
        if rating_value < 1 or rating_value > 5:
            print("Rating must be between 1 and 5")
            return False
        
        try:
            rating = Rating(
                rating_id=str(uuid.uuid4()),
                entry_id=entry_id,
                user_id=st.session_state['user_id'],
                rating_value=rating_value,
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
            self.db.add_rating(rating)
            
            # Clear cache for this entry's ratings
            cache_key = f"ratings_{entry_id}"
            self.cache.clear(cache_key)
            
            print(f"Rating of {rating_value} stars added!")
            return True
        except Exception as e:
            print(f"Error adding rating: {str(e)}")
            return False
    
    def get_all_people_with_stats(self) -> List[Dict]:
        """Get all people with their statistics"""
        cache_key = "all_people_with_stats"
        
        def generate_data():
            try:
                people = self.db.get_all_people()
                result = []
                for person in people:
                    stats = self.get_person_stats(person['person_id'])
                    person_with_stats = {**person, **stats}
                    
                    # Get profile image if exists
                    if person.get('profile_image'):
                        profile_image_path = Config.DATA_DIR / person['folder_name'] / person['profile_image']
                        if profile_image_path.exists():
                            person_with_stats['profile_image_data'] = self.image_processor.get_image_data_url(profile_image_path)
                    
                    result.append(person_with_stats)
                return result
            except Exception as e:
                print(f"Error getting people with stats: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_entries_by_person(self, person_id: str, page: int = 1, search_query: str = None) -> Dict:
        """Get entries for a specific person with pagination"""
        cache_key = f"entries_person_{person_id}_page_{page}_query_{search_query}"
        
        def generate_data():
            try:
                offset = (page - 1) * Config.ITEMS_PER_PAGE
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    if search_query:
                        search_pattern = f'%{search_query}%'
                        cursor.execute('''
                        SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                        (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                        (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                        FROM album_entries ae
                        JOIN people p ON ae.person_id = p.person_id
                        JOIN images i ON ae.image_id = i.image_id
                        WHERE ae.person_id = ? AND
                        (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                        ORDER BY ae.created_at DESC
                        LIMIT ? OFFSET ?
                        ''', (person_id, search_pattern, search_pattern, search_pattern, Config.ITEMS_PER_PAGE, offset))
                    else:
                        cursor.execute('''
                        SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                        (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                        (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                        FROM album_entries ae
                        JOIN people p ON ae.person_id = p.person_id
                        JOIN images i ON ae.image_id = i.image_id
                        WHERE ae.person_id = ?
                        ORDER BY ae.created_at DESC
                        LIMIT ? OFFSET ?
                        ''', (person_id, Config.ITEMS_PER_PAGE, offset))
                    
                    entries = [dict(row) for row in cursor.fetchall()]
                    
                    # Get total count
                    if search_query:
                        search_pattern = f'%{search_query}%'
                        cursor.execute('''
                        SELECT COUNT(*)
                        FROM album_entries ae
                        WHERE ae.person_id = ? AND
                        (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                        ''', (person_id, search_pattern, search_pattern, search_pattern))
                    else:
                        cursor.execute('SELECT COUNT(*) FROM album_entries WHERE person_id = ?', (person_id,))
                    
                    total_count = cursor.fetchone()[0]
                    total_pages = max(1, math.ceil(total_count / Config.ITEMS_PER_PAGE))
                    
                    return {
                        'entries': entries,
                        'total_count': total_count,
                        'total_pages': total_pages,
                        'current_page': page
                    }
            except sqlite3.Error as e:
                print(f"Error getting entries by person: {str(e)}")
                return {
                    'entries': [],
                    'total_count': 0,
                    'total_pages': 1,
                    'current_page': page
                }
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        """Get most recent entries"""
        cache_key = f"recent_entries_{limit}"
        
        def generate_data():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    ORDER BY ae.created_at DESC
                    LIMIT ?
                    ''', (limit,))
                    return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                print(f"Error getting recent entries: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_top_rated_entries(self, limit: int = 10) -> List[Dict]:
        """Get top rated entries"""
        cache_key = f"top_rated_entries_{limit}"
        
        def generate_data():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                    (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.entry_id IN (
                        SELECT r.entry_id
                        FROM ratings r
                        GROUP BY r.entry_id
                        HAVING AVG(r.rating_value) >= 4.0
                        ORDER BY AVG(r.rating_value) DESC
                    )
                    ORDER BY avg_rating DESC
                    LIMIT ?
                    ''', (limit,))
                    return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                print(f"Error getting top rated entries: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_entry_with_details(self, entry_id: str) -> Optional[Dict]:
        """Get complete entry details including comments and ratings"""
        cache_key = f"entry_details_{entry_id}"
        
        def generate_data():
            try:
                entry = self.db.get_entry_details(entry_id)
                if not entry:
                    return None
                
                # Get comments
                comments = self.db.get_entry_comments(entry_id)
                entry['comments'] = comments
                
                # Get ratings
                avg_rating, rating_count = self.db.get_entry_ratings(entry_id)
                entry['avg_rating'] = avg_rating
                entry['rating_count'] = rating_count
                
                # Get image path
                if entry.get('filepath'):
                    image_path = Config.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        entry['image_data_url'] = self.image_processor.get_image_data_url(image_path)
                
                # Get thumbnail
                if entry.get('thumbnail_path'):
                    thumbnail_path = Path(entry['thumbnail_path'])
                    if thumbnail_path.exists():
                        entry['thumbnail_data_url'] = self.image_processor.get_image_data_url(thumbnail_path)
                
                # Check if user has favorited
                user_id = st.session_state['user_id']
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT 1 FROM user_favorites WHERE user_id = ? AND entry_id = ?
                    ''', (user_id, entry_id))
                    entry['is_favorited'] = cursor.fetchone() is not None
                
                # Check user rating
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT rating_value FROM ratings WHERE user_id = ? AND entry_id = ?
                    ''', (user_id, entry_id))
                    user_rating = cursor.fetchone()
                    entry['user_rating'] = user_rating[0] if user_rating else 0
                
                return entry
            except Exception as e:
                print(f"Error getting entry details: {str(e)}")
                return None
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def update_entry(self, entry_id: str, updates: Dict) -> bool:
        """Update album entry"""
        try:
            # Get current entry
            entry = self.get_entry_with_details(entry_id)
            if not entry:
                print("Entry not found")
                return False
            
            # Prepare update values
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query
                set_clauses = []
                values = []
                if 'caption' in updates:
                    set_clauses.append("caption = ?")
                    values.append(updates['caption'][:Config.MAX_CAPTION_LENGTH])
                if 'description' in updates:
                    set_clauses.append("description = ?")
                    values.append(updates['description'])
                if 'location' in updates:
                    set_clauses.append("location = ?")
                    values.append(updates['location'])
                if 'tags' in updates:
                    if isinstance(updates['tags'], list):
                        tags_str = ','.join(updates['tags'])
                    else:
                        tags_str = updates['tags']
                    set_clauses.append("tags = ?")
                    values.append(tags_str)
                if 'privacy_level' in updates:
                    set_clauses.append("privacy_level = ?")
                    values.append(updates['privacy_level'])
                
                if set_clauses:
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    values.append(entry_id)
                    query = f'''
                    UPDATE album_entries
                    SET {', '.join(set_clauses)}
                    WHERE entry_id = ?
                    '''
                    cursor.execute(query, values)
                    conn.commit()
                    
                    # Clear cache
                    self.cache.clear(f"entry_details_{entry_id}")
                    self.cache.clear(f"entries_person_{entry['person_id']}")
                    
                    print("Entry updated successfully!")
                    return True
                else:
                    print("No updates provided")
                    return False
        except Exception as e:
            print(f"Error updating entry: {str(e)}")
            return False
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete album entry (admin only)"""
        try:
            # Get entry details for cleanup
            entry = self.get_entry_with_details(entry_id)
            if not entry:
                print("Entry not found")
                return False
            
            # Check permissions
            if st.session_state['user_role'] != UserRoles.ADMIN.value:
                print("Only administrators can delete entries")
                return False
            
            # Confirm deletion
            if not st.session_state.get(f"confirm_delete_{entry_id}"):
                print(f"Are you sure you want to delete entry '{entry.get('caption', 'Unknown')}'?")
                if st.button("Confirm Deletion", key=f"confirm_btn_{entry_id}"):
                    st.session_state[f"confirm_delete_{entry_id}"] = True
                    st.rerun()
                return False
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                # Delete comments
                cursor.execute('DELETE FROM comments WHERE entry_id = ?', (entry_id,))
                # Delete ratings
                cursor.execute('DELETE FROM ratings WHERE entry_id = ?', (entry_id,))
                # Delete favorites
                cursor.execute('DELETE FROM user_favorites WHERE entry_id = ?', (entry_id,))
                # Delete album entry
                cursor.execute('DELETE FROM album_entries WHERE entry_id = ?', (entry_id,))
                conn.commit()
            
            # Clear relevant caches
            self.cache.clear(f"entry_details_{entry_id}")
            self.cache.clear(f"entries_person_{entry['person_id']}")
            self.cache.clear("recent_entries_")
            self.cache.clear("top_rated_entries_")
            
            print("Entry deleted successfully!")
            
            # Reset confirmation
            st.session_state[f"confirm_delete_{entry_id}"] = False
            return True
        except Exception as e:
            print(f"Error deleting entry: {str(e)}")
            return False
    
    def export_data(self, format_type: str = 'csv') -> Optional[io.BytesIO]:
        """Export album data in specified format"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                # Get all entries with details
                query = '''
                SELECT
                    ae.entry_id,
                    ae.caption,
                    ae.description,
                    ae.location,
                    ae.date_taken,
                    ae.tags,
                    ae.privacy_level,
                    ae.created_at,
                    p.display_name as person_name,
                    i.filename,
                    i.file_size,
                    i.format,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                    (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                ORDER BY ae.created_at DESC
                '''
                df = pd.read_sql_query(query, conn)
                
                if format_type == 'csv':
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    return io.BytesIO(output.getvalue().encode('utf-8'))
                elif format_type == 'json':
                    output = io.StringIO()
                    df.to_json(output, orient='records', indent=2)
                    return io.BytesIO(output.getvalue().encode('utf-8'))
                elif format_type == 'excel':
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Album Data', index=False)
                    output.seek(0)
                    return output
                else:
                    print(f"Unsupported format: {format_type}")
                    return None
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return None
    
    def import_data(self, file_content: bytes, format_type: str) -> bool:
        """Import album data from file"""
        try:
            if format_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            elif format_type == 'json':
                df = pd.read_json(io.BytesIO(file_content))
            elif format_type == 'excel':
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                print(f"Unsupported format: {format_type}")
                return False
            
            # Validate required columns
            required_columns = ['caption', 'person_name', 'filename']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return False
            
            # Process each row
            success_count = 0
            error_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Find or create person
                    people = self.db.get_all_people()
                    person = None
                    for p in people:
                        if p['display_name'] == row['person_name']:
                            person = p
                            break
                    
                    if not person:
                        # Create new person
                        person_profile = PersonProfile(
                            person_id=str(uuid.uuid4()),
                            folder_name=row['person_name'].lower().replace(' ', '-'),
                            display_name=row['person_name'],
                            bio=f"Photos of {row['person_name']}",
                            birth_date=None,
                            relationship="Family/Friend",
                            contact_info="",
                            social_links={},
                            profile_image=None,
                            created_at=datetime.datetime.now()
                        )
                        self.db.add_person(person_profile)
                        person_id = person_profile.person_id
                    else:
                        person_id = person['person_id']
                    
                    # Check if image file exists
                    image_path = Config.DATA_DIR / person_profile.folder_name / row['filename']
                    if not image_path.exists():
                        print(f"Image file not found: {image_path}")
                        error_count += 1
                        continue
                    
                    # Check if entry already exists
                    with sqlite3.connect(self.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                        SELECT COUNT(*) FROM album_entries ae
                        JOIN images i ON ae.image_id = i.image_id
                        WHERE i.filename = ? AND ae.person_id = ?
                        ''', (row['filename'], person_id))
                        if cursor.fetchone()[0] > 0:
                            print(f"Entry already exists for {row['filename']}")
                            success_count += 1
                            continue
                    
                    # Process image and create entry
                    metadata = ImageMetadata.from_image(image_path)
                    thumbnail_path = self.image_processor.create_thumbnail(image_path)
                    thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                    self.db.add_image(metadata, thumbnail_path_str)
                    
                    # Parse tags
                    tags = []
                    if 'tags' in row and pd.notna(row['tags']):
                        if isinstance(row['tags'], str):
                            tags = [tag.strip() for tag in row['tags'].split(',') if tag.strip()]
                    
                    album_entry = AlbumEntry(
                        entry_id=str(uuid.uuid4()),
                        image_id=metadata.image_id,
                        person_id=person_id,
                        caption=row['caption'],
                        description=row.get('description', ''),
                        location=row.get('location', ''),
                        date_taken=row.get('date_taken'),
                        tags=tags,
                        privacy_level=row.get('privacy_level', 'public'),
                        created_by=st.session_state['username'],
                        created_at=datetime.datetime.now(),
                        updated_at=datetime.datetime.now()
                    )
                    self.db.add_album_entry(album_entry)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error processing row: {str(e)}")
            
            # Clear cache
            self.cache.clear()
            
            print(f"Import completed: {success_count} successful, {error_count} failed")
            return success_count > 0
        except Exception as e:
            print(f"Error importing data: {str(e)}")
            return False

# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    """Reusable UI components"""
    @staticmethod
    def rating_stars(rating: float, max_rating: int = 5, size: int = 20) -> str:
        """Generate HTML for rating stars"""
        if rating is None or rating <= 0:
            rating = 0
        
        stars_html = []
        full_stars = int(rating)
        has_half_star = rating - full_stars >= 0.5
        
        for i in range(max_rating):
            if i < full_stars:
                stars_html.append('⭐')
            elif i == full_stars and has_half_star:
                stars_html.append('⭐')
            else:
                stars_html.append('☆')
        
        return f"""
        <div style="color: #FFD700; font-size: {size}px; letter-spacing: 1px; display: inline-block;">
        {''.join(stars_html)}
        <span style="color: #666; font-size: 14px; margin-left: 8px; vertical-align: middle;">
        {rating:.1f}/{max_rating}
        </span>
        </div>
        """
    
    @staticmethod
    def tag_badges(tags: List[str], max_display: int = 5) -> str:
        """Generate HTML for tag badges"""
        if not tags:
            return ""
        
        displayed_tags = tags[:max_display]
        extra_count = len(tags) - max_display if len(tags) > max_display else 0
        
        badges = []
        for tag in displayed_tags:
            badges.append(f'''
            <span style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-transform: capitalize;
            ">
            {tag.replace('-', ' ')}
            </span>
            ''')
        
        html = ' '.join(badges)
        
        if extra_count > 0:
            html += f'''
            <span style="
            background: #f0f0f0;
            color: #666;
            padding: 4px 8px;
            border-radius: 20px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
            ">
            +{extra_count} more
            </span>
            '''
        
        return html

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================
class PhotoAlbumApp:
    """Main application class with UI components and optimizations"""
    def __init__(self):
        self.manager = AlbumManager()
        self.setup_page_config()
        self.check_initialization()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=Config.APP_NAME,
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-album',
                'Report a bug': 'https://github.com/yourusername/photo-album/issues',
                'About': f"# {Config.APP_NAME} v{Config.VERSION}\nA comprehensive photo album management system"
            }
        )
    
    def check_initialization(self):
        """Check and initialize application directories"""
        try:
            Config.init_directories()
            self.initialized = True
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            self.initialized = False
    
    def render_sidebar(self):
        """Render application sidebar"""
        with st.sidebar:
            st.title(f"📸 {Config.APP_NAME}")
            st.caption(f"v{Config.VERSION}")
            
            # User info
            st.divider()
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**{st.session_state['username']}**")
                st.caption(st.session_state['user_role'].title())
            
            # Navigation
            st.divider()
            st.subheader("Navigation")
            
            nav_options = {
                "🏠 Dashboard": "dashboard",
                "👥 People": "people",
                "📁 Gallery": "gallery",
                "⭐ Favorites": "favorites",
                "🔍 Search": "search",
                "📊 Statistics": "statistics",
                "⚙️ Settings": "settings",
                "📤 Import/Export": "import_export"
            }
            
            for label, key in nav_options.items():
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
                    st.rerun()
            
            # Quick actions
            st.divider()
            st.subheader("Quick Actions")
            if st.button("🔄 Scan Directory", use_container_width=True):
                with st.spinner("Scanning directory..."):
                    results = self.manager.scan_directory()
                    if results['new_images'] > 0 or results['updated_images'] > 0:
                        st.success(f"Found {results['new_images']} new and {results['updated_images']} updated images")
                    st.rerun()
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.manager.cache.clear()
                st.success("Cache cleared!")
                st.rerun()
            
            # Display info
            st.divider()
            st.subheader("Info")
            people = self.manager.db.get_all_people()
            total_images = 0
            for person in people:
                stats = self.manager.get_person_stats(person['person_id'])
                total_images += stats['image_count']
            
            st.metric("People", len(people))
            st.metric("Total Images", total_images)
            
            # Performance info
            st.divider()
            st.subheader("Performance")
            st.metric("Cache Size", len(self.manager.cache._cache))
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent}%")
            
            # Theme toggle
            st.divider()
            dark_mode = st.toggle("Dark Mode", value=False)
            if dark_mode:
                st.markdown("""
                <style>
                .stApp {
                    background-color: #1a1a1a;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render dashboard page"""
        st.title("📊 Dashboard")
        st.markdown("Welcome to your photo album management system!")
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            people = self.manager.db.get_all_people()
            st.metric("👥 People", len(people))
        
        with col2:
            total_images = sum(self.manager.get_person_stats(p['person_id'])['image_count'] for p in people)
            st.metric("📸 Total Images", total_images)
        
        with col3:
            recent = self.manager.get_recent_entries(limit=1)
            if recent:
                st.metric("🕐 Last Added", recent[0].get('caption', 'N/A'))
            else:
                st.metric("🕐 Last Added", "None")
        
        with col4:
            top_rated = self.manager.get_top_rated_entries(limit=1)
            if top_rated:
                avg_rating = top_rated[0].get('avg_rating', 0)
                st.metric("⭐ Top Rated", f"{avg_rating:.1f}/5")
            else:
                st.metric("⭐ Top Rated", "N/A")
        
        st.divider()
        
        # Recent Activity
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📅 Recent Activity")
            recent_entries = self.manager.get_recent_entries(limit=5)
            if recent_entries:
                for entry in recent_entries:
                    with st.container():
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            if entry.get('thumbnail_data_url'):
                                st.image(entry['thumbnail_data_url'], width=80)
                            else:
                                st.markdown("🖼️")
                        with col_b:
                            st.markdown(f"**{entry.get('caption', 'Untitled')}**")
                            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                            st.caption(f"📅 {entry.get('created_at', '')}")
                        st.divider()
            else:
                st.info("No recent activity")
        
        with col2:
            st.subheader("🏆 Top Rated")
            top_entries = self.manager.get_top_rated_entries(limit=5)
            if top_entries:
                for entry in top_entries:
                    rating = entry.get('avg_rating', 0)
                    if rating >= 4:
                        st.markdown(f"⭐ **{entry.get('caption', 'Untitled')}**")
                        st.markdown(f"`{rating:.1f}/5` 👤 {entry.get('display_name', 'Unknown')}")
                        st.divider()
            else:
                st.info("No ratings yet")
        
        # Quick Actions
        st.divider()
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Scan for New Photos", use_container_width=True):
                with st.spinner("Scanning..."):
                    results = self.manager.scan_directory()
                    st.success(f"Found {results['new_images']} new photos!")
                    st.rerun()
        
        with col2:
            if st.button("👥 Manage People", use_container_width=True):
                st.session_state['current_page'] = 'people'
                st.rerun()
        
        with col3:
            if st.button("📤 Export Data", use_container_width=True):
                st.session_state['current_page'] = 'import_export'
                st.rerun()
    
    def render_people_page(self):
        """Render people management page"""
        st.title("👥 People")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search people...", key="people_search")
        with col2:
            sort_by = st.selectbox("Sort by", ["Name", "Recently Added", "Image Count"], key="people_sort")
        
        # Get all people
        all_people = self.manager.get_all_people_with_stats()
        
        # Filter and sort
        if search_query:
            all_people = [p for p in all_people if search_query.lower() in p['display_name'].lower()]
        
        if sort_by == "Name":
            all_people.sort(key=lambda x: x['display_name'])
        elif sort_by == "Image Count":
            all_people.sort(key=lambda x: x['image_count'], reverse=True)
        elif sort_by == "Recently Added":
            all_people.sort(key=lambda x: x.get('last_activity', ''), reverse=True)
        
        # Display people
        cols = st.columns(3)
        for idx, person in enumerate(all_people):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Profile image or placeholder
                    if person.get('profile_image_data'):
                        st.image(person['profile_image_data'], use_column_width=True)
                    else:
                        # Create colored placeholder
                        colors = ['#667eea', '#764ba2', '#f56565', '#48bb78', '#ed8936']
                        color = colors[hash(person['person_id']) % len(colors)]
                        st.markdown(
                            f'<div style="background: {color}; height: 150px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">'
                            f'<span style="color: white; font-size: 48px;">{person["display_name"][0].upper()}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.subheader(person['display_name'])
                    st.caption(f"📁 {person['folder_name']}")
                    
                    # Stats
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Photos", person.get('image_count', 0))
                    with col_b:
                        st.metric("Comments", person.get('comment_count', 0))
                    with col_c:
                        st.metric("Rating", f"{person.get('avg_rating', 0):.1f}")
                    
                    # Actions
                    if st.button("View Gallery", key=f"view_{person['person_id']}", use_container_width=True):
                        st.session_state['selected_person'] = person['person_id']
                        st.session_state['current_page'] = 'gallery'
                        st.rerun()
        
        # Add new person
        st.divider()
        with st.expander("➕ Add New Person"):
            with st.form("add_person_form"):
                col1, col2 = st.columns(2)
                with col1:
                    folder_name = st.text_input("Folder Name*")
                    display_name = st.text_input("Display Name*")
                    relationship = st.selectbox(
                        "Relationship",
                        ["Family", "Friend", "Colleague", "Relative", "Other"]
                    )
                with col2:
                    birth_date = st.date_input("Birth Date", value=None)
                    contact_info = st.text_input("Contact Info")
                    bio = st.text_area("Bio", height=100)
                
                if st.form_submit_button("Add Person"):
                    if folder_name and display_name:
                        # Create folder
                        person_dir = Config.DATA_DIR / folder_name
                        person_dir.mkdir(exist_ok=True)
                        
                        # Add to database
                        person_profile = PersonProfile(
                            person_id=str(uuid.uuid4()),
                            folder_name=folder_name,
                            display_name=display_name,
                            bio=bio,
                            birth_date=birth_date,
                            relationship=relationship,
                            contact_info=contact_info,
                            social_links={},
                            profile_image=None,
                            created_at=datetime.datetime.now()
                        )
                        try:
                            self.manager.db.add_person(person_profile)
                            st.success(f"Person '{display_name}' added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding person: {str(e)}")
                    else:
                        st.warning("Please fill in all required fields (*)")
    
    def render_gallery_page(self):
        """Render gallery page"""
        st.title("📁 Gallery")
        
        # Check if we're viewing a specific person
        selected_person_id = st.session_state.get('selected_person')
        if selected_person_id:
            people = self.manager.db.get_all_people()
            selected_person = next((p for p in people if p['person_id'] == selected_person_id), None)
            if selected_person:
                st.subheader(f"Photos of {selected_person['display_name']}")
                if st.button("← Back to All People"):
                    st.session_state['selected_person'] = None
                    st.rerun()
        
        # Search and filter
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                search_query = st.text_input("Search photos...", key="gallery_search")
            with col2:
                view_mode = st.selectbox("View", ["Grid", "List"], key="view_mode")
            with col3:
                sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Rating"], key="gallery_sort")
            with col4:
                items_per_page = st.selectbox("Per Page", [12, 24, 48], key="items_per_page")
        
        # Get entries
        page = st.session_state.get('gallery_page', 1)
        
        if selected_person_id:
            data = self.manager.get_entries_by_person(selected_person_id, page, search_query)
        else:
            # Get all entries with pagination
            offset = (page - 1) * items_per_page
            with sqlite3.connect(self.manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if search_query:
                    search_pattern = f'%{search_query}%'
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (search_pattern, search_pattern, search_pattern, items_per_page, offset))
                else:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (items_per_page, offset))
                
                entries = [dict(row) for row in cursor.fetchall()]
                
                # Get total count
                if search_query:
                    search_pattern = f'%{search_query}%'
                    cursor.execute('''
                    SELECT COUNT(*)
                    FROM album_entries ae
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ''', (search_pattern, search_pattern, search_pattern))
                else:
                    cursor.execute('SELECT COUNT(*) FROM album_entries')
                
                total_count = cursor.fetchone()[0]
                total_pages = max(1, math.ceil(total_count / items_per_page))
            
            data = {
                'entries': entries,
                'total_count': total_count,
                'total_pages': total_pages,
                'current_page': page
            }
        
        # Sort entries
        if sort_by == "Oldest":
            data['entries'].sort(key=lambda x: x.get('created_at', ''))
        elif sort_by == "Rating":
            data['entries'].sort(key=lambda x: x.get('avg_rating', 0), reverse=True)
        else:  # Newest (default)
            data['entries'].sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Display entries
        if view_mode == "Grid":
            cols = st.columns(4)
            for idx, entry in enumerate(data['entries']):
                with cols[idx % 4]:
                    self._render_gallery_item(entry)
        else:  # List view
            for entry in data['entries']:
                self._render_gallery_item_list(entry)
        
        # Pagination
        if data['total_pages'] > 1:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_numbers = []
                for i in range(1, data['total_pages'] + 1):
                    if i == 1 or i == data['total_pages'] or abs(i - page) <= 2:
                        page_numbers.append(i)
                    elif page_numbers[-1] != "...":
                        page_numbers.append("...")
                
                col_btns = st.columns(len(page_numbers) + 2)
                with col_btns[0]:
                    if page > 1 and st.button("◀", key="prev_page"):
                        st.session_state.gallery_page = page - 1
                        st.rerun()
                
                for idx, page_num in enumerate(page_numbers, 1):
                    with col_btns[idx]:
                        if page_num == "...":
                            st.markdown("...")
                        elif page_num == page:
                            st.markdown(f"**{page_num}**")
                        elif st.button(str(page_num), key=f"page_{page_num}"):
                            st.session_state.gallery_page = page_num
                            st.rerun()
                
                with col_btns[-1]:
                    if page < data['total_pages'] and st.button("▶", key="next_page"):
                        st.session_state.gallery_page = page + 1
                        st.rerun()
    
    def _render_gallery_item(self, entry):
        """Render a gallery item in grid view"""
        with st.container(border=True):
            # Image
            if entry.get('thumbnail_data_url'):
                st.image(entry['thumbnail_data_url'], use_column_width=True)
            elif entry.get('thumbnail_path'):
                thumbnail_path = Path(entry['thumbnail_path'])
                if thumbnail_path.exists():
                    data_url = self.manager.image_processor.get_image_data_url(thumbnail_path)
                    st.image(data_url, use_column_width=True)
                else:
                    st.markdown("🖼️")
            else:
                st.markdown("🖼️")
            
            # Caption
            st.markdown(f"**{entry.get('caption', 'Untitled')}**")
            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
            
            # Rating
            if entry.get('avg_rating'):
                st.markdown(UIComponents.rating_stars(entry['avg_rating'], size=15), unsafe_allow_html=True)
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️", key=f"view_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_image'] = entry['entry_id']
                    st.session_state['current_page'] = 'image_detail'
                    st.rerun()
            with col2:
                # Check if favorited
                is_favorited = entry['entry_id'] in st.session_state.get('favorites', set())
                if is_favorited:
                    if st.button("⭐", key=f"unfav_{entry['entry_id']}", use_container_width=True):
                        self.manager.remove_from_favorites(entry['entry_id'])
                        st.rerun()
                else:
                    if st.button("☆", key=f"fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
    
    def _render_gallery_item_list(self, entry):
        """Render a gallery item in list view"""
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                # Thumbnail
                if entry.get('thumbnail_data_url'):
                    st.image(entry['thumbnail_data_url'], width=100)
                elif entry.get('thumbnail_path'):
                    thumbnail_path = Path(entry['thumbnail_path'])
                    if thumbnail_path.exists():
                        data_url = self.manager.image_processor.get_image_data_url(thumbnail_path)
                        st.image(data_url, width=100)
                    else:
                        st.markdown("🖼️")
                else:
                    st.markdown("🖼️")
            
            with col2:
                st.markdown(f"### {entry.get('caption', 'Untitled')}")
                st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                if entry.get('description'):
                    st.markdown(f"*{entry.get('description', '')[:100]}...*")
                if entry.get('tags'):
                    st.markdown(UIComponents.tag_badges(entry['tags'].split(',') if isinstance(entry['tags'], str) else entry['tags'], max_display=3), unsafe_allow_html=True)
            
            with col3:
                # Rating
                if entry.get('avg_rating'):
                    st.markdown(UIComponents.rating_stars(entry['avg_rating']), unsafe_allow_html=True)
                
                # Actions
                if st.button("View Details", key=f"view_det_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_image'] = entry['entry_id']
                    st.session_state['current_page'] = 'image_detail'
                    st.rerun()
                
                # Favorite toggle
                is_favorited = entry['entry_id'] in st.session_state.get('favorites', set())
                if is_favorited:
                    if st.button("Remove Favorite", key=f"rem_fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.remove_from_favorites(entry['entry_id'])
                        st.rerun()
                else:
                    if st.button("Add Favorite", key=f"add_fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
    
    def render_image_detail_page(self):
        """Render image detail page"""
        entry_id = st.session_state.get('selected_image')
        if not entry_id:
            st.error("No image selected")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return
        
        entry = self.manager.get_entry_with_details(entry_id)
        if not entry:
            st.error("Image not found")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return
        
        # Back button
        if st.button("← Back"):
            st.session_state['current_page'] = 'gallery'
            st.rerun()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            if entry.get('image_data_url'):
                st.image(entry['image_data_url'], use_column_width=True)
            else:
                # Try to load from filepath
                if entry.get('filepath'):
                    image_path = Config.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        data_url = self.manager.image_processor.get_image_data_url(image_path)
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
            st.markdown(UIComponents.rating_stars(avg_rating), unsafe_allow_html=True)
            st.caption(f"{rating_count} ratings")
            
            # User rating
            st.subheader("Your Rating")
            user_rating = entry.get('user_rating', 0)
            rating_cols = st.columns(5)
            for i in range(1, 6):
                with rating_cols[i-1]:
                    if st.button(f"{i} ⭐", key=f"rate_{i}", use_container_width=True):
                        self.manager.add_rating_to_entry(entry_id, i)
                        st.rerun()
            
            # Favorite button
            is_favorited = entry.get('is_favorited', False)
            if is_favorited:
                if st.button("⭐ Remove Favorite", use_container_width=True):
                    self.manager.remove_from_favorites(entry_id)
                    st.rerun()
            else:
                if st.button("☆ Add Favorite", use_container_width=True):
                    self.manager.add_to_favorites(entry_id)
                    st.rerun()
            
            # Metadata
            with st.expander("📊 Metadata"):
                if entry.get('description'):
                    st.markdown(f"**Description:** {entry['description']}")
                if entry.get('location'):
                    st.markdown(f"**Location:** {entry['location']}")
                if entry.get('date_taken'):
                    st.markdown(f"**Date Taken:** {entry['date_taken']}")
                if entry.get('tags'):
                    st.markdown("**Tags:**")
                    st.markdown(UIComponents.tag_badges(
                        entry['tags'] if isinstance(entry['tags'], list) else entry['tags'].split(','),
                        max_display=10
                    ), unsafe_allow_html=True)
                if entry.get('file_size'):
                    file_size_mb = entry['file_size'] / (1024 * 1024)
                    st.markdown(f"**File Size:** {file_size_mb:.2f} MB")
                if entry.get('format'):
                    st.markdown(f"**Format:** {entry['format']}")
                if entry.get('dimensions'):
                    st.markdown(f"**Dimensions:** {entry['dimensions']}")
            
            # Comments section
            st.divider()
            st.subheader("💬 Comments")
            
            # Add comment form
            with st.form(key="add_comment_form"):
                comment_text = st.text_area("Add a comment...", height=100, max_chars=Config.MAX_COMMENT_LENGTH)
                submitted = st.form_submit_button("Post Comment")
                if submitted and comment_text.strip():
                    if self.manager.add_comment_to_entry(entry_id, comment_text.strip()):
                        st.rerun()
            
            # Display comments
            comments = entry.get('comments', [])
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
    
    def render_favorites_page(self):
        """Render favorites page"""
        st.title("⭐ Favorites")
        favorites = self.manager.get_user_favorites()
        
        if not favorites:
            st.info("You haven't added any favorites yet.")
            st.markdown("Browse the gallery and click the ☆ button to add favorites!")
            return
        
        # Display favorites
        cols = st.columns(4)
        for idx, entry in enumerate(favorites):
            with cols[idx % 4]:
                self._render_gallery_item(entry)
    
    def render_search_page(self):
        """Render search page"""
        st.title("🔍 Search")
        
        # Search options
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search term", key="global_search")
        with col2:
            search_in = st.selectbox(
                "Search in",
                ["All Fields", "Captions", "Descriptions", "Tags", "People"]
            )
        
        if search_query:
            with st.spinner("Searching..."):
                # Perform search
                results = []
                
                if search_in in ["All Fields", "Captions", "Descriptions", "Tags"]:
                    db_results = self.manager.db.search_entries(search_query)
                    results.extend(db_results)
                
                if search_in in ["All Fields", "People"]:
                    people = self.manager.db.get_all_people()
                    for person in people:
                        if search_query.lower() in person['display_name'].lower():
                            # Get person's entries
                            with sqlite3.connect(self.manager.db.db_path) as conn:
                                conn.row_factory = sqlite3.Row
                                cursor = conn.cursor()
                                cursor.execute('''
                                SELECT ae.*, p.display_name, i.filename, i.thumbnail_path
                                FROM album_entries ae
                                JOIN people p ON ae.person_id = p.person_id
                                JOIN images i ON ae.image_id = i.image_id
                                WHERE ae.person_id = ?
                                LIMIT 5
                                ''', (person['person_id'],))
                                person_entries = [dict(row) for row in cursor.fetchall()]
                                results.extend(person_entries)
                
                # Remove duplicates
                seen_ids = set()
                unique_results = []
                for result in results:
                    if result['entry_id'] not in seen_ids:
                        seen_ids.add(result['entry_id'])
                        unique_results.append(result)
                
                # Display results
                st.subheader(f"Found {len(unique_results)} results")
                if unique_results:
                    for entry in unique_results:
                        self._render_gallery_item_list(entry)
                else:
                    st.info("No results found")
        else:
            st.info("Enter a search term to find photos")
    
    def render_statistics_page(self):
        """Render statistics page"""
        st.title("📊 Statistics")
        
        # Get all data
        all_people = self.manager.get_all_people_with_stats()
        if not all_people:
            st.info("No data available. Scan your directory first.")
            return
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        total_images = sum(p['image_count'] for p in all_people)
        total_comments = sum(p['comment_count'] for p in all_people)
        avg_ratings = [p['avg_rating'] for p in all_people if p['avg_rating'] > 0]
        avg_rating = sum(avg_ratings) / len(avg_ratings) if avg_ratings else 0
        
        with col1:
            st.metric("Total People", len(all_people))
        with col2:
            st.metric("Total Images", total_images)
        with col3:
            st.metric("Total Comments", total_comments)
        with col4:
            st.metric("Average Rating", f"{avg_rating:.1f}")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Images per Person")
            # Prepare data for chart
            people_names = [p['display_name'] for p in all_people]
            image_counts = [p['image_count'] for p in all_people]
            
            if image_counts:
                chart_data = pd.DataFrame({
                    'Person': people_names,
                    'Images': image_counts
                })
                st.bar_chart(chart_data.set_index('Person'))
        
        with col2:
            st.subheader("⭐ Ratings Distribution")
            # Get all ratings
            with sqlite3.connect(self.manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT rating_value, COUNT(*) as count
                FROM ratings
                GROUP BY rating_value
                ORDER BY rating_value
                ''')
                rating_data = cursor.fetchall()
            
            if rating_data:
                ratings = [r[0] for r in rating_data]
                counts = [r[1] for r in rating_data]
                chart_data = pd.DataFrame({
                    'Rating': ratings,
                    'Count': counts
                })
                st.bar_chart(chart_data.set_index('Rating'))
        
        # Detailed table
        st.divider()
        st.subheader("👥 People Details")
        table_data = []
        for person in all_people:
            table_data.append({
                'Name': person['display_name'],
                'Images': person['image_count'],
                'Comments': person['comment_count'],
                'Avg Rating': f"{person['avg_rating']:.1f}",
                'Last Activity': person.get('last_activity', 'Never')
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
            # Export button
            if st.button("📊 Export Statistics"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="photo_album_stats.csv",
                    mime="text/csv"
                )
    
    def render_settings_page(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["Application", "User", "Database", "Advanced", "Performance"])
        
        with tabs[0]:  # Application
            st.subheader("Application Settings")
            
            # Directory settings
            with st.expander("📁 Directory Settings"):
                st.info(f"**Data Directory:** {Config.DATA_DIR}")
                st.info(f"**Thumbnail Directory:** {Config.THUMBNAIL_DIR}")
                st.info(f"**Database Directory:** {Config.DB_DIR}")
                
                if st.button("Create Missing Directories"):
                    Config.init_directories()
                    st.success("Directories created/verified!")
            
            # Image settings
            with st.expander("🖼️ Image Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    new_thumb_size = st.number_input("Thumbnail Width", value=Config.THUMBNAIL_SIZE[0], min_value=100, max_value=800)
                    Config.THUMBNAIL_SIZE = (int(new_thumb_size), int(new_thumb_size))
                with col2:
                    new_preview_size = st.number_input("Preview Width", value=Config.PREVIEW_SIZE[0], min_value=400, max_value=1200)
                    Config.PREVIEW_SIZE = (int(new_preview_size), int(new_preview_size))
                
                if st.button("Apply Image Settings"):
                    st.success("Image settings updated!")
            
            # Gallery settings
            with st.expander("📁 Gallery Settings"):
                items_per_page = st.number_input("Items per Page", value=Config.ITEMS_PER_PAGE, min_value=1, max_value=100)
                grid_columns = st.number_input("Grid Columns", value=Config.GRID_COLUMNS, min_value=1, max_value=8)
                Config.ITEMS_PER_PAGE = int(items_per_page)
                Config.GRID_COLUMNS = int(grid_columns)
                
                if st.button("Apply Gallery Settings"):
                    st.success("Gallery settings updated!")
        
        with tabs[1]:  # User
            st.subheader("User Settings")
            
            current_username = st.session_state['username']
            current_role = st.session_state['user_role']
            
            new_username = st.text_input("Username", value=current_username)
            new_role = st.selectbox(
                "Role",
                [role.value for role in UserRoles],
                index=[role.value for role in UserRoles].index(current_role)
            )
            
            if st.button("Update User Settings"):
                st.session_state['username'] = new_username
                st.session_state['user_role'] = new_role
                st.success("User settings updated!")
        
        with tabs[2]:  # Database
            st.subheader("Database Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Rebuild Database", help="Recreate all database tables (data will be preserved)"):
                    try:
                        self.manager.db._init_database()
                        st.success("Database rebuilt successfully!")
                    except Exception as e:
                        st.error(f"Error rebuilding database: {str(e)}")
                
                if st.button("📊 Optimize Database", help="Run database optimization commands"):
                    try:
                        with self.manager.db.pool.get_connection() as conn:
                            conn.execute("VACUUM")
                            conn.execute("ANALYZE")
                        st.success("Database optimized!")
                    except Exception as e:
                        st.error(f"Error optimizing database: {str(e)}")
            
            with col2:
                if st.button("🗑️ Clear All Data", help="WARNING: This will delete all data!"):
                    if st.checkbox("I understand this will delete ALL data"):
                        try:
                            os.remove(self.manager.db.db_path)
                            self.manager.db = DatabaseManager()
                            st.success("All data cleared!")
                        except Exception as e:
                            st.error(f"Error clearing data: {str(e)}")
                
                db_size = self.manager.db.db_path.stat().st_size if self.manager.db.db_path.exists() else 0
                st.info(f"**Database Size:** {db_size / (1024*1024):.2f} MB")
        
        with tabs[3]:  # Advanced
            st.subheader("Advanced Settings")
            
            # Cache settings
            with st.expander("💾 Cache Settings"):
                cache_ttl = st.number_input("Cache TTL (seconds)", value=Config.CACHE_TTL, min_value=60, max_value=86400)
                Config.CACHE_TTL = int(cache_ttl)
                
                cache_size = len(self.manager.cache._cache)
                st.info(f"**Cache Items:** {cache_size}")
                
                if st.button("Clear Cache"):
                    self.manager.cache.clear()
                    st.success("Cache cleared!")
            
            # Performance settings
            with st.expander("⚡ Performance"):
                st.checkbox("Enable image lazy loading", value=True)
                st.checkbox("Enable background processing", value=Config.BACKGROUND_PROCESSING_ENABLED)
                st.checkbox("Use compression for thumbnails", value=True)
            
            # Debug settings
            with st.expander("🐛 Debug"):
                debug_mode = st.checkbox("Enable debug mode", value=False)
                if debug_mode:
                    st.warning("Debug mode enabled. Additional logging will be shown.")
                
                if st.button("Show Session State"):
                    st.write(st.session_state)
                
                if st.button("Show Configuration"):
                    st.write({
                        'APP_NAME': Config.APP_NAME,
                        'VERSION': Config.VERSION,
                        'DATA_DIR': str(Config.DATA_DIR),
                        'THUMBNAIL_SIZE': Config.THUMBNAIL_SIZE,
                        'ITEMS_PER_PAGE': Config.ITEMS_PER_PAGE,
                        'CACHE_TTL': Config.CACHE_TTL,
                        'MAX_WORKERS': Config.MAX_WORKERS,
                        'BATCH_SIZE': Config.BATCH_SIZE
                    })
        
        with tabs[4]:  # Performance
            st.subheader("Performance Optimization")
            
            st.info("Current Performance Settings:")
            st.metric("Max Workers", Config.MAX_WORKERS)
            st.metric("Batch Size", Config.BATCH_SIZE)
            st.metric("Cache Max Size", Config.CACHE_MAX_SIZE)
            st.metric("Image Cache Max Size", Config.IMAGE_CACHE_MAX_SIZE)
            
            if PSUTIL_AVAILABLE:
                st.divider()
                st.subheader("System Resources")
                
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CPU Usage", f"{cpu_percent}%")
                with col2:
                    st.metric("Memory Usage", f"{memory.percent}%")
                with col3:
                    st.metric("Available Memory", f"{memory.available / (1024**3):.2f} GB")
            
            st.divider()
            st.subheader("Optimization Recommendations")
            
            recommendations = [
                "Use connection pooling for database operations",
                "Enable lazy loading for large galleries",
                "Use batch processing for image operations",
                "Implement efficient caching strategies",
                "Monitor system resources and adjust accordingly"
            ]
            
            for rec in recommendations:
                st.text(f"✓ {rec}")
    
    def render_import_export_page(self):
        """Render import/export page"""
        st.title("📤 Import/Export")
        
        tabs = st.tabs(["Export", "Import", "Backup"])
        
        with tabs[0]:  # Export
            st.subheader("Export Data")
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel"],
                key="export_format"
            )
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                export_people = st.multiselect(
                    "Select People",
                    [p['display_name'] for p in self.manager.db.get_all_people()],
                    key="export_people"
                )
            with col2:
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.date.today() - datetime.timedelta(days=365), datetime.date.today()),
                    key="export_dates"
                )
            
            if st.button("Generate Export", type="primary"):
                with st.spinner("Generating export file..."):
                    export_data = self.manager.export_data(export_format.lower())
                    if export_data:
                        file_ext = export_format.lower()
                        if file_ext == "excel":
                            file_ext = "xlsx"
                        
                        st.download_button(
                            label=f"Download {export_format} File",
                            data=export_data,
                            file_name=f"photo_album_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime={
                                "csv": "text/csv",
                                "json": "application/json",
                                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            }[export_format.lower()]
                        )
        
        with tabs[1]:  # Import
            st.subheader("Import Data")
            
            st.info("""
            **Import Format:** CSV, JSON, or Excel files with the following columns:
            - `caption` (required): Photo caption
            - `person_name` (required): Name of the person
            - `filename` (required): Image filename (must exist in person's folder)
            - `description`: Photo description
            - `location`: Location where photo was taken
            - `tags`: Comma-separated tags
            - `privacy_level`: public/private
            """)
            
            import_file = st.file_uploader(
                "Choose import file",
                type=['csv', 'json', 'xlsx', 'xls'],
                key="import_file"
            )
            
            if import_file:
                file_ext = import_file.name.split('.')[-1].lower()
                if file_ext == 'xls':
                    file_ext = 'excel'
                
                if st.button("Import Data", type="primary"):
                    with st.spinner("Importing data..."):
                        success = self.manager.import_data(import_file.getvalue(), file_ext)
                        if success:
                            st.success("Import completed successfully!")
                            st.rerun()
        
        with tabs[2]:  # Backup
            st.subheader("Backup & Restore")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Create Backup")
                
                backup_name = st.text_input(
                    "Backup Name",
                    value=f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    key="backup_name"
                )
                
                if st.button("🔒 Create Backup", use_container_width=True):
                    # Create backup of database
                    backup_path = Config.EXPORT_DIR / f"{backup_name}.db"
                    import shutil
                    shutil.copy2(self.manager.db.db_path, backup_path)
                    st.success(f"Backup created: {backup_path}")
                    
                    # Offer download
                    with open(backup_path, 'rb') as f:
                        st.download_button(
                            label="Download Backup",
                            data=f,
                            file_name=backup_path.name,
                            mime="application/x-sqlite3"
                        )
            
            with col2:
                st.markdown("### Restore Backup")
                
                backup_file = st.file_uploader(
                    "Choose backup file",
                    type=['db', 'sqlite', 'sqlite3'],
                    key="restore_file"
                )
                
                if backup_file:
                    if st.button("⚠️ Restore Backup", type="secondary", use_container_width=True):
                        st.warning("This will replace your current database!")
                        if st.checkbox("I understand this will overwrite all current data"):
                            try:
                                # Create backup of current database first
                                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                current_backup = Config.EXPORT_DIR / f"pre_restore_{timestamp}.db"
                                shutil.copy2(self.manager.db.db_path, current_backup)
                                
                                # Write uploaded file
                                with open(self.manager.db.db_path, 'wb') as f:
                                    f.write(backup_file.getvalue())
                                
                                # Reinitialize manager
                                self.manager = AlbumManager()
                                st.success("Backup restored successfully!")
                                st.info(f"Previous database backed up to: {current_backup}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error restoring backup: {str(e)}")
    
    def render_main(self):
        """Main application renderer"""
        # Periodically clean up expired session state
        self.manager.session_manager.clear_expired()
        
        # Render sidebar
        self.render_sidebar()
        
        # Get current page
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Render appropriate page
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'people':
            self.render_people_page()
        elif current_page == 'gallery':
            self.render_gallery_page()
        elif current_page == 'image_detail':
            self.render_image_detail_page()
        elif current_page == 'favorites':
            self.render_favorites_page()
        elif current_page == 'search':
            self.render_search_page()
        elif current_page == 'statistics':
            self.render_statistics_page()
        elif current_page == 'settings':
            self.render_settings_page()
        elif current_page == 'import_export':
            self.render_import_export_page()
        else:
            self.render_dashboard()
        
        # Footer
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")
            st.caption("A comprehensive photo album management system")
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                st.caption(f"Memory: {memory.percent}% | Cache: {len(self.manager.cache._cache)} items")
    
    def cleanup(self):
        """Cleanup resources before exit"""
        if Config.BACKGROUND_PROCESSING_ENABLED:
            self.manager.background_processor.stop()
        self.manager.db.pool.close_all()
        self.manager.image_processor.clear_cache()

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================
def main():
    """Main application entry point"""
    try:
        # Initialize app
        app = PhotoAlbumApp()
        
        # Check initialization
        if not app.initialized:
            st.error("Application failed to initialize. Check directory permissions.")
            return
        
        # Render main app
        app.render_main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the logs for details.")
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.exception(e)
        
        # Recovery options
        if st.button("Try Again"):
            st.rerun()
        
        if st.button("Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    finally:
        # Cleanup on exit
        if 'app' in locals():
            app.cleanup()

# ============================================================================
# REQUIREMENTS GENERATOR
# ============================================================================
def generate_requirements():
    """Generate requirements.txt content"""
    requirements = """# Core dependencies
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0

# Performance optimizations
numba>=0.57.0
psutil>=5.9.0
joblib>=1.3.0

# AI/ML dependencies (optional)
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.19.0
accelerate>=0.21.0

# Image processing (optional)
opencv-python>=4.8.0
scipy>=1.11.0

# Database
sqlalchemy>=2.0.0

# Utilities
dask>=2023.7.0
"""
    return requirements

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    # Show requirements in sidebar for easy access
    with st.sidebar.expander("📋 Requirements"):
        st.code(generate_requirements(), language="text")
    
    if not AI_AVAILABLE:
        st.warning("""
        ⚠️ AI libraries not installed.
        Running in basic mode.
        To enable AI features:
        ```bash
        pip install torch torchvision transformers diffusers
        ```
        """)
    
    # Run main application
    main()
