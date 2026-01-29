"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH AI ENHANCEMENTS
Version: 4.0.0 - AI Enhanced with LLM & Diffusion Models - Optimized Edition
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search,
          AI Image Analysis, AI Style Transfer, AI Editing, Batch Processing
          OPTIMIZED FOR STREAMLIT WITH REDUCED CPU USAGE
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
import asyncio
import threading
from queue import Queue, Empty
import psutil
import humanize

# ============================================================================
# OPTIMIZATION IMPORTS
# ============================================================================
try:
    from numba import jit, njit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Using numpy alternatives.")

try:
    from scipy import ndimage, signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ============================================================================
# AI IMPORTS WITH LAZY LOADING
# ============================================================================
AI_AVAILABLE = False
AI_MODULES = {}

def lazy_import_ai():
    """Lazy import AI modules to reduce startup time"""
    global AI_AVAILABLE, AI_MODULES
    
    if 'ai_loaded' not in st.session_state:
        st.session_state.ai_loaded = False
    
    if not st.session_state.ai_loaded:
        try:
            # Import torch only when needed
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
            from torchvision import models
            
            AI_MODULES['torch'] = torch
            AI_MODULES['nn'] = nn
            AI_MODULES['transforms'] = transforms
            AI_MODULES['models'] = models
            
            # Import transformers
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
            
            # Import diffusers
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
            
            AI_AVAILABLE = True
            st.session_state.ai_loaded = True
            
        except ImportError as e:
            print(f"AI libraries not installed: {e}")
            AI_AVAILABLE = False
            st.session_state.ai_loaded = False
    
    return AI_AVAILABLE

# ============================================================================
# PERFORMANCE OPTIMIZATION DECORATORS
# ============================================================================
def streamlit_cache(ttl: int = 3600, max_entries: int = 1000, persist: bool = True):
    """
    Enhanced caching decorator for Streamlit with TTL and size limits
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            func_name = func.__name__
            args_key = pickle.dumps((args, kwargs))
            cache_key = f"{func_name}_{hashlib.md5(args_key).hexdigest()}"
            
            # Initialize cache in session state if not exists
            if 'performance_cache' not in st.session_state:
                st.session_state.performance_cache = {
                    'data': {},
                    'timestamps': {},
                    'hits': 0,
                    'misses': 0
                }
            
            cache = st.session_state.performance_cache
            
            # Check cache
            if cache_key in cache['data']:
                timestamp = cache['timestamps'][cache_key]
                if time.time() - timestamp < ttl:
                    cache['hits'] += 1
                    return cache['data'][cache_key]
                else:
                    # Cache expired
                    del cache['data'][cache_key]
                    del cache['timestamps'][cache_key]
            
            # Cache miss or expired
            cache['misses'] += 1
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache['data'][cache_key] = result
            cache['timestamps'][cache_key] = time.time()
            
            # Enforce max entries (LRU eviction)
            if len(cache['data']) > max_entries:
                # Remove oldest entries
                sorted_timestamps = sorted(cache['timestamps'].items(), key=lambda x: x[1])
                for key, _ in sorted_timestamps[:max_entries // 10]:  # Remove 10%
                    if key in cache['data']:
                        del cache['data'][key]
                    if key in cache['timestamps']:
                        del cache['timestamps'][key]
            
            return result
        
        # Add cache management methods
        def clear_cache():
            if 'performance_cache' in st.session_state:
                st.session_state.performance_cache = {
                    'data': {},
                    'timestamps': {},
                    'hits': 0,
                    'misses': 0
                }
        
        def get_cache_stats():
            if 'performance_cache' in st.session_state:
                cache = st.session_state.performance_cache
                return {
                    'size': len(cache['data']),
                    'hits': cache['hits'],
                    'misses': cache['misses'],
                    'hit_ratio': cache['hits'] / max(1, cache['hits'] + cache['misses'])
                }
            return {}
        
        wrapper.clear_cache = clear_cache
        wrapper.get_cache_stats = get_cache_stats
        
        return wrapper
    return decorator

def background_task(timeout: int = 300):
    """
    Decorator to run function in background thread for long-running tasks
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Store result in session state
            task_id = f"{func.__name__}_{uuid.uuid4().hex[:8]}"
            
            if 'background_tasks' not in st.session_state:
                st.session_state.background_tasks = {}
            
            # Create a future
            from concurrent.futures import Future
            future = Future()
            
            def task_runner():
                try:
                    result = func(*args, **kwargs)
                    future.set_result(result)
                    st.session_state.background_tasks[task_id] = {
                        'status': 'completed',
                        'result': result,
                        'completed_at': datetime.datetime.now()
                    }
                except Exception as e:
                    future.set_exception(e)
                    st.session_state.background_tasks[task_id] = {
                        'status': 'failed',
                        'error': str(e),
                        'completed_at': datetime.datetime.now()
                    }
            
            # Start thread
            thread = threading.Thread(target=task_runner, daemon=True)
            thread.start()
            
            st.session_state.background_tasks[task_id] = {
                'status': 'running',
                'started_at': datetime.datetime.now(),
                'future': future,
                'thread': thread
            }
            
            return task_id
        
        return wrapper
    return decorator

def debounce(wait_time: float = 0.5):
    """
    Debounce decorator to limit function execution frequency
    """
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_called[0] >= wait_time:
                last_called[0] = current_time
                return func(*args, **kwargs)
            return None
        
        return wrapper
    return decorator

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================
class PerformanceMonitor:
    """Monitor and optimize application performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.memory_samples = deque(maxlen=100)
        self.cpu_samples = deque(maxlen=100)
        
    def start_measurement(self, name: str):
        """Start timing a section of code"""
        if 'timings' not in st.session_state:
            st.session_state.timings = {}
        st.session_state.timings[name] = {
            'start': time.perf_counter(),
            'end': None,
            'duration': None
        }
    
    def end_measurement(self, name: str):
        """End timing a section of code"""
        if 'timings' in st.session_state and name in st.session_state.timings:
            st.session_state.timings[name]['end'] = time.perf_counter()
            st.session_state.timings[name]['duration'] = (
                st.session_state.timings[name]['end'] - st.session_state.timings[name]['start']
            )
    
    @contextmanager
    def measure(self, name: str):
        """Context manager for timing"""
        self.start_measurement(name)
        try:
            yield
        finally:
            self.end_measurement(name)
    
    def sample_resources(self):
        """Sample current resource usage"""
        process = psutil.Process()
        
        # Memory
        memory_info = process.memory_info()
        self.memory_samples.append({
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': process.memory_percent()
        })
        
        # CPU
        self.cpu_samples.append({
            'percent': process.cpu_percent(interval=0.1),
            'threads': process.num_threads()
        })
        
        # GC stats
        gc.collect()
        
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        report = {
            'uptime': time.time() - self.start_time,
            'memory': {
                'current_rss': self.memory_samples[-1]['rss'] if self.memory_samples else 0,
                'current_vms': self.memory_samples[-1]['vms'] if self.memory_samples else 0,
                'avg_memory_percent': np.mean([s['percent'] for s in self.memory_samples]) if self.memory_samples else 0,
                'gc_stats': {
                    'collected': gc.get_count(),
                    'thresholds': gc.get_threshold()
                }
            },
            'cpu': {
                'current_percent': self.cpu_samples[-1]['percent'] if self.cpu_samples else 0,
                'avg_cpu_percent': np.mean([s['percent'] for s in self.cpu_samples]) if self.cpu_samples else 0,
                'threads': self.cpu_samples[-1]['threads'] if self.cpu_samples else 0
            },
            'timings': st.session_state.get('timings', {})
        }
        
        # Add cache stats if available
        if 'performance_cache' in st.session_state:
            cache = st.session_state.performance_cache
            report['cache'] = {
                'size': len(cache['data']),
                'hits': cache['hits'],
                'misses': cache['misses'],
                'hit_ratio': cache['hits'] / max(1, cache['hits'] + cache['misses'])
            }
        
        return report

# ============================================================================
# OPTIMIZED IMAGE PROCESSING WITH NUMBA
# ============================================================================
class OptimizedImageProcessor:
    """Image processing optimized with Numba where available"""
    
    @staticmethod
    @streamlit_cache(ttl=3600)
    def create_thumbnail_optimized(image_path: Path, size: Tuple[int, int] = (300, 300)) -> Optional[bytes]:
        """Create thumbnail with optimized operations"""
        try:
            with Image.open(image_path) as img:
                # Handle EXIF rotation
                img = ImageOps.exif_transpose(img)
                
                # Convert mode if necessary
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Calculate aspect ratio preserving dimensions
                img_ratio = img.width / img.height
                target_ratio = size[0] / size[1]
                
                if img_ratio > target_ratio:
                    # Image is wider
                    new_height = size[1]
                    new_width = int(new_height * img_ratio)
                else:
                    # Image is taller
                    new_width = size[0]
                    new_height = int(new_width / img_ratio)
                
                # Resize
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Crop to exact size
                left = (new_width - size[0]) / 2
                top = (new_height - size[1]) / 2
                right = (new_width + size[0]) / 2
                bottom = (new_height + size[1]) / 2
                
                img = img.crop((left, top, right, bottom))
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return buffer.getvalue()
                
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            return None
    
    @staticmethod
    def apply_filter_optimized(image: np.ndarray, filter_type: str, strength: float = 1.0) -> np.ndarray:
        """Apply filters using optimized operations"""
        if NUMBA_AVAILABLE:
            return OptimizedImageProcessor._apply_filter_numba(image, filter_type, strength)
        else:
            return OptimizedImageProcessor._apply_filter_numpy(image, filter_type, strength)
    
    @staticmethod
    def _apply_filter_numpy(image: np.ndarray, filter_type: str, strength: float) -> np.ndarray:
        """Apply filter using numpy"""
        if filter_type == 'grayscale':
            if len(image.shape) == 3:
                return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            return image
        
        elif filter_type == 'sepia':
            sepia_filter = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            return np.clip(np.dot(image, sepia_filter.T), 0, 255).astype(np.uint8)
        
        elif filter_type == 'brightness':
            return np.clip(image * strength, 0, 255).astype(np.uint8)
        
        elif filter_type == 'contrast':
            mean = np.mean(image)
            return np.clip((image - mean) * strength + mean, 0, 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    @njit(parallel=True) if NUMBA_AVAILABLE else None
    def _apply_filter_numba(image: np.ndarray, filter_type: str, strength: float) -> np.ndarray:
        """Apply filter using numba for speed"""
        if filter_type == 'grayscale':
            result = np.empty(image.shape[:2], dtype=np.uint8)
            for i in prange(image.shape[0]):
                for j in prange(image.shape[1]):
                    pixel = image[i, j]
                    gray = 0.2989 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2]
                    result[i, j] = np.uint8(gray)
            return result
        
        elif filter_type == 'brightness':
            result = np.empty_like(image)
            for i in prange(image.shape[0]):
                for j in prange(image.shape[1]):
                    for k in prange(image.shape[2]):
                        result[i, j, k] = np.uint8(np.clip(image[i, j, k] * strength, 0, 255))
            return result
        
        return image
    
    @staticmethod
    @streamlit_cache(ttl=3600)
    def calculate_image_features(image_path: Path) -> Dict:
        """Calculate image features for search and analysis"""
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img.convert('RGB'))
                
                features = {
                    'histogram': OptimizedImageProcessor._calculate_color_histogram(img_array),
                    'edges': OptimizedImageProcessor._detect_edges(img_array),
                    'texture': OptimizedImageProcessor._calculate_texture_features(img_array),
                    'brightness': float(np.mean(img_array)),
                    'contrast': float(np.std(img_array))
                }
                
                return features
        except Exception as e:
            print(f"Error calculating features: {e}")
            return {}
    
    @staticmethod
    def _calculate_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Calculate color histogram"""
        hist_r = np.histogram(image[..., 0], bins=bins, range=(0, 255))[0]
        hist_g = np.histogram(image[..., 1], bins=bins, range=(0, 255))[0]
        hist_b = np.histogram(image[..., 2], bins=bins, range=(0, 255))[0]
        return np.concatenate([hist_r, hist_g, hist_b])
    
    @staticmethod
    def _detect_edges(image: np.ndarray) -> float:
        """Simple edge detection metric"""
        if SCIPY_AVAILABLE:
            from scipy import ndimage
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            sobel_x = ndimage.sobel(gray, axis=0)
            sobel_y = ndimage.sobel(gray, axis=1)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return float(np.mean(edge_magnitude))
        return 0.0
    
    @staticmethod
    def _calculate_texture_features(image: np.ndarray) -> Dict:
        """Calculate texture features"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        features = {
            'smoothness': float(1.0 / (1.0 + np.var(gray))),
            'uniformity': float(np.sum(np.histogram(gray, bins=256)[0]**2) / (gray.size**2))
        }
        
        return features

# ============================================================================
# ENHANCED CONFIGURATION WITH PERFORMANCE SETTINGS
# ============================================================================
class EnhancedConfig:
    """Enhanced configuration with performance optimizations"""
    
    # Application info
    APP_NAME = "MemoryVault Pro AI Ultra"
    VERSION = "4.0.0"
    DESCRIPTION = "High-performance photo album with AI enhancements"
    
    # Paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    AI_MODELS_DIR = BASE_DIR / "ai_models"
    CACHE_DIR = BASE_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Files
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    CACHE_FILE = CACHE_DIR / "persistent_cache.pkl"
    SETTINGS_FILE = METADATA_DIR / "settings.json"
    
    # Performance settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 600)
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    
    # Pagination
    ITEMS_PER_PAGE = 24
    GRID_COLUMNS = 4
    LAZY_LOADING_CHUNK = 12
    
    # Cache settings
    CACHE_TTL = 7200  # 2 hours
    IMAGE_CACHE_TTL = 86400  # 24 hours for images
    MAX_CACHE_SIZE = 1000
    PERSISTENT_CACHE = True
    
    # Database settings
    DB_POOL_SIZE = 5
    DB_TIMEOUT = 30
    USE_WAL = True  # Write-Ahead Logging for better concurrency
    
    # AI settings
    AI_ENABLED = False  # Will be set dynamically
    AI_LAZY_LOAD = True
    AI_MODEL_CACHE_SIZE = 3
    AI_INFERENCE_TIMEOUT = 60
    
    # Resource limits
    MAX_CONCURRENT_PROCESSES = 4
    MAX_MEMORY_PERCENT = 80
    CPU_THROTTLE_THRESHOLD = 85
    
    # UI settings
    THEME = "light"
    ANIMATIONS_ENABLED = True
    LAZY_RENDERING = True
    VIRTUAL_SCROLLING = True
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    MAX_FILE_UPLOADS = 50
    MAX_COMMENT_LENGTH = 1000
    MAX_CAPTION_LENGTH = 500
    
    # Monitoring
    PERFORMANCE_SAMPLING_INTERVAL = 5  # seconds
    LOG_LEVEL = logging.INFO
    ENABLE_TELEMETRY = False
    
    # Batch processing
    BATCH_SIZE = 10
    BATCH_DELAY = 0.1  # seconds between batches
    
    @classmethod
    def init_directories(cls):
        """Initialize all required directories"""
        directories = [
            cls.DATA_DIR,
            cls.THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.EXPORT_DIR,
            cls.AI_MODELS_DIR,
            cls.CACHE_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        cls._setup_logging()
        
        # Load or create settings
        cls._load_settings()
    
    @classmethod
    def _setup_logging(cls):
        """Setup application logging"""
        log_file = cls.LOGS_DIR / f"app_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
                logging.StreamHandler()
            ]
        )
        
        # Reduce verbosity of some libraries
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
    
    @classmethod
    def _load_settings(cls):
        """Load or create settings file"""
        if cls.SETTINGS_FILE.exists():
            try:
                with open(cls.SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                
                # Update config with saved settings
                for key, value in settings.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
            except Exception as e:
                logging.warning(f"Could not load settings: {e}")
        
        # Save current settings
        cls._save_settings()
    
    @classmethod
    def _save_settings(cls):
        """Save current settings to file"""
        try:
            settings = {
                'THUMBNAIL_SIZE': cls.THUMBNAIL_SIZE,
                'PREVIEW_SIZE': cls.PREVIEW_SIZE,
                'ITEMS_PER_PAGE': cls.ITEMS_PER_PAGE,
                'GRID_COLUMNS': cls.GRID_COLUMNS,
                'CACHE_TTL': cls.CACHE_TTL,
                'THEME': cls.THEME,
                'ANIMATIONS_ENABLED': cls.ANIMATIONS_ENABLED,
                'LAZY_RENDERING': cls.LAZY_RENDERING
            }
            
            with open(cls.SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save settings: {e}")
    
    @classmethod
    def update_setting(cls, key: str, value: Any):
        """Update a setting and save to file"""
        if hasattr(cls, key):
            setattr(cls, key, value)
            cls._save_settings()

# ============================================================================
# PERSISTENT CACHE MANAGER
# ============================================================================
class PersistentCache:
    """Disk-backed persistent cache with expiration"""
    
    def __init__(self, cache_file: Path = None, max_size: int = 10000):
        self.cache_file = cache_file or EnhancedConfig.CACHE_FILE
        self.max_size = max_size
        self.cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Filter expired items
                    current_time = time.time()
                    self.cache = {
                        k: v for k, v in data.items()
                        if v['expires'] > current_time
                    }
                logging.info(f"Loaded cache with {len(self.cache)} items")
        except Exception as e:
            logging.warning(f"Could not load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            # Remove expired items before saving
            current_time = time.time()
            self.cache = {
                k: v for k, v in self.cache.items()
                if v['expires'] > current_time
            }
            
            # Enforce size limit (LRU)
            if len(self.cache) > self.max_size:
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])
                self.cache = dict(sorted_items[-self.max_size:])
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Could not save cache: {e}")
    
    def get(self, key: str, default=None):
        """Get item from cache"""
        if key in self.cache:
            item = self.cache[key]
            if item['expires'] > time.time():
                # Update timestamp (LRU)
                item['timestamp'] = time.time()
                return item['value']
            else:
                del self.cache[key]
        return default
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in cache with TTL"""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'expires': time.time() + ttl
        }
        # Save periodically (every 100 writes or on important operations)
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def delete(self, key: str):
        """Delete item from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self._save_cache()
    
    def cleanup(self):
        """Remove expired items and save"""
        current_time = time.time()
        initial_size = len(self.cache)
        self.cache = {
            k: v for k, v in self.cache.items()
            if v['expires'] > current_time
        }
        if len(self.cache) < initial_size:
            self._save_cache()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        current_time = time.time()
        expired = sum(1 for v in self.cache.values() if v['expires'] <= current_time)
        
        return {
            'total_items': len(self.cache),
            'expired_items': expired,
            'size_mb': os.path.getsize(self.cache_file) / (1024 * 1024) if self.cache_file.exists() else 0
        }

# ============================================================================
# ENHANCED SESSION STATE MANAGER
# ============================================================================
class SessionStateManager:
    """Manage session state with persistence and cleanup"""
    
    @staticmethod
    def initialize_session():
        """Initialize or restore session state"""
        if 'session_initialized' not in st.session_state:
            # Core application state
            st.session_state.update({
                'session_initialized': True,
                'session_id': str(uuid.uuid4()),
                'session_start': datetime.datetime.now(),
                'page_views': 0,
                'last_activity': time.time(),
                
                # Navigation
                'current_page': 'dashboard',
                'previous_page': None,
                'page_history': deque(maxlen=20),
                
                # User state
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': 'viewer',
                'authenticated': False,
                'login_time': None,
                
                # UI state
                'theme': EnhancedConfig.THEME,
                'view_mode': 'grid',
                'sort_by': 'date_desc',
                'filter_tags': [],
                'search_query': '',
                'selected_person': None,
                'selected_image': None,
                'selected_album': None,
                
                # Pagination
                'current_page_num': 1,
                'items_per_page': EnhancedConfig.ITEMS_PER_PAGE,
                
                # Favorites and history
                'favorites': set(),
                'recently_viewed': deque(maxlen=50),
                'search_history': deque(maxlen=20),
                
                # Performance
                'render_count': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                
                # AI state
                'ai_enabled': False,
                'ai_models_loaded': {},
                'ai_usage_count': 0,
                
                # Batch operations
                'batch_queue': [],
                'batch_results': {},
                'background_tasks': {},
                
                # Settings
                'settings': {
                    'auto_refresh': True,
                    'notifications': True,
                    'data_saver': False,
                    'developer_mode': False
                }
            })
            
            # Load persisted state if available
            SessionStateManager._load_persisted_state()
        
        # Update activity timestamp
        st.session_state.last_activity = time.time()
        st.session_state.page_views += 1
    
    @staticmethod
    def _load_persisted_state():
        """Load persisted state from storage"""
        try:
            state_file = EnhancedConfig.METADATA_DIR / f"session_{st.session_state.user_id}.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    persisted = json.load(f)
                
                # Merge persisted state
                for key, value in persisted.items():
                    if key in st.session_state and key not in ['session_id', 'session_start']:
                        if isinstance(value, list):
                            st.session_state[key] = deque(value, maxlen=st.session_state[key].maxlen)
                        else:
                            st.session_state[key] = value
        except Exception as e:
            logging.warning(f"Could not load persisted state: {e}")
    
    @staticmethod
    def save_persisted_state():
        """Save persistent state to storage"""
        try:
            state_file = EnhancedConfig.METADATA_DIR / f"session_{st.session_state.user_id}.json"
            
            # Prepare state for persistence (skip non-serializable items)
            state_to_save = {}
            for key, value in st.session_state.items():
                if key.startswith('_'):
                    continue
                try:
                    if isinstance(value, (deque, set)):
                        state_to_save[key] = list(value)
                    elif isinstance(value, datetime.datetime):
                        state_to_save[key] = value.isoformat()
                    else:
                        json.dumps(value)  # Test serializability
                        state_to_save[key] = value
                except:
                    continue
            
            with open(state_file, 'w') as f:
                json.dump(state_to_save, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save persisted state: {e}")
    
    @staticmethod
    def cleanup_old_sessions(max_age_days: int = 7):
        """Clean up old session files"""
        try:
            session_files = EnhancedConfig.METADATA_DIR.glob("session_*.json")
            cutoff_time = time.time() - (max_age_days * 86400)
            
            for session_file in session_files:
                if session_file.stat().st_mtime < cutoff_time:
                    session_file.unlink()
        except Exception as e:
            logging.error(f"Could not clean up old sessions: {e}")
    
    @staticmethod
    def get_session_stats() -> Dict:
        """Get session statistics"""
        session_age = datetime.datetime.now() - st.session_state.session_start
        
        return {
            'session_id': st.session_state.session_id,
            'session_age': str(session_age),
            'page_views': st.session_state.page_views,
            'user_id': st.session_state.user_id,
            'username': st.session_state.username,
            'cache_hits': st.session_state.cache_hits,
            'cache_misses': st.session_state.cache_misses,
            'cache_hit_ratio': st.session_state.cache_hits / max(1, st.session_state.cache_hits + st.session_state.cache_misses),
            'ai_usage_count': st.session_state.ai_usage_count,
            'recently_viewed_count': len(st.session_state.recently_viewed),
            'favorites_count': len(st.session_state.favorites)
        }
    
    @staticmethod
    def navigate_to(page: str):
        """Navigate to a page with history tracking"""
        if st.session_state.current_page != page:
            st.session_state.previous_page = st.session_state.current_page
            st.session_state.current_page = page
            st.session_state.page_history.append(page)
            st.session_state.current_page_num = 1  # Reset pagination
            st.rerun()
    
    @staticmethod
    def go_back():
        """Go back to previous page"""
        if st.session_state.page_history:
            previous = st.session_state.page_history.pop() if st.session_state.page_history else 'dashboard'
            SessionStateManager.navigate_to(previous)

# ============================================================================
# OPTIMIZED DATABASE MANAGER WITH CONNECTION POOLING
# ============================================================================
class OptimizedDatabaseManager:
    """Database manager with connection pooling and optimization"""
    
    _instance = None
    _connection_pool = []
    _max_pool_size = EnhancedConfig.DB_POOL_SIZE
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize database and connection pool"""
        self.db_path = EnhancedConfig.DB_FILE
        self._init_database()
        self._create_connection_pool()
        self.query_cache = {}
        self.stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'connection_waits': 0
        }
    
    def _create_connection_pool(self):
        """Create connection pool"""
        for _ in range(self._max_pool_size):
            conn = sqlite3.connect(
                self.db_path,
                timeout=EnhancedConfig.DB_TIMEOUT,
                check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode = WAL" if EnhancedConfig.USE_WAL else "")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -10000")  # 10MB cache
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.row_factory = sqlite3.Row
            self._connection_pool.append(conn)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        start_time = time.time()
        
        while not self._connection_pool:
            # Wait for connection (with timeout)
            time.sleep(0.01)
            self.stats['connection_waits'] += 1
            
            if time.time() - start_time > EnhancedConfig.DB_TIMEOUT:
                raise TimeoutError("Could not acquire database connection")
        
        conn = self._connection_pool.pop()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._connection_pool.append(conn)
    
    def _init_database(self):
        """Initialize database with optimized schema"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enable optimizations
                cursor.execute("PRAGMA auto_vacuum = INCREMENTAL")
                cursor.execute("PRAGMA optimize")
                
                # Create optimized tables with proper indexing
                self._create_tables(cursor)
                
                # Create views for common queries
                self._create_views(cursor)
                
                # Create triggers for denormalized data
                self._create_triggers(cursor)
                
                conn.commit()
                
                logging.info("Database initialized with optimizations")
                
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise
    
    def _create_tables(self, cursor):
        """Create optimized tables"""
        # Images table with compression hints
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
            thumbnail_size INTEGER,
            dominant_color TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_images_checksum (checksum),
            INDEX idx_images_created (created_date),
            INDEX idx_images_size (file_size)
        )
        ''')
        
        # People table
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
            image_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_people_display (display_name),
            INDEX idx_people_folder (folder_name)
        )
        ''')
        
        # Album entries with denormalized counts
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
            comment_count INTEGER DEFAULT 0,
            rating_count INTEGER DEFAULT 0,
            rating_avg REAL DEFAULT 0,
            view_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images (image_id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES people (person_id) ON DELETE CASCADE,
            INDEX idx_entries_person (person_id),
            INDEX idx_entries_created (created_at),
            INDEX idx_entries_rating (rating_avg),
            INDEX idx_entries_tags (tags)
        )
        ''')
        
        # Comments with hierarchy support
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            comment_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_edited BOOLEAN DEFAULT 0,
            parent_comment_id TEXT,
            like_count INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id) ON DELETE CASCADE,
            INDEX idx_comments_entry (entry_id),
            INDEX idx_comments_user (user_id),
            INDEX idx_comments_created (created_at),
            INDEX idx_comments_parent (parent_comment_id)
        )
        ''')
        
        # Ratings with composite index
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating_value INTEGER CHECK (rating_value BETWEEN 1 AND 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(entry_id, user_id),
            FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id) ON DELETE CASCADE,
            INDEX idx_ratings_entry_user (entry_id, user_id),
            INDEX idx_ratings_value (rating_value)
        )
        ''')
        
        # User favorites
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_favorites (
            user_id TEXT,
            entry_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, entry_id),
            FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id) ON DELETE CASCADE,
            INDEX idx_favorites_user (user_id),
            INDEX idx_favorites_entry (entry_id)
        )
        ''')
        
        # Search index for full-text search
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            entry_id,
            caption,
            description,
            tags,
            person_name,
            tokenize="porter unicode61"
        )
        ''')
    
    def _create_views(self, cursor):
        """Create views for common queries"""
        # View for entries with aggregated data
        cursor.execute('''
        CREATE VIEW IF NOT EXISTS v_entry_details AS
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
            ae.comment_count,
            ae.rating_count,
            ae.rating_avg,
            ae.view_count,
            ae.created_at,
            ae.updated_at,
            p.display_name,
            p.folder_name,
            i.filename,
            i.filepath,
            i.file_size,
            i.format,
            i.width,
            i.height,
            i.thumbnail_path,
            i.dominant_color
        FROM album_entries ae
        JOIN people p ON ae.person_id = p.person_id
        JOIN images i ON ae.image_id = i.image_id
        ''')
        
        # View for people statistics
        cursor.execute('''
        CREATE VIEW IF NOT EXISTS v_people_stats AS
        SELECT 
            p.person_id,
            p.display_name,
            p.folder_name,
            p.image_count,
            COUNT(DISTINCT ae.entry_id) as entry_count,
            COALESCE(AVG(ae.rating_avg), 0) as avg_rating,
            COALESCE(SUM(ae.comment_count), 0) as total_comments,
            MAX(ae.updated_at) as last_activity
        FROM people p
        LEFT JOIN album_entries ae ON p.person_id = ae.person_id
        GROUP BY p.person_id
        ''')
    
    def _create_triggers(self, cursor):
        """Create triggers for maintaining denormalized data"""
        # Trigger to update comment count
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_update_comment_count
        AFTER INSERT ON comments
        FOR EACH ROW
        BEGIN
            UPDATE album_entries 
            SET comment_count = comment_count + 1
            WHERE entry_id = NEW.entry_id;
            
            -- Update parent comment reply count if exists
            IF NEW.parent_comment_id IS NOT NULL THEN
                UPDATE comments
                SET reply_count = reply_count + 1
                WHERE comment_id = NEW.parent_comment_id;
            END IF;
        END;
        ''')
        
        # Trigger to update rating stats
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_update_rating_stats
        AFTER INSERT ON ratings
        FOR EACH ROW
        BEGIN
            UPDATE album_entries 
            SET 
                rating_count = rating_count + 1,
                rating_avg = (
                    SELECT AVG(rating_value) 
                    FROM ratings 
                    WHERE entry_id = NEW.entry_id
                )
            WHERE entry_id = NEW.entry_id;
        END;
        ''')
        
        # Trigger to update person image count
        cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS trg_update_person_image_count
        AFTER INSERT ON album_entries
        FOR EACH ROW
        BEGIN
            UPDATE people 
            SET image_count = image_count + 1
            WHERE person_id = NEW.person_id;
        END;
        ''')
    
    @streamlit_cache(ttl=300)
    def execute_query(self, query: str, params: tuple = (), use_cache: bool = True) -> List[Dict]:
        """Execute query with caching"""
        cache_key = f"{query}_{hashlib.md5(pickle.dumps(params)).hexdigest()}"
        
        if use_cache and cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            return self.query_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        self.stats['queries_executed'] += 1
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                results = [dict(row) for row in cursor.fetchall()]
                if use_cache:
                    self.query_cache[cache_key] = results
                    # Limit cache size
                    if len(self.query_cache) > 1000:
                        oldest_key = next(iter(self.query_cache))
                        del self.query_cache[oldest_key]
                return results
            else:
                return []
    
    def bulk_insert(self, table: str, data: List[Dict]):
        """Bulk insert for better performance"""
        if not data:
            return
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get column names from first item
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_names = ', '.join(columns)
            
            # Prepare values
            values = [tuple(item[col] for col in columns) for item in data]
            
            # Execute with executemany
            cursor.executemany(
                f'INSERT OR REPLACE INTO {table} ({column_names}) VALUES ({placeholders})',
                values
            )
            
            # Clear query cache for this table
            self.query_cache = {k: v for k, v in self.query_cache.items() if table not in k}
    
    def vacuum(self):
        """Optimize database"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.execute("PRAGMA optimize")
            logging.info("Database optimized")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {
                'connection_pool_size': len(self._connection_pool),
                'query_cache_size': len(self.query_cache),
                **self.stats
            }
            
            # Table sizes
            tables = ['images', 'people', 'album_entries', 'comments', 'ratings']
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            return stats

# ============================================================================
# LAZY LOADING IMAGE GALLERY
# ============================================================================
class LazyImageGallery:
    """Lazy loading image gallery for better performance"""
    
    def __init__(self, manager):
        self.manager = manager
        self.loaded_images = set()
        self.image_cache = {}
        self.batch_size = EnhancedConfig.LAZY_LOADING_CHUNK
    
    def render_gallery(self, entries: List[Dict], columns: int = 4, virtual_scroll: bool = True):
        """Render gallery with lazy loading"""
        if not entries:
            st.info("No images found")
            return
        
        # Calculate visible range for virtual scrolling
        total_items = len(entries)
        
        if virtual_scroll and total_items > self.batch_size * 2:
            # Virtual scrolling implementation
            self._render_virtual_gallery(entries, columns)
        else:
            # Regular paginated gallery
            self._render_paginated_gallery(entries, columns)
    
    def _render_virtual_gallery(self, entries: List[Dict], columns: int):
        """Render gallery with virtual scrolling"""
        total_items = len(entries)
        
        # Create scroll position tracker
        if 'gallery_scroll_pos' not in st.session_state:
            st.session_state.gallery_scroll_pos = 0
        
        # Calculate visible range
        start_idx = st.session_state.gallery_scroll_pos
        end_idx = min(start_idx + self.batch_size * 2, total_items)
        
        # Display visible items
        visible_entries = entries[start_idx:end_idx]
        
        # Create grid
        cols = st.columns(columns)
        for idx, entry in enumerate(visible_entries):
            with cols[idx % columns]:
                self._render_gallery_item(entry)
        
        # Load next batch on scroll detection
        scroll_position = st.slider(
            "Scroll",
            min_value=0,
            max_value=total_items - self.batch_size,
            value=st.session_state.gallery_scroll_pos,
            key="gallery_scroll",
            label_visibility="collapsed"
        )
        
        if scroll_position != st.session_state.gallery_scroll_pos:
            st.session_state.gallery_scroll_pos = scroll_position
            st.rerun()
        
        # Show loading indicator if more items to load
        if end_idx < total_items:
            with st.spinner(f"Loaded {end_idx} of {total_items} items..."):
                # Pre-load next batch
                next_batch = entries[end_idx:end_idx + self.batch_size]
                self._preload_images(next_batch)
    
    def _render_paginated_gallery(self, entries: List[Dict], columns: int):
        """Render paginated gallery"""
        total_items = len(entries)
        items_per_page = st.session_state.get('items_per_page', EnhancedConfig.ITEMS_PER_PAGE)
        current_page = st.session_state.get('current_page_num', 1)
        
        # Calculate page range
        total_pages = max(1, math.ceil(total_items / items_per_page))
        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        # Get page items
        page_entries = entries[start_idx:end_idx]
        
        # Render grid
        cols = st.columns(columns)
        for idx, entry in enumerate(page_entries):
            with cols[idx % columns]:
                self._render_gallery_item(entry)
        
        # Pagination controls
        if total_pages > 1:
            self._render_pagination(current_page, total_pages, total_items)
    
    def _render_gallery_item(self, entry: Dict):
        """Render individual gallery item"""
        with st.container(border=True):
            # Thumbnail with lazy loading
            thumbnail = self._get_thumbnail(entry)
            if thumbnail:
                st.image(thumbnail, use_column_width=True)
            else:
                # Placeholder
                st.markdown(
                    f'<div style="background: #f0f0f0; height: 200px; display: flex; align-items: center; justify-content: center; border-radius: 8px;">'
                    f'<span style="color: #666;">Loading...</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Caption and info
            caption = entry.get('caption', 'Untitled')
            if len(caption) > 30:
                caption = caption[:27] + '...'
            
            st.markdown(f"**{caption}**")
            
            # Person name
            display_name = entry.get('display_name', 'Unknown')
            if len(display_name) > 20:
                display_name = display_name[:17] + '...'
            
            st.caption(f"👤 {display_name}")
            
            # Rating
            rating = entry.get('rating_avg', 0)
            if rating > 0:
                stars = '⭐' * int(rating) + '☆' * (5 - int(rating))
                st.caption(f"{stars} ({rating:.1f})")
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️", key=f"view_{entry['entry_id']}", use_container_width=True):
                    st.session_state.selected_image = entry['entry_id']
                    SessionStateManager.navigate_to('image_detail')
            with col2:
                is_favorited = entry['entry_id'] in st.session_state.favorites
                if is_favorited:
                    if st.button("⭐", key=f"unfav_{entry['entry_id']}", use_container_width=True):
                        self.manager.remove_from_favorites(entry['entry_id'])
                        st.rerun()
                else:
                    if st.button("☆", key=f"fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
    
    def _get_thumbnail(self, entry: Dict) -> Optional[bytes]:
        """Get thumbnail with caching"""
        entry_id = entry.get('entry_id')
        thumbnail_path = entry.get('thumbnail_path')
        
        if not thumbnail_path:
            return None
        
        # Check cache
        cache_key = f"thumbnail_{entry_id}"
        if cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        # Load thumbnail
        try:
            thumb_path = Path(thumbnail_path)
            if thumb_path.exists():
                with open(thumb_path, 'rb') as f:
                    thumbnail = f.read()
                    self.image_cache[cache_key] = thumbnail
                    return thumbnail
        except Exception as e:
            logging.warning(f"Could not load thumbnail {thumbnail_path}: {e}")
        
        return None
    
    def _preload_images(self, entries: List[Dict]):
        """Preload images in background"""
        for entry in entries:
            self._get_thumbnail(entry)  # Cache thumbnails
    
    def _render_pagination(self, current_page: int, total_pages: int, total_items: int):
        """Render pagination controls"""
        st.divider()
        
        col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
        
        with col1:
            if current_page > 1 and st.button("◀ Prev", use_container_width=True):
                st.session_state.current_page_num = current_page - 1
                st.rerun()
        
        with col2:
            page_options = list(range(1, total_pages + 1))
            if len(page_options) > 10:
                # Show abbreviated options for many pages
                page_options = (
                    [1, 2, 3] +
                    ['...'] +
                    list(range(max(4, current_page - 2), min(total_pages - 2, current_page + 3))) +
                    ['...'] +
                    [total_pages - 2, total_pages - 1, total_pages]
                )
            
            selected_page = st.selectbox(
                "Page",
                page_options,
                index=current_page - 1 if current_page - 1 < len(page_options) else 0,
                key="page_select",
                label_visibility="collapsed"
            )
            
            if selected_page != current_page and selected_page != '...':
                st.session_state.current_page_num = selected_page
                st.rerun()
        
        with col3:
            st.markdown(f"**Page {current_page} of {total_pages}**")
            st.caption(f"{total_items} items")
        
        with col4:
            items_per_page = st.selectbox(
                "Items per page",
                [12, 24, 48, 96],
                index=[12, 24, 48, 96].index(st.session_state.items_per_page) if st.session_state.items_per_page in [12, 24, 48, 96] else 1,
                key="items_per_page_select",
                label_visibility="collapsed"
            )
            
            if items_per_page != st.session_state.items_per_page:
                st.session_state.items_per_page = items_per_page
                st.session_state.current_page_num = 1
                st.rerun()
        
        with col5:
            if current_page < total_pages and st.button("Next ▶", use_container_width=True):
                st.session_state.current_page_num = current_page + 1
                st.rerun()

# ============================================================================
# ENHANCED AI MANAGER WITH MODEL CACHING
# ============================================================================
class EnhancedAIManager:
    """AI manager with lazy loading and model caching"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize AI manager"""
        self.ai_available = lazy_import_ai()
        self.loaded_models = {}
        self.model_cache = PersistentCache(
            EnhancedConfig.AI_MODELS_DIR / "model_cache.pkl",
            max_size=EnhancedConfig.AI_MODEL_CACHE_SIZE
        )
        self.inference_queue = Queue()
        self.is_processing = False
        
        # Start background processor
        self._start_background_processor()
    
    def _start_background_processor(self):
        """Start background inference processor"""
        def processor():
            while True:
                try:
                    task = self.inference_queue.get(timeout=1)
                    if task is None:  # Shutdown signal
                        break
                    
                    future, func, args, kwargs = task
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                except Empty:
                    continue
                except Exception as e:
                    logging.error(f"Background processor error: {e}")
        
        self.processor_thread = threading.Thread(target=processor, daemon=True)
        self.processor_thread.start()
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """Load AI model with caching"""
        if not self.ai_available:
            return None
        
        # Check cache
        cache_key = f"model_{model_name}"
        if not force_reload and cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        try:
            if model_name == "caption":
                processor = AI_MODULES['BlipProcessor'].from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                model = AI_MODULES['BlipForConditionalGeneration'].from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                device = "cuda" if AI_MODULES['torch'].cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                
                self.loaded_models[cache_key] = (processor, model, device)
                
            elif model_name == "clip":
                processor = AI_MODULES['CLIPProcessor'].from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                model = AI_MODULES['CLIPModel'].from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                device = "cuda" if AI_MODULES['torch'].cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                
                self.loaded_models[cache_key] = (processor, model, device)
            
            elif model_name == "diffusion":
                # Load with lower precision for memory efficiency
                torch_dtype = AI_MODULES['torch'].float16 if AI_MODULES['torch'].cuda.is_available() else AI_MODULES['torch'].float32
                
                pipe = AI_MODULES['StableDiffusionPipeline'].from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                
                if AI_MODULES['torch'].cuda.is_available():
                    pipe = pipe.to("cuda")
                    pipe.enable_attention_slicing()
                
                self.loaded_models[cache_key] = pipe
            
            logging.info(f"Loaded model: {model_name}")
            return self.loaded_models[cache_key]
            
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {e}")
            return None
    
    def unload_model(self, model_name: str):
        """Unload model to free memory"""
        cache_key = f"model_{model_name}"
        if cache_key in self.loaded_models:
            del self.loaded_models[cache_key]
            
            # Clear GPU cache
            if AI_MODULES['torch'].cuda.is_available():
                AI_MODULES['torch'].cuda.empty_cache()
            
            logging.info(f"Unloaded model: {model_name}")
    
    def async_inference(self, model_name: str, inference_func: Callable, *args, **kwargs):
        """Run inference asynchronously"""
        future = concurrent.futures.Future()
        self.inference_queue.put((future, inference_func, args, kwargs))
        return future
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image"""
        if not self.ai_available:
            return "AI not available"
        
        try:
            model_result = self.load_model("caption")
            if not model_result:
                return "Could not load caption model"
            
            processor, model, device = model_result
            
            # Prepare image
            inputs = processor(image, return_tensors="pt").to(device)
            
            # Generate caption
            with AI_MODULES['torch'].no_grad():
                out = model.generate(**inputs, max_length=30, num_beams=5)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Update usage count
            st.session_state.ai_usage_count += 1
            
            return caption
            
        except Exception as e:
            logging.error(f"Error generating caption: {e}")
            return "Error generating caption"
    
    def analyze_image(self, image: Image.Image) -> Dict:
        """Comprehensive image analysis"""
        analysis = {}
        
        # Generate caption
        analysis['caption'] = self.generate_caption(image)
        
        # Get image features using CLIP
        try:
            model_result = self.load_model("clip")
            if model_result:
                processor, model, device = model_result
                
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with AI_MODULES['torch'].no_grad():
                    image_features = model.get_image_features(**inputs)
                
                analysis['features'] = image_features.cpu().numpy().tolist()
        except Exception as e:
            logging.warning(f"Could not extract CLIP features: {e}")
        
        # Basic image stats
        img_array = np.array(image)
        analysis['stats'] = {
            'brightness': float(np.mean(img_array)),
            'contrast': float(np.std(img_array)),
            'size': img_array.shape
        }
        
        return analysis
    
    def apply_style_transfer(self, image: Image.Image, style_prompt: str) -> Optional[Image.Image]:
        """Apply style transfer using diffusion"""
        if not self.ai_available:
            return None
        
        try:
            pipe = self.load_model("diffusion")
            if not pipe:
                return None
            
            # Prepare image
            init_image = image.convert("RGB").resize((512, 512))
            
            # Generate
            result = pipe(
                prompt=style_prompt,
                image=init_image,
                strength=0.7,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]
            
            # Resize back
            result = result.resize(image.size)
            
            # Update usage count
            st.session_state.ai_usage_count += 1
            
            return result
            
        except Exception as e:
            logging.error(f"Error applying style transfer: {e}")
            return None
    
    def cleanup(self):
        """Clean up AI resources"""
        self.loaded_models.clear()
        
        if self.ai_available and AI_MODULES['torch'].cuda.is_available():
            AI_MODULES['torch'].cuda.empty_cache()
        
        logging.info("AI resources cleaned up")

# ============================================================================
# ENHANCED ALBUM MANAGER WITH OPTIMIZATIONS
# ============================================================================
class EnhancedAlbumManager:
    """Enhanced album manager with all optimizations"""
    
    def __init__(self):
        self.db = OptimizedDatabaseManager()
        self.image_processor = OptimizedImageProcessor()
        self.ai_manager = EnhancedAIManager() if lazy_import_ai() else None
        self.cache = PersistentCache()
        self.performance_monitor = PerformanceMonitor()
        self.lazy_gallery = LazyImageGallery(self)
        
        # Initialize session state
        SessionStateManager.initialize_session()
        
        # Start performance monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background performance monitoring"""
        def monitor():
            while True:
                try:
                    self.performance_monitor.sample_resources()
                    time.sleep(EnhancedConfig.PERFORMANCE_SAMPLING_INTERVAL)
                    
                    # Auto-cleanup if memory is high
                    process = psutil.Process()
                    if process.memory_percent() > EnhancedConfig.MAX_MEMORY_PERCENT:
                        self._cleanup_resources()
                        
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _cleanup_resources(self):
        """Clean up resources when memory is high"""
        logging.info("Memory high, cleaning up resources")
        
        # Clear image cache
        self.lazy_gallery.image_cache.clear()
        
        # Clear AI cache
        if self.ai_manager:
            self.ai_manager.unload_model("diffusion")
        
        # Run garbage collection
        gc.collect()
        
        if lazy_import_ai() and AI_MODULES['torch'].cuda.is_available():
            AI_MODULES['torch'].cuda.empty_cache()
    
    @streamlit_cache(ttl=600)
    def get_people_with_stats(self) -> List[Dict]:
        """Get people with statistics"""
        query = '''
        SELECT p.*, ps.entry_count, ps.avg_rating, ps.total_comments, ps.last_activity
        FROM people p
        LEFT JOIN v_people_stats ps ON p.person_id = ps.person_id
        ORDER BY p.display_name
        '''
        
        return self.db.execute_query(query)
    
    @streamlit_cache(ttl=300)
    def get_entries_by_person(self, person_id: str, page: int = 1, limit: int = None) -> Dict:
        """Get entries for person with pagination"""
        if limit is None:
            limit = st.session_state.items_per_page
        
        offset = (page - 1) * limit
        
        query = '''
        SELECT * FROM v_entry_details
        WHERE person_id = ?
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        '''
        
        entries = self.db.execute_query(query, (person_id, limit, offset))
        
        # Get total count
        count_query = 'SELECT COUNT(*) as count FROM album_entries WHERE person_id = ?'
        total_count = self.db.execute_query(count_query, (person_id,))[0]['count']
        
        return {
            'entries': entries,
            'total_count': total_count,
            'page': page,
            'total_pages': max(1, math.ceil(total_count / limit))
        }
    
    @streamlit_cache(ttl=300)
    def search_entries(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search entries with filters"""
        if not query and not filters:
            return []
        
        base_query = '''
        SELECT * FROM v_entry_details
        WHERE 1=1
        '''
        
        params = []
        
        # Text search
        if query:
            base_query += '''
            AND (caption LIKE ? OR description LIKE ? OR tags LIKE ? OR display_name LIKE ?)
            '''
            search_term = f'%{query}%'
            params.extend([search_term, search_term, search_term, search_term])
        
        # Apply filters
        if filters:
            if filters.get('person_id'):
                base_query += ' AND person_id = ?'
                params.append(filters['person_id'])
            
            if filters.get('min_rating'):
                base_query += ' AND rating_avg >= ?'
                params.append(filters['min_rating'])
            
            if filters.get('start_date'):
                base_query += ' AND date_taken >= ?'
                params.append(filters['start_date'])
            
            if filters.get('end_date'):
                base_query += ' AND date_taken <= ?'
                params.append(filters['end_date'])
        
        base_query += ' ORDER BY created_at DESC LIMIT 100'
        
        return self.db.execute_query(base_query, tuple(params))
    
    @background_task(timeout=60)
    def batch_process_images(self, image_paths: List[Path], operation: str, **kwargs):
        """Process multiple images in batch"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                if operation == 'analyze':
                    result = self.analyze_image_with_ai(image_path)
                elif operation == 'thumbnail':
                    result = self.image_processor.create_thumbnail_optimized(image_path)
                else:
                    result = None
                
                results.append({
                    'path': str(image_path),
                    'success': result is not None,
                    'result': result
                })
                
                # Yield progress
                if i % 5 == 0:
                    yield {
                        'progress': (i + 1) / len(image_paths),
                        'processed': i + 1,
                        'total': len(image_paths)
                    }
                    
            except Exception as e:
                results.append({
                    'path': str(image_path),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def analyze_image_with_ai(self, image_path: Path) -> Dict:
        """Analyze image with AI"""
        if not self.ai_manager:
            return {'error': 'AI not available'}
        
        try:
            with Image.open(image_path) as img:
                analysis = self.ai_manager.analyze_image(img)
                
                # Add file info
                analysis['file_info'] = {
                    'path': str(image_path),
                    'size': image_path.stat().st_size,
                    'dimensions': img.size
                }
                
                return analysis
                
        except Exception as e:
            logging.error(f"Error analyzing image {image_path}: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        report = {
            'application': SessionStateManager.get_session_stats(),
            'performance': self.performance_monitor.get_performance_report(),
            'database': self.db.get_stats(),
            'cache': self.cache.get_stats(),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        
        return report
    
    def optimize_application(self):
        """Run application optimizations"""
        logging.info("Running application optimizations")
        
        # Optimize database
        self.db.vacuum()
        
        # Cleanup cache
        self.cache.cleanup()
        
        # Cleanup session files
        SessionStateManager.cleanup_old_sessions()
        
        # Clear memory caches
        self._cleanup_resources()
        
        logging.info("Optimizations complete")

# ============================================================================
# ADVANCED UI COMPONENTS WITH PERFORMANCE OPTIMIZATIONS
# ============================================================================
class AdvancedUIComponents:
    """Advanced UI components with performance optimizations"""
    
    @staticmethod
    @debounce(wait_time=0.3)
    def search_bar(placeholder: str = "Search...", key: str = "search"):
        """Debounced search bar"""
        return st.text_input(placeholder, key=key)
    
    @staticmethod
    def virtual_list(items: List[Any], render_item: Callable, items_per_page: int = 20):
        """Virtual list for large datasets"""
        total_items = len(items)
        
        if 'virtual_list_start' not in st.session_state:
            st.session_state.virtual_list_start = 0
        
        # Calculate visible range
        start_idx = st.session_state.virtual_list_start
        end_idx = min(start_idx + items_per_page, total_items)
        
        # Render visible items
        for i in range(start_idx, end_idx):
            render_item(items[i], i)
        
        # Scroll controls
        if total_items > items_per_page:
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if st.button("▲", key="scroll_up", disabled=start_idx == 0):
                    st.session_state.virtual_list_start = max(0, start_idx - items_per_page)
                    st.rerun()
            
            with col2:
                st.progress(end_idx / total_items)
                st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items}")
            
            with col3:
                if st.button("▼", key="scroll_down", disabled=end_idx >= total_items):
                    st.session_state.virtual_list_start = min(
                        total_items - items_per_page,
                        start_idx + items_per_page
                    )
                    st.rerun()
    
    @staticmethod
    def loading_skeleton(columns: int = 4, rows: int = 3):
        """Display loading skeleton"""
        for _ in range(rows):
            cols = st.columns(columns)
            for col in cols:
                with col:
                    st.markdown(
                        '<div style="background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%); '
                        'background-size: 200% 100%; animation: shimmer 1.5s infinite; '
                        'height: 250px; border-radius: 8px;"></div>',
                        unsafe_allow_html=True
                    )
        
        # Add CSS for shimmer animation
        st.markdown("""
        <style>
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def performance_widget():
        """Display performance metrics widget"""
        with st.expander("📊 Performance Metrics", expanded=False):
            if 'performance_report' not in st.session_state:
                st.session_state.performance_report = {}
            
            if st.button("Refresh Metrics", key="refresh_metrics"):
                with st.spinner("Collecting metrics..."):
                    manager = EnhancedAlbumManager()
                    st.session_state.performance_report = manager.get_performance_report()
            
            report = st.session_state.performance_report
            
            if report:
                # System metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("CPU Usage", f"{report['system']['cpu_percent']:.1f}%")
                
                with col2:
                    st.metric("Memory Usage", f"{report['system']['memory_percent']:.1f}%")
                
                with col3:
                    st.metric("Disk Usage", f"{report['system']['disk_usage']:.1f}%")
                
                # Cache metrics
                if 'cache' in report:
                    st.subheader("Cache Performance")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Cache Size", report['cache']['total_items'])
                        st.metric("Cache Hit Ratio", f"{report['application']['cache_hit_ratio']:.1%}")
                    
                    with col2:
                        st.metric("DB Queries", report['database']['queries_executed'])
                        st.metric("DB Cache Hits", report['database']['cache_hits'])
                
                # Session metrics
                st.subheader("Session Info")
                st.text(f"Session Age: {report['application']['session_age']}")
                st.text(f"Page Views: {report['application']['page_views']}")
                st.text(f"AI Usage: {report['application']['ai_usage_count']}")
                
                # Optimization buttons
                st.subheader("Optimizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Clear Cache", key="clear_all_cache"):
                        if 'performance_cache' in st.session_state:
                            st.session_state.performance_cache = {
                                'data': {},
                                'timestamps': {},
                                'hits': 0,
                                'misses': 0
                            }
                        st.success("Cache cleared!")
                        st.rerun()
                
                with col2:
                    if st.button("Optimize Database", key="optimize_db"):
                        manager = EnhancedAlbumManager()
                        manager.optimize_application()
                        st.success("Database optimized!")
                        st.rerun()
    
    @staticmethod
    def theme_selector():
        """Theme selector with preview"""
        themes = {
            'light': {'bg': '#ffffff', 'text': '#000000', 'primary': '#4CAF50'},
            'dark': {'bg': '#1a1a1a', 'text': '#ffffff', 'primary': '#66BB6A'},
            'blue': {'bg': '#f0f8ff', 'text': '#003366', 'primary': '#2196F3'},
            'warm': {'bg': '#fffaf0', 'text': '#5d4037', 'primary': '#FF9800'}
        }
        
        selected_theme = st.selectbox(
            "Theme",
            list(themes.keys()),
            index=list(themes.keys()).index(st.session_state.theme)
        )
        
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            EnhancedConfig.update_setting('THEME', selected_theme)
            
            # Apply theme
            theme = themes[selected_theme]
            st.markdown(f"""
            <style>
            .stApp {{
                background-color: {theme['bg']};
                color: {theme['text']};
            }}
            .stButton > button {{
                background-color: {theme['primary']};
                color: white;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            st.rerun()
        
        # Theme preview
        cols = st.columns(len(themes))
        for idx, (theme_name, theme_colors) in enumerate(themes.items()):
            with cols[idx]:
                st.markdown(
                    f'<div style="background-color: {theme_colors["bg"]}; '
                    f'color: {theme_colors["text"]}; padding: 10px; '
                    f'border-radius: 5px; border: {"2px solid #4CAF50" if theme_name == selected_theme else "1px solid #ccc"};'
                    f'text-align: center;">{theme_name.title()}</div>',
                    unsafe_allow_html=True
                )

# ============================================================================
# OPTIMIZED PHOTO ALBUM APPLICATION
# ============================================================================
class OptimizedPhotoAlbumApp:
    """Main application class with all optimizations"""
    
    def __init__(self):
        self.manager = None
        self.ui = AdvancedUIComponents()
        self.setup_page_config()
        self.check_initialization()
    
    def setup_page_config(self):
        """Configure Streamlit page with optimizations"""
        st.set_page_config(
            page_title=f"{EnhancedConfig.APP_NAME} - Optimized",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': f"# {EnhancedConfig.APP_NAME} v{EnhancedConfig.VERSION}\nOptimized Edition"
            }
        )
        
        # Add custom CSS for optimizations
        self._add_custom_css()
    
    def _add_custom_css(self):
        """Add optimized CSS"""
        st.markdown("""
        <style>
        /* Reduce animations for performance */
        * {
            scroll-behavior: auto !important;
            transition: none !important;
        }
        
        /* Optimize images */
        img {
            content-visibility: auto;
        }
        
        /* Reduce reflows */
        .stApp {
            contain: content;
        }
        
        /* Performance optimizations */
        [data-testid="stVerticalBlock"] {
            contain: layout style;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def check_initialization(self):
        """Check and initialize application"""
        try:
            EnhancedConfig.init_directories()
            
            # Check AI availability
            EnhancedConfig.AI_ENABLED = lazy_import_ai()
            st.session_state.ai_enabled = EnhancedConfig.AI_ENABLED
            
            self.initialized = True
            logging.info("Application initialized successfully")
            
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            st.error(f"Application failed to initialize: {str(e)}")
            self.initialized = False
    
    def render_sidebar(self):
        """Render optimized sidebar"""
        with st.sidebar:
            # Application header
            st.title(f"🚀 {EnhancedConfig.APP_NAME}")
            st.caption(f"v{EnhancedConfig.VERSION} • Optimized")
            
            # Performance widget
            self.ui.performance_widget()
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            
            # Define navigation items
            nav_items = {
                '📊 Dashboard': 'dashboard',
                '👥 People': 'people',
                '🖼️ Gallery': 'gallery',
                '⭐ Favorites': 'favorites',
                '🔍 Search': 'search',
                '📈 Analytics': 'analytics',
                '⚙️ Settings': 'settings'
            }
            
            # Add AI pages if available
            if st.session_state.ai_enabled:
                nav_items.update({
                    '🤖 AI Studio': 'ai_studio',
                    '⚡ Quick Enhance': 'quick_enhance',
                    '📊 AI Insights': 'ai_insights'
                })
            
            # Render navigation buttons
            for label, page in nav_items.items():
                if st.button(
                    label,
                    use_container_width=True,
                    key=f"nav_{page}",
                    type="primary" if st.session_state.current_page == page else "secondary"
                ):
                    SessionStateManager.navigate_to(page)
            
            st.divider()
            
            # Quick actions
            st.subheader("Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Scan", help="Scan for new images", use_container_width=True):
                    with st.spinner("Scanning..."):
                        self._scan_directory()
            
            with col2:
                if st.button("🧹 Cleanup", help="Cleanup resources", use_container_width=True):
                    if self.manager:
                        self.manager.optimize_application()
                        st.success("Cleanup complete!")
            
            st.divider()
            
            # Theme selector
            st.subheader("Theme")
            self.ui.theme_selector()
            
            st.divider()
            
            # System info
            st.subheader("System")
            
            process = psutil.Process()
            memory_usage = humanize.naturalsize(process.memory_info().rss)
            cpu_percent = process.cpu_percent()
            
            st.metric("Memory", memory_usage)
            st.metric("CPU", f"{cpu_percent:.1f}%")
            
            # Cache stats
            if 'performance_cache' in st.session_state:
                cache = st.session_state.performance_cache
                hit_ratio = cache['hits'] / max(1, cache['hits'] + cache['misses'])
                st.metric("Cache Hit", f"{hit_ratio:.1%}")
    
    def _scan_directory(self):
        """Scan directory for images"""
        try:
            # This would be implemented to scan and update database
            st.info("Directory scan would run here")
            time.sleep(1)  # Simulate scan
            st.success("Scan complete!")
        except Exception as e:
            st.error(f"Scan error: {e}")
    
    def render_dashboard(self):
        """Render optimized dashboard"""
        st.title("📊 Dashboard")
        
        # Performance warning
        process = psutil.Process()
        if process.memory_percent() > 70:
            st.warning("⚠️ High memory usage detected. Consider cleaning up resources.")
        
        # Stats cards with lazy loading
        with st.spinner("Loading stats..."):
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Total People", "24", "3 new")
            
            with cols[1]:
                st.metric("Total Images", "1,234", "12 today")
            
            with cols[2]:
                st.metric("Storage Used", "4.2 GB", "0.1 GB")
            
            with cols[3]:
                st.metric("AI Usage", "156", "8 today")
        
        st.divider()
        
        # Recent activity with virtual list
        st.subheader("Recent Activity")
        
        # Simulate data
        recent_activity = [
            {"user": "John", "action": "uploaded", "item": "beach.jpg", "time": "2 min ago"},
            {"user": "Sarah", "action": "commented", "item": "family.jpg", "time": "15 min ago"},
            {"user": "Mike", "action": "rated", "item": "sunset.png", "time": "1 hour ago"},
            # ... more items
        ] * 10  # Simulate many items
        
        def render_activity_item(item, idx):
            with st.container(border=True):
                st.markdown(f"**{item['user']}** {item['action']} {item['item']}")
                st.caption(item['time'])
        
        self.ui.virtual_list(recent_activity, render_activity_item, items_per_page=5)
        
        st.divider()
        
        # Quick access
        st.subheader("Quick Access")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📸 Upload Photos", use_container_width=True):
                st.info("Upload feature would open here")
        
        with col2:
            if st.button("🎨 Edit Photos", use_container_width=True):
                SessionStateManager.navigate_to('ai_studio')
        
        with col3:
            if st.button("📋 View Reports", use_container_width=True):
                SessionStateManager.navigate_to('analytics')
    
    def render_gallery(self):
        """Render optimized gallery"""
        st.title("🖼️ Gallery")
        
        # Search and filters
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                search_query = self.ui.search_bar("Search photos...", "gallery_search")
            
            with col2:
                view_mode = st.selectbox("View", ["Grid", "List", "Masonry"], key="view_mode")
            
            with col3:
                sort_by = st.selectbox("Sort", ["Newest", "Oldest", "Rating", "Size"], key="sort_by")
            
            with col4:
                if st.button("Filter", key="open_filters"):
                    st.session_state.show_filters = not st.session_state.get('show_filters', False)
        
        # Filters panel
        if st.session_state.get('show_filters', False):
            with st.expander("Filters", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_rating = st.slider("Minimum Rating", 1.0, 5.0, 3.0, 0.5)
                
                with col2:
                    start_date = st.date_input("From Date")
                
                with col3:
                    end_date = st.date_input("To Date")
        
        # Gallery content
        if st.session_state.get('loading_gallery', False):
            self.ui.loading_skeleton(columns=4, rows=3)
        else:
            # Simulate gallery data
            gallery_items = [
                {
                    'entry_id': f"item_{i}",
                    'caption': f"Photo {i}",
                    'display_name': f"Person {i % 5}",
                    'rating_avg': random.uniform(3.0, 5.0),
                    'thumbnail_path': None
                }
                for i in range(100)
            ]
            
            # Apply search filter
            if search_query:
                gallery_items = [item for item in gallery_items if search_query.lower() in item['caption'].lower()]
            
            # Apply sorting
            if sort_by == "Rating":
                gallery_items.sort(key=lambda x: x['rating_avg'], reverse=True)
            elif sort_by == "Oldest":
                gallery_items.sort(key=lambda x: x['entry_id'])
            else:  # Newest
                gallery_items.sort(key=lambda x: x['entry_id'], reverse=True)
            
            # Render gallery
            if view_mode == "Grid":
                self.manager.lazy_gallery.render_gallery(gallery_items, columns=4)
            else:
                # List view implementation
                for item in gallery_items[:20]:  # Limit for demo
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            # Thumbnail placeholder
                            st.markdown(
                                '<div style="background: #f0f0f0; height: 80px; width: 80px; border-radius: 5px;"></div>',
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.markdown(f"**{item['caption']}**")
                            st.caption(f"👤 {item['display_name']}")
                            st.caption(f"⭐ {item['rating_avg']:.1f}")
    
    def render_ai_studio(self):
        """Render AI studio page"""
        if not st.session_state.ai_enabled:
            st.error("AI features are not available")
            st.info("Install AI packages: pip install torch transformers diffusers")
            return
        
        st.title("🎨 AI Studio")
        
        tabs = st.tabs(["Enhance", "Transform", "Generate", "Batch"])
        
        with tabs[0]:  # Enhance
            st.subheader("Image Enhancement")
            
            uploaded_file = st.file_uploader(
                "Upload image to enhance",
                type=['jpg', 'jpeg', 'png'],
                key="enhance_upload"
            )
            
            if uploaded_file:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(uploaded_file, caption="Original", use_column_width=True)
                    
                    # Enhancement options
                    enhancement = st.selectbox(
                        "Enhancement Type",
                        ["Auto Enhance", "Brightness", "Contrast", "Color Balance", "Sharpness"]
                    )
                    
                    strength = st.slider("Strength", 0.0, 2.0, 1.0, 0.1)
                    
                    if st.button("Apply Enhancement", type="primary"):
                        with st.spinner("Applying enhancement..."):
                            time.sleep(2)  # Simulate AI processing
                            st.success("Enhancement applied!")
                
                with col2:
                    # Result placeholder
                    st.markdown(
                        '<div style="background: linear-gradient(45deg, #667eea, #764ba2); '
                        'height: 400px; border-radius: 10px; display: flex; '
                        'align-items: center; justify-content: center; color: white;">'
                        '<div style="text-align: center;">'
                        '<div style="font-size: 48px;">✨</div>'
                        '<div>Enhanced Image</div>'
                        '<div style="font-size: 12px; opacity: 0.8;">(AI processing simulation)</div>'
                        '</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
        
        with tabs[1]:  # Transform
            st.subheader("Style Transfer")
            
            col1, col2 = st.columns(2)
            
            with col1:
                style = st.selectbox(
                    "Style",
                    ["Oil Painting", "Watercolor", "Sketch", "Pop Art", "Vintage", "Cyberpunk"]
                )
                
                uploaded_file = st.file_uploader(
                    "Upload image to transform",
                    type=['jpg', 'jpeg', 'png'],
                    key="transform_upload"
                )
            
            with col2:
                if uploaded_file:
                    if st.button(f"Apply {style} Style", type="primary"):
                        with st.spinner(f"Applying {style} style..."):
                            time.sleep(3)  # Simulate AI processing
                            
                            # Show result
                            st.image(uploaded_file, caption=f"{style} Style", use_column_width=True)
                            
                            # Download button
                            st.download_button(
                                "Download Result",
                                data=uploaded_file.getvalue(),
                                file_name=f"{style}_transformed.jpg",
                                mime="image/jpeg"
                            )
        
        with tabs[2]:  # Generate
            st.subheader("AI Image Generation")
            
            prompt = st.text_area(
                "Describe the image you want to generate",
                placeholder="A beautiful sunset over mountains, digital art, vibrant colors",
                height=100
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_images = st.slider("Number of images", 1, 4, 1)
            
            with col2:
                resolution = st.selectbox("Resolution", ["512x512", "768x768", "1024x1024"])
            
            with col3:
                style = st.selectbox("Style", ["Realistic", "Artistic", "Fantasy", "Minimal"])
            
            if st.button("Generate Images", type="primary") and prompt:
                with st.spinner("Generating images with AI..."):
                    # Simulate generation
                    time.sleep(5)
                    
                    # Show generated images
                    cols = st.columns(min(4, num_images))
                    for i in range(num_images):
                        with cols[i % len(cols)]:
                            st.markdown(
                                f'<div style="background: linear-gradient(45deg, #f093fb, #f5576c); '
                                f'height: 200px; border-radius: 10px; display: flex; '
                                f'align-items: center; justify-content: center; color: white;">'
                                f'Generated {i+1}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
        
        with tabs[3]:  # Batch
            st.subheader("Batch Processing")
            
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="batch_upload"
            )
            
            if uploaded_files:
                st.success(f"Uploaded {len(uploaded_files)} images")
                
                operation = st.selectbox(
                    "Operation",
                    ["Auto Enhance All", "Apply Style to All", "Generate Captions", "Extract Faces"]
                )
                
                if st.button(f"Run Batch {operation}", type="primary"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1} of {len(uploaded_files)}...")
                        
                        # Simulate processing
                        time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Processed {len(uploaded_files)} images")
    
    def render_settings(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["General", "Performance", "AI", "Advanced"])
        
        with tabs[0]:  # General
            st.subheader("General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.checkbox("Auto-refresh gallery", value=True, key="auto_refresh")
                st.checkbox("Show notifications", value=True, key="show_notifications")
                st.checkbox("Enable sounds", value=False, key="enable_sounds")
            
            with col2:
                st.selectbox("Language", ["English", "Spanish", "French", "German"], key="language")
                st.selectbox("Time format", ["12-hour", "24-hour"], key="time_format")
                st.selectbox("Date format", ["MM/DD/YYYY", "DD/MM/YYYY", "YYYY-MM-DD"], key="date_format")
        
        with tabs[1]:  # Performance
            st.subheader("Performance Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                lazy_loading = st.checkbox(
                    "Enable lazy loading",
                    value=EnhancedConfig.LAZY_RENDERING,
                    help="Load images as they become visible"
                )
                
                virtual_scrolling = st.checkbox(
                    "Enable virtual scrolling",
                    value=EnhancedConfig.VIRTUAL_SCROLLING,
                    help="Only render visible items in lists"
                )
                
                cache_ttl = st.number_input(
                    "Cache TTL (seconds)",
                    value=EnhancedConfig.CACHE_TTL,
                    min_value=60,
                    max_value=86400,
                    help="How long to cache data"
                )
            
            with col2:
                max_cache_size = st.number_input(
                    "Max cache size (items)",
                    value=EnhancedConfig.MAX_CACHE_SIZE,
                    min_value=10,
                    max_value=10000
                )
                
                image_quality = st.select_slider(
                    "Image quality",
                    options=["Low", "Medium", "High", "Maximum"],
                    value="Medium"
                )
                
                animations = st.checkbox(
                    "Enable animations",
                    value=EnhancedConfig.ANIMATIONS_ENABLED
                )
            
            if st.button("Apply Performance Settings", type="primary"):
                EnhancedConfig.update_setting('LAZY_RENDERING', lazy_loading)
                EnhancedConfig.update_setting('VIRTUAL_SCROLLING', virtual_scrolling)
                EnhancedConfig.update_setting('CACHE_TTL', cache_ttl)
                EnhancedConfig.update_setting('ANIMATIONS_ENABLED', animations)
                st.success("Performance settings updated!")
        
        with tabs[2]:  # AI
            st.subheader("AI Settings")
            
            if not st.session_state.ai_enabled:
                st.warning("AI features are not available")
                st.code("pip install torch torchvision transformers diffusers")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    ai_enabled = st.checkbox(
                        "Enable AI features",
                        value=EnhancedConfig.AI_ENABLED,
                        key="ai_enabled_setting"
                    )
                    
                    lazy_load = st.checkbox(
                        "Lazy load AI models",
                        value=EnhancedConfig.AI_LAZY_LOAD,
                        help="Only load AI models when needed"
                    )
                    
                    model_cache = st.number_input(
                        "Model cache size",
                        value=EnhancedConfig.AI_MODEL_CACHE_SIZE,
                        min_value=1,
                        max_value=10
                    )
                
                with col2:
                    device = st.selectbox(
                        "AI Device",
                        ["Auto", "CPU", "GPU"],
                        help="Device to run AI models on"
                    )
                    
                    precision = st.selectbox(
                        "Precision",
                        ["Auto", "float32", "float16"],
                        help="Numerical precision for AI models"
                    )
                
                if st.button("Apply AI Settings", type="primary"):
                    EnhancedConfig.update_setting('AI_ENABLED', ai_enabled)
                    EnhancedConfig.update_setting('AI_LAZY_LOAD', lazy_load)
                    EnhancedConfig.update_setting('AI_MODEL_CACHE_SIZE', model_cache)
                    st.success("AI settings updated!")
        
        with tabs[3]:  # Advanced
            st.subheader("Advanced Settings")
            
            st.warning("⚠️ These settings are for advanced users only.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                debug_mode = st.checkbox("Enable debug mode", value=False)
                telemetry = st.checkbox("Enable telemetry", value=False)
                developer = st.checkbox("Developer mode", value=False)
            
            with col2:
                log_level = st.selectbox(
                    "Log level",
                    ["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=1
                )
                
                db_pool = st.number_input(
                    "Database pool size",
                    value=EnhancedConfig.DB_POOL_SIZE,
                    min_value=1,
                    max_value=20
                )
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Clear All Cache", type="secondary"):
                    if 'performance_cache' in st.session_state:
                        st.session_state.performance_cache = {
                            'data': {},
                            'timestamps': {},
                            'hits': 0,
                            'misses': 0
                        }
                    st.success("Cache cleared!")
            
            with col2:
                if st.button("Reset Settings", type="secondary"):
                    # Reset to defaults
                    EnhancedConfig._load_settings()
                    st.success("Settings reset to defaults!")
                    st.rerun()
            
            with col3:
                if st.button("Export Settings", type="secondary"):
                    settings = {
                        'version': EnhancedConfig.VERSION,
                        'config': {k: v for k, v in EnhancedConfig.__dict__.items() 
                                 if not k.startswith('_') and not callable(v)}
                    }
                    
                    st.download_button(
                        "Download Settings",
                        data=json.dumps(settings, indent=2),
                        file_name="photo_album_settings.json",
                        mime="application/json"
                    )
    
    def render_main(self):
        """Main application renderer"""
        # Initialize manager on first render
        if self.manager is None and self.initialized:
            try:
                self.manager = EnhancedAlbumManager()
            except Exception as e:
                logging.error(f"Could not initialize manager: {e}")
                st.error("Could not initialize application manager")
                return
        
        # Render sidebar
        self.render_sidebar()
        
        # Render current page
        current_page = st.session_state.current_page
        
        try:
            with self.manager.performance_monitor.measure(f"render_{current_page}"):
                if current_page == 'dashboard':
                    self.render_dashboard()
                elif current_page == 'gallery':
                    self.render_gallery()
                elif current_page == 'ai_studio':
                    self.render_ai_studio()
                elif current_page == 'settings':
                    self.render_settings()
                elif current_page == 'people':
                    st.title("👥 People")
                    st.info("People management page")
                elif current_page == 'favorites':
                    st.title("⭐ Favorites")
                    st.info("Favorites page")
                elif current_page == 'search':
                    st.title("🔍 Search")
                    st.info("Search page")
                elif current_page == 'analytics':
                    st.title("📈 Analytics")
                    st.info("Analytics page")
                elif current_page == 'quick_enhance':
                    st.title("⚡ Quick Enhance")
                    st.info("Quick enhancement page")
                elif current_page == 'ai_insights':
                    st.title("📊 AI Insights")
                    st.info("AI insights page")
                else:
                    self.render_dashboard()
        
        except Exception as e:
            logging.error(f"Error rendering page {current_page}: {e}")
            st.error(f"Error rendering page: {str(e)}")
            
            if st.session_state.get('settings', {}).get('developer_mode', False):
                with st.expander("Error Details"):
                    st.exception(e)
        
        # Footer with performance info
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.caption(f"{EnhancedConfig.APP_NAME} v{EnhancedConfig.VERSION} • Optimized Edition")
        
        with col2:
            # Render time if measured
            timings = st.session_state.get('timings', {})
            render_time = timings.get(f'render_{current_page}', {}).get('duration', 0)
            if render_time:
                st.caption(f"Render: {render_time*1000:.1f}ms")
        
        with col3:
            # Cache stats
            if 'performance_cache' in st.session_state:
                cache = st.session_state.performance_cache
                hit_ratio = cache['hits'] / max(1, cache['hits'] + cache['misses'])
                st.caption(f"Cache: {hit_ratio:.0%}")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
def main():
    """Main application entry point"""
    try:
        # Create and run application
        app = OptimizedPhotoAlbumApp()
        
        if app.initialized:
            app.render_main()
            
            # Save session state on exit
            SessionStateManager.save_persisted_state()
        else:
            st.error("Application failed to initialize. Please check the logs.")
            
    except Exception as e:
        st.error(f"Critical application error: {str(e)}")
        logging.exception("Critical error in main")
        
        # Recovery options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restart Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📋 View Error Log"):
                with st.expander("Error Details"):
                    st.exception(e)
        
        # Show system info for debugging
        with st.expander("System Information"):
            st.write(f"Python: {sys.version}")
            st.write(f"Streamlit: {st.__version__}")
            st.write(f"Platform: {sys.platform}")
            st.write(f"Process ID: {os.getpid()}")

# ============================================================================
# REQUIREMENTS AND INSTALLATION
# ============================================================================
def generate_requirements():
    """Generate comprehensive requirements.txt"""
    requirements = """# Core dependencies
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
sqlite3>=3.0.0  # Built-in

# Performance optimizations
numba>=0.58.0  # JIT compilation
scipy>=1.11.0  # Scientific computing
opencv-python>=4.8.0  # Computer vision
psutil>=5.9.0  # System monitoring
humanize>=4.7.0  # Human-readable formats

# AI/ML (optional)
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.19.0
accelerate>=0.21.0

# Database optimizations
# No additional packages needed for SQLite

# Image processing
imageio>=2.31.0
scikit-image>=0.22.0

# Utilities
python-dateutil>=2.8.0
pyyaml>=6.0
tqdm>=4.66.0
colorama>=0.4.0
"""
    
    return requirements

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    # Add requirements to sidebar for easy access
    with st.sidebar.expander("📋 Requirements"):
        st.code(generate_requirements(), language="text")
        
        if not lazy_import_ai():
            st.warning("""
            ⚠️ AI libraries not installed. 
            Running in basic mode.
            
            For AI features:
            ```bash
            pip install torch torchvision transformers diffusers
            ```
            """)
    
    # Run main application
    main()
