"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH AI ENHANCEMENTS
Version: 3.0.0 - AI Enhanced with LLM & Diffusion Models
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search,
          AI Image Analysis, AI Style Transfer, AI Editing, Batch Processing
          
PERFORMANCE OPTIMIZATION EXPANSION:
- Session state management with TTL and automatic cleanup
- Database connection pooling and query optimization
- Enhanced caching with memory and disk layers
- Asynchronous operations for non-blocking UI
- Lazy loading of AI models and images
- Optimized image processing with caching
- Virtual scrolling for large datasets
- Memory management and monitoring
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
import threading
import concurrent.futures
import warnings
from collections import OrderedDict
import gc

# ============================================================================
# PERFORMANCE OPTIMIZATION IMPORTS
# ============================================================================
# Optional performance libraries - gracefully degrade if not available
NUMBA_AVAILABLE = False
try:
    from numba import jit, njit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    print("Numba not available, using fallback methods for numerical operations")

JOBLIB_AVAILABLE = False
try:
    from joblib import Memory, Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    print("Joblib not available for disk caching")

# ============================================================================
# AI IMPORTS WITH LAZY LOADING SUPPORT
# ============================================================================
AI_AVAILABLE = False
AI_MODULES = {}

def lazy_load_ai_modules():
    """Dynamically load AI modules only when needed to reduce initial load time"""
    global AI_AVAILABLE, AI_MODULES
    
    if AI_AVAILABLE:  # Already loaded
        return True
        
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        from torchvision import models
        
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
        print("AI modules loaded on-demand")
        return True
    except ImportError as e:
        print(f"AI libraries not available: {e}")
        AI_AVAILABLE = False
        return False

# ============================================================================
# ENHANCED SESSION STATE MANAGEMENT WITH TTL
# ============================================================================

class EnhancedSessionState:
    """
    Enhanced session state management with Time-To-Live (TTL) and automatic cleanup
    Prevents memory bloat by automatically removing stale session state entries
    """
    
    def __init__(self):
        self._init_ttl_tracking()
        
    def _init_ttl_tracking(self):
        """Initialize TTL tracking structures"""
        if '_session_ttl' not in st.session_state:
            st.session_state['_session_ttl'] = {}
        if '_session_last_access' not in st.session_state:
            st.session_state['_session_last_access'] = {}
        if '_session_creation_time' not in st.session_state:
            st.session_state['_session_creation_time'] = {}
            
        # Cleanup expired entries periodically (every 10th access)
        if '_access_count' not in st.session_state:
            st.session_state['_access_count'] = 0
        st.session_state['_access_count'] += 1
        
        if st.session_state['_access_count'] % 10 == 0:
            self._cleanup_expired_entries()
    
    def set_with_ttl(self, key: str, value: Any, ttl_seconds: int = 3600):
        """
        Set a session state value with a Time-To-Live
        
        Args:
            key: Session state key
            value: Value to store
            ttl_seconds: Time to live in seconds (default: 1 hour)
        """
        st.session_state[key] = value
        current_time = time.time()
        st.session_state['_session_ttl'][key] = ttl_seconds
        st.session_state['_session_last_access'][key] = current_time
        st.session_state['_session_creation_time'][key] = current_time
    
    def get_with_ttl(self, key: str, default: Any = None) -> Any:
        """
        Get a session state value, checking TTL
        
        Args:
            key: Session state key
            default: Default value if key doesn't exist or TTL expired
            
        Returns:
            The value if valid, otherwise default
        """
        if key not in st.session_state:
            return default
            
        # Check TTL
        if key in st.session_state['_session_ttl']:
            ttl = st.session_state['_session_ttl'][key]
            last_access = st.session_state['_session_last_access'].get(key, 0)
            
            if time.time() - last_access > ttl:
                # TTL expired, remove the entry
                self.delete_with_ttl(key)
                return default
                
            # Update last access time
            st.session_state['_session_last_access'][key] = time.time()
            
        return st.session_state[key]
    
    def delete_with_ttl(self, key: str):
        """Delete a key and its TTL tracking"""
        if key in st.session_state:
            del st.session_state[key]
        if key in st.session_state['_session_ttl']:
            del st.session_state['_session_ttl'][key]
        if key in st.session_state['_session_last_access']:
            del st.session_state['_session_last_access'][key]
        if key in st.session_state['_session_creation_time']:
            del st.session_state['_session_creation_time'][key]
    
    def _cleanup_expired_entries(self):
        """Clean up all expired session state entries"""
        current_time = time.time()
        keys_to_delete = []
        
        for key, ttl in st.session_state['_session_ttl'].items():
            if key in st.session_state['_session_last_access']:
                last_access = st.session_state['_session_last_access'][key]
                if current_time - last_access > ttl:
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.delete_with_ttl(key)
        
        # Also cleanup very old entries without TTL (older than 24 hours)
        if '_session_creation_time' in st.session_state:
            for key, creation_time in list(st.session_state['_session_creation_time'].items()):
                if key not in st.session_state['_session_ttl'] and current_time - creation_time > 86400:  # 24 hours
                    if key in st.session_state and not key.startswith('_'):
                        del st.session_state[key]
                    del st.session_state['_session_creation_time'][key]
    
    def clear_all_ttl(self):
        """Clear all TTL-managed entries"""
        keys_to_delete = list(st.session_state['_session_ttl'].keys())
        for key in keys_to_delete:
            self.delete_with_ttl(key)
    
    def get_stats(self) -> Dict:
        """Get statistics about session state usage"""
        total_keys = len([k for k in st.session_state.keys() if not k.startswith('_')])
        ttl_keys = len(st.session_state['_session_ttl'])
        expired_count = 0
        
        current_time = time.time()
        for key, ttl in st.session_state['_session_ttl'].items():
            if key in st.session_state['_session_last_access']:
                last_access = st.session_state['_session_last_access'][key]
                if current_time - last_access > ttl:
                    expired_count += 1
        
        return {
            'total_keys': total_keys,
            'ttl_managed_keys': ttl_keys,
            'expired_entries': expired_count,
            'access_count': st.session_state.get('_access_count', 0)
        }

# ============================================================================
# ADVANCED CACHING SYSTEM WITH MULTI-LAYER SUPPORT
# ============================================================================

class MultiLayerCache:
    """
    Multi-layer caching system with memory and optional disk layers
    Implements LRU (Least Recently Used) eviction policy
    """
    
    def __init__(self, max_memory_items: int = 1000, disk_cache_dir: Optional[Path] = None):
        """
        Initialize multi-layer cache
        
        Args:
            max_memory_items: Maximum items to keep in memory cache
            disk_cache_dir: Directory for disk caching (optional)
        """
        self.max_memory_items = max_memory_items
        self.memory_cache = OrderedDict()  # LRU ordered dictionary
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lock = threading.Lock()
        
        # Disk cache setup (optional)
        self.disk_cache_enabled = False
        self.disk_cache_dir = disk_cache_dir
        
        if disk_cache_dir and JOBLIB_AVAILABLE:
            try:
                self.disk_memory = Memory(disk_cache_dir, verbose=0, compress=9)
                self.disk_cache_enabled = True
                print(f"Disk cache enabled at {disk_cache_dir}")
            except Exception as e:
                print(f"Failed to enable disk cache: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self.cache_lock:
            # Try memory cache first
            if key in self.memory_cache:
                # Move to end (most recently used)
                value = self.memory_cache.pop(key)
                self.memory_cache[key] = value
                self.cache_hits += 1
                return value
            
            # Try disk cache if enabled
            if self.disk_cache_enabled and hasattr(self, 'disk_memory'):
                try:
                    # Disk cache keys are hashed
                    disk_key = hashlib.md5(key.encode()).hexdigest()
                    
                    @self.disk_memory.cache
                    def _disk_cached_function():
                        return None
                    
                    # Check if in disk cache
                    cache_path = self.disk_memory._get_cache_path([disk_key])
                    if os.path.exists(cache_path):
                        self.cache_hits += 1
                        return pickle.load(open(cache_path, 'rb'))
                except Exception as e:
                    print(f"Disk cache read error: {e}")
            
            self.cache_misses += 1
            return default
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL in seconds
        """
        with self.cache_lock:
            # Store in memory cache
            if key in self.memory_cache:
                # Remove existing to update order
                del self.memory_cache[key]
            
            self.memory_cache[key] = {
                'value': value,
                'expires': time.time() + ttl_seconds if ttl_seconds else None
            }
            
            # Enforce size limit
            if len(self.memory_cache) > self.max_memory_items:
                # Remove oldest item (first in OrderedDict)
                self.memory_cache.popitem(last=False)
            
            # Optionally store in disk cache for large values
            if self.disk_cache_enabled and isinstance(value, (np.ndarray, pd.DataFrame, Image.Image)):
                try:
                    disk_key = hashlib.md5(key.encode()).hexdigest()
                    cache_path = self.disk_memory._get_cache_path([disk_key])
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"Disk cache write error: {e}")
    
    def delete(self, key: str):
        """Delete key from all cache layers"""
        with self.cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            if self.disk_cache_enabled:
                try:
                    disk_key = hashlib.md5(key.encode()).hexdigest()
                    cache_path = self.disk_memory._get_cache_path([disk_key])
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                except Exception as e:
                    print(f"Disk cache delete error: {e}")
    
    def clear(self):
        """Clear all cache layers"""
        with self.cache_lock:
            self.memory_cache.clear()
            
            if self.disk_cache_enabled:
                try:
                    import shutil
                    if os.path.exists(self.disk_cache_dir):
                        shutil.rmtree(self.disk_cache_dir)
                        os.makedirs(self.disk_cache_dir)
                except Exception as e:
                    print(f"Disk cache clear error: {e}")
            
            self.cache_hits = 0
            self.cache_misses = 0
    
    def cleanup_expired(self):
        """Remove expired entries from memory cache"""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = []
            
            for key, data in self.memory_cache.items():
                if data['expires'] and data['expires'] < current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
            
            # Count expired entries
            current_time = time.time()
            expired_count = 0
            for data in self.memory_cache.values():
                if data['expires'] and data['expires'] < current_time:
                    expired_count += 1
            
            return {
                'memory_items': len(self.memory_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'expired_entries': expired_count,
                'disk_cache_enabled': self.disk_cache_enabled
            }

# ============================================================================
# DATABASE CONNECTION POOLING AND OPTIMIZATION
# ============================================================================

class DatabaseConnectionPool:
    """
    Connection pool for SQLite database to reduce connection overhead
    Improves performance by reusing connections and managing concurrent access
    """
    
    def __init__(self, db_path: Path, pool_size: int = 5, timeout: float = 30.0):
        """
        Initialize connection pool
        
        Args:
            db_path: Path to SQLite database
            pool_size: Maximum number of connections in pool
            timeout: Connection timeout in seconds
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        
        # Thread-safe connection pool
        self._pool = []
        self._in_use = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Performance metrics
        self.total_connections_created = 0
        self.connection_wait_time = 0
        
        # Configure connection defaults
        self._configure_connection_defaults()
    
    def _configure_connection_defaults(self):
        """Configure default connection parameters for optimal performance"""
        self.connection_params = {
            'check_same_thread': False,
            'timeout': self.timeout,
            'isolation_level': None  # Autocommit mode
        }
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection from the pool
        
        Returns:
            SQLite connection object
        """
        start_time = time.time()
        
        with self._condition:
            # Wait for available connection with timeout
            end_time = time.time() + self.timeout
            
            while True:
                # Try to get existing connection from pool
                if self._pool:
                    conn = self._pool.pop()
                    if self._test_connection(conn):
                        self._in_use.add(conn)
                        self.connection_wait_time += time.time() - start_time
                        return conn
                    else:
                        # Connection is stale, close it
                        try:
                            conn.close()
                        except:
                            pass
                
                # Create new connection if pool not full
                if len(self._pool) + len(self._in_use) < self.pool_size:
                    conn = self._create_new_connection()
                    self._in_use.add(conn)
                    self.connection_wait_time += time.time() - start_time
                    return conn
                
                # Wait for connection to become available
                remaining = end_time - time.time()
                if remaining <= 0:
                    raise TimeoutError("Timeout waiting for database connection")
                
                self._condition.wait(remaining)
    
    def _create_new_connection(self) -> sqlite3.Connection:
        """Create a new database connection with performance optimizations"""
        conn = sqlite3.connect(self.db_path, **self.connection_params)
        self.total_connections_created += 1
        
        # Configure for performance
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = -2000")  # 2MB cache
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB memory map
        cursor.execute("PRAGMA busy_timeout = 5000")  # 5 second busy timeout
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn
    
    def _test_connection(self, conn: sqlite3.Connection) -> bool:
        """Test if connection is still valid"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except:
            return False
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        with self._condition:
            if conn in self._in_use:
                self._in_use.remove(conn)
                
                # Reset connection state
                try:
                    conn.rollback()  # Rollback any uncommitted transactions
                    
                    # Clear any remaining statements
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA optimize")
                except:
                    # If connection is broken, close it
                    try:
                        conn.close()
                    except:
                        pass
                    return
                
                # Add back to pool if not full
                if len(self._pool) < self.pool_size:
                    self._pool.append(conn)
                
                # Notify waiting threads
                self._condition.notify()
    
    def close_all(self):
        """Close all connections in pool"""
        with self._condition:
            for conn in self._pool:
                try:
                    conn.close()
                except:
                    pass
            self._pool.clear()
            
            for conn in list(self._in_use):
                try:
                    conn.close()
                except:
                    pass
            self._in_use.clear()
    
    def get_stats(self) -> Dict:
        """Get connection pool statistics"""
        with self._lock:
            return {
                'pool_size': self.pool_size,
                'available_connections': len(self._pool),
                'in_use_connections': len(self._in_use),
                'total_connections_created': self.total_connections_created,
                'avg_wait_time_ms': (self.connection_wait_time / max(self.total_connections_created, 1)) * 1000
            }

# ============================================================================
# ASYNCHRONOUS TASK EXECUTOR FOR NON-BLOCKING OPERATIONS
# ============================================================================

class AsyncTaskExecutor:
    """
    Executes long-running tasks asynchronously to prevent UI blocking
    Manages task queues, prioritization, and result caching
    """
    
    def __init__(self, max_workers: int = 4, max_pending_tasks: int = 100):
        """
        Initialize async task executor
        
        Args:
            max_workers: Maximum concurrent worker threads
            max_pending_tasks: Maximum tasks in queue
        """
        self.max_workers = max_workers
        self.max_pending_tasks = max_pending_tasks
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='album_worker'
        )
        
        # Task tracking
        self.tasks = {}
        self.task_results = {}
        self.task_lock = threading.Lock()
        
        # Performance metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0
    
    def submit_task(self, task_id: str, task_func, *args, priority: int = 0, **kwargs) -> str:
        """
        Submit a task for asynchronous execution
        
        Args:
            task_id: Unique task identifier
            task_func: Function to execute
            *args: Function arguments
            priority: Task priority (higher = more important)
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        with self.task_lock:
            # Check queue capacity
            if len(self.tasks) >= self.max_pending_tasks:
                # Remove lowest priority pending task
                self._remove_lowest_priority_task()
            
            # Store task metadata
            self.tasks[task_id] = {
                'func': task_func,
                'args': args,
                'kwargs': kwargs,
                'priority': priority,
                'status': 'pending',
                'submitted_at': time.time(),
                'future': None
            }
            
            # Submit to thread pool
            future = self.thread_pool.submit(self._execute_task_wrapper, task_id)
            self.tasks[task_id]['future'] = future
            self.tasks[task_id]['status'] = 'running'
            
            return task_id
    
    def _execute_task_wrapper(self, task_id: str):
        """Wrapper for task execution with error handling and timing"""
        start_time = time.time()
        
        try:
            with self.task_lock:
                if task_id not in self.tasks:
                    return None
                
                task_info = self.tasks[task_id]
                func = task_info['func']
                args = task_info['args']
                kwargs = task_info['kwargs']
            
            # Execute the task
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            with self.task_lock:
                self.tasks[task_id]['status'] = 'completed'
                self.tasks[task_id]['completed_at'] = time.time()
                self.tasks[task_id]['execution_time'] = execution_time
                self.task_results[task_id] = result
                self.tasks_completed += 1
                self.total_execution_time += execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self.task_lock:
                self.tasks[task_id]['status'] = 'failed'
                self.tasks[task_id]['error'] = str(e)
                self.tasks[task_id]['execution_time'] = execution_time
                self.tasks_failed += 1
                self.total_execution_time += execution_time
            
            print(f"Task {task_id} failed: {e}")
            return None
    
    def _remove_lowest_priority_task(self):
        """Remove the lowest priority pending task"""
        pending_tasks = [(tid, info) for tid, info in self.tasks.items() 
                        if info['status'] == 'pending']
        
        if pending_tasks:
            # Find task with lowest priority
            lowest_task = min(pending_tasks, key=lambda x: x[1]['priority'])
            task_id = lowest_task[0]
            
            # Remove from tasks
            del self.tasks[task_id]
            print(f"Removed low priority task {task_id} from queue")
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get result of a completed task
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait for completion
            
        Returns:
            Task result or None if not complete/failed
        """
        with self.task_lock:
            if task_id in self.task_results:
                return self.task_results[task_id]
            
            if task_id not in self.tasks:
                return None
            
            task_info = self.tasks[task_id]
        
        # Wait for completion if timeout specified
        if timeout and task_info['future']:
            try:
                result = task_info['future'].result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                return None
            except Exception as e:
                print(f"Task {task_id} error: {e}")
                return None
        
        return None
    
    def cancel_task(self, task_id: str):
        """Cancel a pending or running task"""
        with self.task_lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                
                if task_info['future'] and not task_info['future'].done():
                    task_info['future'].cancel()
                
                task_info['status'] = 'cancelled'
                
                # Clean up
                if task_id in self.task_results:
                    del self.task_results[task_id]
    
    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """Clean up old completed tasks"""
        current_time = time.time()
        tasks_to_remove = []
        
        with self.task_lock:
            for task_id, task_info in self.tasks.items():
                if task_info['status'] in ['completed', 'failed', 'cancelled']:
                    completed_at = task_info.get('completed_at', task_info['submitted_at'])
                    if current_time - completed_at > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                if task_id in self.task_results:
                    del self.task_results[task_id]
                del self.tasks[task_id]
    
    def get_stats(self) -> Dict:
        """Get executor statistics"""
        with self.task_lock:
            status_counts = defaultdict(int)
            for task_info in self.tasks.values():
                status_counts[task_info['status']] += 1
            
            avg_execution_time = (
                self.total_execution_time / max(self.tasks_completed + self.tasks_failed, 1)
            )
            
            return {
                'total_tasks': len(self.tasks),
                'tasks_completed': self.tasks_completed,
                'tasks_failed': self.tasks_failed,
                'status_counts': dict(status_counts),
                'avg_execution_time_seconds': avg_execution_time,
                'max_workers': self.max_workers,
                'pending_slots': self.max_pending_tasks - len(self.tasks)
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor"""
        self.thread_pool.shutdown(wait=wait)

# ============================================================================
# OPTIMIZED CONFIGURATION WITH PERFORMANCE SETTINGS
# ============================================================================

class EnhancedConfig:
    """Enhanced configuration with performance optimization settings"""
    APP_NAME = "MemoryVault Pro AI - Optimized"
    VERSION = "3.0.0-performance"
    
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    AI_MODELS_DIR = BASE_DIR / "ai_models"
    CACHE_DIR = BASE_DIR / "cache"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Files and paths
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album_optimized.db"
    CACHE_FILE = CACHE_DIR / "app_cache.db"
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Performance settings
    CACHE_TTL = 3600  # 1 hour
    SESSION_TTL = 1800  # 30 minutes for session data
    MAX_CACHE_SIZE = 1000
    DB_CONNECTION_POOL_SIZE = 5
    MAX_WORKER_THREADS = 4
    MAX_PENDING_TASKS = 50
    LAZY_LOADING_ENABLED = True
    VIRTUAL_SCROLLING_THRESHOLD = 50  # Use virtual scrolling for >50 items
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # AI Settings (lazy loaded)
    AI_ENABLED = False
    
    # Memory management
    MAX_MEMORY_MB = 500  # Warning threshold
    AUTO_CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes
    
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
            cls.CACHE_DIR,
            cls.TEMP_DIR
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Set permissions (read/write for owner, read for others)
                os.chmod(directory, 0o755)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {str(e)}")
        
        # Initialize AI availability
        cls.AI_ENABLED = lazy_load_ai_modules()
        
        # Initialize performance monitoring
        if 'performance_start_time' not in st.session_state:
            st.session_state.performance_start_time = time.time()
            st.session_state.memory_warnings_shown = 0

# ============================================================================
# OPTIMIZED DATA MODELS WITH PERFORMANCE ENHANCEMENTS
# ============================================================================

@dataclass
class OptimizedImageMetadata(ImageMetadata):
    """Optimized image metadata with caching and efficient serialization"""
    
    @classmethod
    def from_image(cls, image_path: Path) -> 'OptimizedImageMetadata':
        """Create metadata from image file with performance optimizations"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Try to get from cache first
        cache_key = f"metadata_{image_path}_{image_path.stat().st_mtime}"
        if 'metadata_cache' in st.session_state and cache_key in st.session_state.metadata_cache:
            return st.session_state.metadata_cache[cache_key]
            
        try:
            # Open image with minimal loading
            with Image.open(image_path) as img:
                # Get basic info without loading full image
                stats = image_path.stat()
                
                # Extract dimensions efficiently
                width, height = img.size
                
                # Extract EXIF data efficiently (skip if takes too long)
                exif_data = None
                try:
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        if exif:
                            # Only extract important EXIF tags
                            important_tags = {
                                271: 'Make',
                                272: 'Model',
                                274: 'Orientation',
                                306: 'DateTime',
                                36867: 'DateTimeOriginal',
                                33434: 'ExposureTime',
                                33437: 'FNumber',
                                34855: 'ISOSpeedRatings',
                            }
                            exif_data = {}
                            for tag_id, value in exif.items():
                                tag_name = important_tags.get(tag_id)
                                if tag_name:
                                    try:
                                        exif_data[tag_name] = str(value)
                                    except:
                                        pass
                except Exception:
                    exif_data = None
                
                # Create metadata object
                metadata = cls(
                    image_id=str(uuid.uuid4()),
                    filename=image_path.name,
                    filepath=str(image_path.relative_to(EnhancedConfig.DATA_DIR)),
                    file_size=stats.st_size,
                    dimensions=(width, height),
                    format=img.format,
                    created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
                    modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
                    exif_data=exif_data,
                    checksum=cls._calculate_checksum_fast(image_path)
                )
                
                # Cache the metadata
                if 'metadata_cache' not in st.session_state:
                    st.session_state.metadata_cache = {}
                
                # Limit cache size
                if len(st.session_state.metadata_cache) > 100:
                    # Remove oldest entry (simplified FIFO)
                    oldest_key = next(iter(st.session_state.metadata_cache))
                    del st.session_state.metadata_cache[oldest_key]
                
                st.session_state.metadata_cache[cache_key] = metadata
                
                return metadata
                
        except Exception as e:
            print(f"Error creating metadata for {image_path}: {str(e)}")
            raise
    
    @staticmethod
    def _calculate_checksum_fast(image_path: Path) -> str:
        """Fast checksum calculation using file stats"""
        try:
            stats = image_path.stat()
            # Use a combination of file properties for quick checksum
            checksum_data = f"{stats.st_size}-{stats.st_mtime}-{stats.st_ctime}"
            return hashlib.md5(checksum_data.encode()).hexdigest()
        except:
            # Fallback to full file hash if needed
            return ImageMetadata._calculate_checksum(image_path)

# ============================================================================
# OPTIMIZED DATABASE MANAGER
# ============================================================================

class OptimizedDatabaseManager(DatabaseManager):
    """Optimized database manager with connection pooling and query optimization"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or EnhancedConfig.DB_FILE
        self.connection_pool = DatabaseConnectionPool(
            self.db_path, 
            pool_size=EnhancedConfig.DB_CONNECTION_POOL_SIZE
        )
        self.query_cache = MultiLayerCache(max_memory_items=500)
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            yield conn
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                self.connection_pool.return_connection(conn)
    
    def _init_database(self):
        """Initialize database with performance optimizations"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable performance optimizations
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = -2000")
                cursor.execute("PRAGMA temp_store = MEMORY")
                cursor.execute("PRAGMA mmap_size = 268435456")
                cursor.execute("PRAGMA busy_timeout = 5000")
                
                # Create tables with optimized schema
                # (Original table creation code from DatabaseManager._init_database)
                # ... [Include all original table creation code here]
                
                # Create additional indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_checksum ON images(checksum)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_image ON album_entries(image_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_user ON comments(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_user_entry ON ratings(user_id, entry_id)')
                
                conn.commit()
                st.success("✅ Database initialized with performance optimizations!")
                
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise
    
    def execute_cached_query(self, query: str, params: tuple = (), ttl: int = 300) -> List[Dict]:
        """
        Execute a query with result caching
        
        Args:
            query: SQL query
            params: Query parameters
            ttl: Cache time-to-live in seconds
            
        Returns:
            Query results
        """
        cache_key = f"query_{hashlib.md5((query + str(params)).encode()).hexdigest()}"
        
        # Try cache first
        cached_result = self.query_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute query
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Cache the results
                self.query_cache.set(cache_key, results, ttl)
                
                return results
        except Exception as e:
            st.error(f"Query execution error: {str(e)}")
            return []
    
    def add_image(self, metadata: ImageMetadata, thumbnail_path: str = None):
        """Optimized image addition with bulk operation support"""
        # Use original implementation but with connection pooling
        super().add_image(metadata, thumbnail_path)
        
        # Invalidate relevant caches
        self.query_cache.delete(f"query_{hashlib.md5(b'SELECT * FROM images').hexdigest()}")
    
    def search_entries_optimized(self, query: str, person_id: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Optimized search with pagination and full-text search support
        """
        search_pattern = f'%{query}%'
        cache_key = f"search_{query}_{person_id}_{limit}_{offset}"
        
        # Try cache first
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if person_id:
                    # Use prepared statement for security and performance
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.person_id = ? AND
                    (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (person_id, search_pattern, search_pattern, search_pattern, limit, offset))
                else:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (search_pattern, search_pattern, search_pattern, limit, offset))
                
                results = [dict(row) for row in cursor.fetchall()]
                
                # Cache results
                self.query_cache.set(cache_key, results, ttl=60)  # 1 minute cache
                
                return results
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database performance statistics"""
        db_stats = {}
        pool_stats = self.connection_pool.get_stats()
        cache_stats = self.query_cache.get_stats()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    db_stats[f"{table_name}_count"] = count
                
                # Get database size
                if os.path.exists(self.db_path):
                    db_stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
                
        except Exception as e:
            db_stats['error'] = str(e)
        
        return {
            'database': db_stats,
            'connection_pool': pool_stats,
            'query_cache': cache_stats
        }

# ============================================================================
# OPTIMIZED IMAGE PROCESSOR WITH CACHING AND PERFORMANCE
# ============================================================================

class OptimizedImageProcessor(ImageProcessor):
    """Optimized image processor with caching, parallel processing, and memory management"""
    
    # Class-level caches (shared across instances)
    _thumbnail_cache = MultiLayerCache(max_memory_items=200, disk_cache_dir=EnhancedConfig.CACHE_DIR / "thumbnails")
    _image_data_cache = MultiLayerCache(max_memory_items=100)
    
    # Thread pool for parallel processing
    _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    @staticmethod
    def create_thumbnail_optimized(image_path: Path, thumbnail_dir: Path = None, 
                                   size: Tuple[int, int] = None, quality: int = 85) -> Optional[Path]:
        """
        Create thumbnail with intelligent caching and optimization
        
        Args:
            image_path: Path to source image
            thumbnail_dir: Directory for thumbnails
            size: Thumbnail size (width, height)
            quality: JPEG quality (1-100)
            
        Returns:
            Path to thumbnail or None if failed
        """
        if not image_path.exists():
            return None
            
        thumbnail_dir = thumbnail_dir or EnhancedConfig.THUMBNAIL_DIR
        size = size or EnhancedConfig.THUMBNAIL_SIZE
        
        # Create cache key based on file properties
        stats = image_path.stat()
        cache_key = f"thumb_{image_path}_{stats.st_mtime}_{stats.st_size}_{size[0]}x{size[1]}_{quality}"
        
        # Check cache first
        cached_path = OptimizedImageProcessor._thumbnail_cache.get(cache_key)
        if cached_path and isinstance(cached_path, Path) and cached_path.exists():
            return cached_path
        
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb_{size[0]}x{size[1]}.jpg"
        
        try:
            # Use optimized image loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                with Image.open(image_path) as img:
                    # Apply EXIF orientation
                    img = ImageOps.exif_transpose(img)
                    
                    # Convert to RGB if necessary (optimized)
                    if img.mode not in ('RGB', 'L'):
                        if img.mode == 'RGBA':
                            # Efficient RGBA to RGB conversion
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[-1])
                            img = background
                        else:
                            img = img.convert('RGB')
                    
                    # Create thumbnail with aspect ratio preservation
                    img.thumbnail(size, Image.Resampling.LANCZOS)
                    
                    # Optimized save with progressive encoding for web
                    save_kwargs = {
                        'format': 'JPEG',
                        'quality': quality,
                        'optimize': True,
                        'progressive': True,  # Progressive JPEG loads faster
                        'subsampling': -1     # Keep all color information
                    }
                    
                    img.save(thumbnail_path, **save_kwargs)
            
            # Cache the thumbnail path
            OptimizedImageProcessor._thumbnail_cache.set(cache_key, thumbnail_path, ttl=86400)  # 24 hours
            
            return thumbnail_path
            
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def create_thumbnails_batch(image_paths: List[Path], thumbnail_dir: Path = None) -> Dict[Path, Optional[Path]]:
        """
        Create thumbnails for multiple images in parallel
        
        Args:
            image_paths: List of image paths
            thumbnail_dir: Directory for thumbnails
            
        Returns:
            Dictionary mapping original paths to thumbnail paths
        """
        results = {}
        
        # Submit thumbnail creation tasks in parallel
        futures = {}
        for img_path in image_paths:
            future = OptimizedImageProcessor._thread_pool.submit(
                OptimizedImageProcessor.create_thumbnail_optimized,
                img_path,
                thumbnail_dir
            )
            futures[future] = img_path
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            img_path = futures[future]
            try:
                thumb_path = future.result()
                results[img_path] = thumb_path
            except Exception as e:
                print(f"Failed to create thumbnail for {img_path}: {e}")
                results[img_path] = None
        
        return results
    
    @staticmethod
    def get_image_data_url_optimized(image_path: Path, max_size: Optional[Tuple[int, int]] = None) -> str:
        """
        Get image as data URL with caching and optimization
        
        Args:
            image_path: Path to image
            max_size: Optional maximum size for resizing
            
        Returns:
            Data URL string
        """
        if not image_path.exists():
            return ""
        
        # Create cache key
        cache_key_parts = [str(image_path), image_path.stat().st_mtime]
        if max_size:
            cache_key_parts.extend([str(max_size[0]), str(max_size[1])])
        cache_key = hashlib.md5('_'.join(map(str, cache_key_parts)).encode()).hexdigest()
        
        # Check cache
        cached_url = OptimizedImageProcessor._image_data_cache.get(cache_key)
        if cached_url:
            return cached_url
        
        try:
            with Image.open(image_path) as img:
                # Resize if needed
                if max_size and (img.width > max_size[0] or img.height > max_size[1]):
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to bytes
                img_bytes = io.BytesIO()
                
                # Determine format and optimize save
                if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                    img.save(img_bytes, format='JPEG', quality=85, optimize=True)
                    mime_type = 'image/jpeg'
                elif image_path.suffix.lower() == '.png':
                    img.save(img_bytes, format='PNG', optimize=True)
                    mime_type = 'image/png'
                elif image_path.suffix.lower() == '.webp':
                    img.save(img_bytes, format='WEBP', quality=85)
                    mime_type = 'image/webp'
                else:
                    # Default to JPEG
                    img = img.convert('RGB') if img.mode != 'RGB' else img
                    img.save(img_bytes, format='JPEG', quality=85, optimize=True)
                    mime_type = 'image/jpeg'
                
                img_bytes.seek(0)
                encoded = base64.b64encode(img_bytes.read()).decode('utf-8')
                data_url = f"data:{mime_type};base64,{encoded}"
                
                # Cache the data URL
                OptimizedImageProcessor._image_data_cache.set(cache_key, data_url, ttl=3600)  # 1 hour
                
                return data_url
                
        except Exception as e:
            print(f"Error creating data URL for {image_path}: {str(e)}")
            return ""
    
    @staticmethod
    def optimize_image_memory(image: Image.Image) -> Image.Image:
        """
        Optimize image for memory usage
        
        Args:
            image: PIL Image object
            
        Returns:
            Optimized image
        """
        # Convert to efficient format
        if image.mode == 'RGBA':
            # Convert RGBA to RGB with white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        elif image.mode not in ['RGB', 'L', 'P']:
            # Convert to RGB
            return image.convert('RGB')
        else:
            return image
    
    @staticmethod
    def clear_caches():
        """Clear all image processor caches"""
        OptimizedImageProcessor._thumbnail_cache.clear()
        OptimizedImageProcessor._image_data_cache.clear()
        gc.collect()

# ============================================================================
# OPTIMIZED ALBUM MANAGER WITH PERFORMANCE ENHANCEMENTS
# ============================================================================

class OptimizedAlbumManager(AlbumManager):
    """
    Optimized album manager with performance enhancements:
    - Asynchronous operations
    - Intelligent caching
    - Memory management
    - Connection pooling
    """
    
    def __init__(self):
        # Initialize enhanced components
        self.db = OptimizedDatabaseManager()
        self.cache = MultiLayerCache(
            max_memory_items=EnhancedConfig.MAX_CACHE_SIZE,
            disk_cache_dir=EnhancedConfig.CACHE_DIR
        )
        self.image_processor = OptimizedImageProcessor()
        self.session_manager = EnhancedSessionState()
        self.async_executor = AsyncTaskExecutor(
            max_workers=EnhancedConfig.MAX_WORKER_THREADS,
            max_pending_tasks=EnhancedConfig.MAX_PENDING_TASKS
        )
        self._init_optimized_session_state()
        
        # Performance monitoring
        self._init_performance_monitoring()
    
    def _init_optimized_session_state(self):
        """Initialize session state with performance optimizations"""
        # Core session state with TTL
        defaults = {
            'current_page': ('dashboard', 1800),
            'selected_person': (None, 900),
            'selected_image': (None, 900),
            'search_query': ('', 600),
            'view_mode': ('grid', 3600),
            'sort_by': ('date', 3600),
            'sort_order': ('desc', 3600),
            'selected_tags': ([], 1800),
            'user_id': (str(uuid.uuid4()), 86400),
            'username': ('Guest', 86400),
            'user_role': (UserRoles.VIEWER.value, 86400),
            'favorites': (set(), 3600),
            'recently_viewed': ([], 1800),
            'toc_page': (1, 300),
            'gallery_page': (1, 300),
            'show_directory_info': (True, 3600),
            'performance_mode': (True, 86400)
        }
        
        for key, (default_value, ttl) in defaults.items():
            if key not in st.session_state:
                self.session_manager.set_with_ttl(key, default_value, ttl)
        
        # Initialize caches in session state
        if 'image_cache' not in st.session_state:
            st.session_state.image_cache = {}
        if 'thumbnail_cache' not in st.session_state:
            st.session_state.thumbnail_cache = {}
        if 'metadata_cache' not in st.session_state:
            st.session_state.metadata_cache = {}
    
    def _init_performance_monitoring(self):
        """Initialize performance monitoring"""
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'db_queries': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'image_processing_time': 0,
                'start_time': time.time()
            }
    
    def scan_directory_async(self, data_dir: Path = None) -> str:
        """
        Scan directory asynchronously
        
        Args:
            data_dir: Directory to scan
            
        Returns:
            Task ID for tracking
        """
        data_dir = data_dir or EnhancedConfig.DATA_DIR
        task_id = f"scan_{int(time.time())}"
        
        def scan_task():
            # Run the original scan_directory method in background
            return super().scan_directory(data_dir)
        
        # Submit to async executor
        self.async_executor.submit_task(task_id, scan_task, priority=1)
        
        # Store task info in session state
        self.session_manager.set_with_ttl(f"task_{task_id}", {
            'type': 'directory_scan',
            'status': 'running',
            'started_at': time.time()
        }, ttl=3600)
        
        return task_id
    
    def get_scan_status(self, task_id: str) -> Dict:
        """
        Get status of a directory scan task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status dictionary
        """
        task_info = self.session_manager.get_with_ttl(f"task_{task_id}")
        if not task_info:
            return {'status': 'unknown'}
        
        # Check with async executor
        result = self.async_executor.get_task_result(task_id, timeout=0.1)
        if result is not None:
            task_info['status'] = 'completed'
            task_info['result'] = result
            task_info['completed_at'] = time.time()
        
        return task_info
    
    @PerformanceOptimizer.st_cache_data(ttl=300)  # Cache for 5 minutes
    def get_person_stats(self, person_id: str) -> Dict:
        """Get statistics for a person with enhanced caching"""
        return super().get_person_stats(person_id)
    
    def get_entries_by_person_optimized(self, person_id: str, page: int = 1, 
                                        search_query: str = None, 
                                        batch_size: int = None) -> Dict:
        """
        Optimized version of get_entries_by_person with intelligent caching
        
        Args:
            person_id: Person identifier
            page: Page number
            search_query: Optional search query
            batch_size: Number of items per page
            
        Returns:
            Dictionary with entries and pagination info
        """
        batch_size = batch_size or EnhancedConfig.ITEMS_PER_PAGE
        
        # Create cache key
        cache_key_parts = [f"entries_person_{person_id}", f"page_{page}", f"batch_{batch_size}"]
        if search_query:
            cache_key_parts.append(f"search_{hashlib.md5(search_query.encode()).hexdigest()[:8]}")
        cache_key = '_'.join(cache_key_parts)
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            st.session_state.performance_metrics['cache_hits'] += 1
            return cached_result
        
        st.session_state.performance_metrics['cache_misses'] += 1
        
        # Calculate offset
        offset = (page - 1) * batch_size
        
        try:
            with self.db.get_connection() as conn:
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
                    ''', (person_id, search_pattern, search_pattern, search_pattern, batch_size, offset))
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
                    ''', (person_id, batch_size, offset))
                
                entries = [dict(row) for row in cursor.fetchall()]
                
                # Get total count (cached separately)
                count_cache_key = f"count_person_{person_id}"
                if search_query:
                    count_cache_key += f"_search_{hashlib.md5(search_query.encode()).hexdigest()[:8]}"
                
                total_count = self.cache.get(count_cache_key)
                if total_count is None:
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
                    self.cache.set(count_cache_key, total_count, ttl=60)  # Cache count for 1 minute
                
                total_pages = max(1, math.ceil(total_count / batch_size))
                
                result = {
                    'entries': entries,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'current_page': page,
                    'batch_size': batch_size
                }
                
                # Cache the result
                self.cache.set(cache_key, result, ttl=120)  # Cache for 2 minutes
                
                return result
                
        except Exception as e:
            st.error(f"Error getting entries: {str(e)}")
            return {
                'entries': [],
                'total_count': 0,
                'total_pages': 1,
                'current_page': page,
                'batch_size': batch_size
            }
    
    def get_entry_with_details_optimized(self, entry_id: str, preload_images: bool = True) -> Optional[Dict]:
        """
        Get entry details with optimization options
        
        Args:
            entry_id: Entry identifier
            preload_images: Whether to preload image data URLs
            
        Returns:
            Entry details dictionary
        """
        cache_key = f"entry_details_{entry_id}"
        
        # Try cache first
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            return cached_entry
        
        # Get from database
        entry = super().get_entry_with_details(entry_id)
        if not entry:
            return None
        
        # Optimize image loading
        if preload_images and entry.get('filepath'):
            image_path = EnhancedConfig.DATA_DIR / entry['filepath']
            if image_path.exists():
                # Use optimized data URL generation
                entry['image_data_url'] = OptimizedImageProcessor.get_image_data_url_optimized(
                    image_path, 
                    max_size=EnhancedConfig.PREVIEW_SIZE
                )
        
        # Cache the entry
        self.cache.set(cache_key, entry, ttl=300)  # Cache for 5 minutes
        
        return entry
    
    def batch_process_images(self, image_paths: List[Path], operation: str, **kwargs) -> str:
        """
        Batch process multiple images asynchronously
        
        Args:
            image_paths: List of image paths
            operation: Operation to perform ('thumbnail', 'analyze', 'enhance')
            **kwargs: Operation-specific parameters
            
        Returns:
            Task ID for tracking
        """
        task_id = f"batch_{operation}_{int(time.time())}"
        
        def batch_task():
            results = []
            total = len(image_paths)
            
            for i, img_path in enumerate(image_paths):
                try:
                    if operation == 'thumbnail':
                        result = OptimizedImageProcessor.create_thumbnail_optimized(img_path)
                    elif operation == 'analyze' and hasattr(self, 'analyze_image_with_ai'):
                        result = self.analyze_image_with_ai(img_path)
                    elif operation == 'enhance' and hasattr(self, 'apply_ai_modification'):
                        result = self.apply_ai_modification(img_path, 'auto_enhance')
                    else:
                        result = None
                    
                    results.append((img_path, result))
                    
                    # Update progress in session state
                    progress_key = f"batch_progress_{task_id}"
                    self.session_manager.set_with_ttl(progress_key, {
                        'current': i + 1,
                        'total': total,
                        'percent': ((i + 1) / total) * 100
                    }, ttl=300)
                    
                except Exception as e:
                    results.append((img_path, {'error': str(e)}))
            
            return results
        
        # Submit batch task
        self.async_executor.submit_task(task_id, batch_task, priority=2)
        
        # Store task info
        self.session_manager.set_with_ttl(f"task_{task_id}", {
            'type': f'batch_{operation}',
            'status': 'running',
            'total_images': len(image_paths),
            'started_at': time.time()
        }, ttl=3600)
        
        return task_id
    
    def get_batch_progress(self, task_id: str) -> Dict:
        """
        Get progress of a batch processing task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Progress dictionary
        """
        # Get progress from session state
        progress = self.session_manager.get_with_ttl(f"batch_progress_{task_id}")
        if not progress:
            progress = {'current': 0, 'total': 0, 'percent': 0}
        
        # Get task status
        task_info = self.session_manager.get_with_ttl(f"task_{task_id}")
        if task_info:
            progress.update({
                'status': task_info.get('status', 'unknown'),
                'type': task_info.get('type', 'unknown')
            })
        
        # Check if task is complete
        result = self.async_executor.get_task_result(task_id, timeout=0.1)
        if result is not None:
            progress['status'] = 'completed'
            progress['result'] = result
        
        return progress
    
    def cleanup_resources(self):
        """Cleanup resources and optimize memory"""
        # Cleanup expired session state
        self.session_manager._cleanup_expired_entries()
        
        # Cleanup old async tasks
        self.async_executor.cleanup_old_tasks()
        
        # Clear expired cache entries
        self.cache.cleanup_expired()
        
        # Force garbage collection
        gc.collect()
        
        # Clear image processor caches if memory is high
        try:
            import psutil
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            if memory_mb > EnhancedConfig.MAX_MEMORY_MB:
                OptimizedImageProcessor.clear_caches()
                print(f"Cleared image caches due to high memory: {memory_mb:.1f}MB")
        except:
            pass
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        db_stats = self.db.get_stats()
        cache_stats = self.cache.get_stats()
        async_stats = self.async_executor.get_stats()
        session_stats = self.session_manager.get_stats()
        
        # Calculate uptime
        uptime = time.time() - st.session_state.performance_metrics['start_time']
        
        # Memory usage
        memory_mb = 0
        try:
            import psutil
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except:
            pass
        
        return {
            'uptime_seconds': uptime,
            'uptime_human': str(datetime.timedelta(seconds=int(uptime))),
            'memory_usage_mb': memory_mb,
            'database': db_stats,
            'cache': cache_stats,
            'async_tasks': async_stats,
            'session_state': session_stats,
            'performance_metrics': st.session_state.performance_metrics
        }

# ============================================================================
# OPTIMIZED UI COMPONENTS WITH VIRTUAL SCROLLING AND LAZY LOADING
# ============================================================================

class OptimizedUIComponents(UIComponents):
    """Optimized UI components with virtual scrolling and lazy loading"""
    
    @staticmethod
    def render_virtual_scroll_gallery(entries: List[Dict], items_per_row: int = 4, 
                                      batch_size: int = 20, height: int = 800):
        """
        Render gallery with virtual scrolling for performance
        
        Args:
            entries: List of entry dictionaries
            items_per_row: Number of items per row
            batch_size: Number of items to render at once
            height: Container height in pixels
        """
        if not entries:
            st.info("No images to display")
            return
        
        # Create scroll position tracking
        scroll_key = f"scroll_{hash(str(entries[:10]))}"
        if scroll_key not in st.session_state:
            st.session_state[scroll_key] = {
                'offset': 0,
                'total': len(entries)
            }
        
        scroll_state = st.session_state[scroll_key]
        offset = scroll_state['offset']
        
        # Calculate visible range
        end_idx = min(offset + batch_size, len(entries))
        visible_entries = entries[offset:end_idx]
        
        # Create container with fixed height for scrolling
        container = st.container(height=height)
        
        with container:
            # Render visible items
            cols = st.columns(items_per_row)
            for idx, entry in enumerate(visible_entries):
                col_idx = idx % items_per_row
                with cols[col_idx]:
                    OptimizedUIComponents._render_gallery_item_optimized(entry)
            
            # Show loading indicator if more items exist
            if end_idx < len(entries):
                st.progress((end_idx / len(entries)) * 100, text="Loading more images...")
            
            # Navigation controls
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if offset > 0:
                    if st.button("⬆ Load Previous", key=f"{scroll_key}_prev"):
                        scroll_state['offset'] = max(0, offset - batch_size)
                        st.rerun()
            
            with col2:
                st.caption(f"Showing {offset + 1}-{end_idx} of {len(entries)} images")
            
            with col3:
                if end_idx < len(entries):
                    if st.button("⬇ Load More", key=f"{scroll_key}_next"):
                        scroll_state['offset'] = end_idx
                        st.rerun()
    
    @staticmethod
    def _render_gallery_item_optimized(entry: Dict):
        """Optimized gallery item rendering with lazy image loading"""
        with st.container(border=True):
            # Use lazy image loading
            if entry.get('thumbnail_data_url'):
                st.image(
                    entry['thumbnail_data_url'],
                    use_column_width=True,
                    output_format="JPEG",
                    clamp=True
                )
            elif entry.get('thumbnail_path'):
                thumbnail_path = Path(entry['thumbnail_path'])
                if thumbnail_path.exists():
                    # Generate data URL on demand
                    data_url = OptimizedImageProcessor.get_image_data_url_optimized(
                        thumbnail_path,
                        max_size=EnhancedConfig.THUMBNAIL_SIZE
                    )
                    st.image(data_url, use_column_width=True)
                else:
                    st.markdown("🖼️")
            else:
                st.markdown("🖼️")
            
            # Caption with truncation
            caption = entry.get('caption', 'Untitled')
            if len(caption) > 30:
                caption = caption[:27] + "..."
            st.markdown(f"**{caption}**")
            
            # Person name
            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
            
            # Rating (if available)
            if entry.get('avg_rating'):
                st.markdown(UIComponents.rating_stars(entry['avg_rating'], size=15))
            
            # Actions in compact layout
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️", key=f"view_{entry['entry_id']}", 
                           help="View details", use_container_width=True):
                    st.session_state['selected_image'] = entry['entry_id']
                    st.session_state['current_page'] = 'image_detail'
                    st.rerun()
            with col2:
                is_favorited = entry['entry_id'] in st.session_state.get('favorites', set())
                if is_favorited:
                    if st.button("⭐", key=f"unfav_{entry['entry_id']}", 
                               help="Remove favorite", use_container_width=True):
                        # This would call manager.remove_from_favorites in full implementation
                        pass
                else:
                    if st.button("☆", key=f"fav_{entry['entry_id']}", 
                               help="Add favorite", use_container_width=True):
                        # This would call manager.add_to_favorites in full implementation
                        pass
    
    @staticmethod
    def render_performance_dashboard(manager: OptimizedAlbumManager):
        """Render performance monitoring dashboard"""
        st.title("📊 Performance Dashboard")
        
        # Get performance stats
        with st.spinner("Collecting performance data..."):
            stats = manager.get_performance_stats()
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Uptime", stats['uptime_human'])
        
        with col2:
            st.metric("Memory Usage", f"{stats['memory_usage_mb']:.1f} MB")
        
        with col3:
            cache_hit_rate = stats['cache'].get('hit_rate', '0%')
            st.metric("Cache Hit Rate", cache_hit_rate)
        
        with col4:
            db_size = stats['database'].get('database_size_mb', 0)
            st.metric("Database Size", f"{db_size:.1f} MB")
        
        # Tabs for detailed stats
        tab1, tab2, tab3, tab4 = st.tabs(["Database", "Cache", "Async Tasks", "Session"])
        
        with tab1:
            st.subheader("Database Statistics")
            db_stats = stats['database']
            
            if 'database' in db_stats:
                st.json(db_stats['database'])
            
            if 'connection_pool' in db_stats:
                st.markdown("**Connection Pool:**")
                pool_stats = db_stats['connection_pool']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Available", pool_stats.get('available_connections', 0))
                with col2:
                    st.metric("In Use", pool_stats.get('in_use_connections', 0))
                with col3:
                    st.metric("Total Created", pool_stats.get('total_connections_created', 0))
        
        with tab2:
            st.subheader("Cache Statistics")
            cache_stats = stats['cache']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Memory Items", cache_stats.get('memory_items', 0))
            with col2:
                st.metric("Cache Hits", cache_stats.get('cache_hits', 0))
            with col3:
                st.metric("Cache Misses", cache_stats.get('cache_misses', 0))
            
            st.metric("Hit Rate", cache_stats.get('hit_rate', '0%'))
            
            if cache_stats.get('disk_cache_enabled'):
                st.success("✅ Disk cache enabled")
            else:
                st.info("ℹ️ Disk cache not available")
        
        with tab3:
            st.subheader("Async Task Statistics")
            task_stats = stats['async_tasks']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tasks", task_stats.get('total_tasks', 0))
            with col2:
                st.metric("Completed", task_stats.get('tasks_completed', 0))
            with col3:
                st.metric("Failed", task_stats.get('tasks_failed', 0))
            
            st.metric("Avg Execution Time", 
                     f"{task_stats.get('avg_execution_time_seconds', 0):.2f}s")
            
            if 'status_counts' in task_stats:
                st.markdown("**Task Status:**")
                for status, count in task_stats['status_counts'].items():
                    st.text(f"{status}: {count}")
        
        with tab4:
            st.subheader("Session State Statistics")
            session_stats = stats['session_state']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Keys", session_stats.get('total_keys', 0))
            with col2:
                st.metric("TTL Managed", session_stats.get('ttl_managed_keys', 0))
            with col3:
                st.metric("Expired Entries", session_stats.get('expired_entries', 0))
            
            st.metric("Total Accesses", session_stats.get('access_count', 0))
        
        # Performance actions
        st.divider()
        st.subheader("Performance Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Clear All Caches", use_container_width=True):
                manager.cache.clear()
                OptimizedImageProcessor.clear_caches()
                st.success("Caches cleared!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Cleanup Resources", use_container_width=True):
                manager.cleanup_resources()
                st.success("Resources cleaned up!")
                st.rerun()
        
        with col3:
            if st.button("📊 Refresh Stats", use_container_width=True):
                st.rerun()

# ============================================================================
# ENHANCED PHOTO ALBUM APP WITH PERFORMANCE OPTIMIZATIONS
# ============================================================================

class EnhancedPhotoAlbumApp(PhotoAlbumApp):
    """Enhanced photo album app with all performance optimizations"""
    
    def __init__(self):
        # Initialize configuration
        EnhancedConfig.init_directories()
        
        # Initialize optimized manager
        self.manager = OptimizedAlbumManager()
        self.ui_components = OptimizedUIComponents()
        self.performance_mode = True
        
        # Setup page with optimizations
        self.setup_optimized_page_config()
        self.check_initialization()
    
    def setup_optimized_page_config(self):
        """Configure Streamlit page with performance optimizations"""
        st.set_page_config(
            page_title=f"{EnhancedConfig.APP_NAME} (Optimized)",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-album',
                'Report a bug': 'https://github.com/yourusername/photo-album/issues',
                'About': f"# {EnhancedConfig.APP_NAME} v{EnhancedConfig.VERSION}\nPerformance-optimized version"
            }
        )
        
        # Add performance-optimized CSS
        st.markdown("""
        <style>
        /* Performance optimizations */
        .stImage > img, .stImage > div > img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        
        /* Reduce animations */
        * {
            animation-duration: 0.1s !important;
            transition-duration: 0.1s !important;
        }
        
        /* Optimize scrolling */
        .element-container {
            contain: content;
        }
        
        /* Compact buttons */
        .stButton > button {
            padding: 0.25rem 0.5rem;
            font-size: 0.875rem;
        }
        
        /* Performance warning */
        .performance-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.25rem;
            padding: 0.75rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_optimized_sidebar(self):
        """Render optimized sidebar with performance controls"""
        with st.sidebar:
            st.title(f"⚡ {EnhancedConfig.APP_NAME}")
            st.caption(f"v{EnhancedConfig.VERSION} - Performance Mode")
            
            # Performance toggle
            st.divider()
            performance_mode = st.checkbox(
                "Enable Performance Mode", 
                value=self.performance_mode,
                help="Toggle performance optimizations"
            )
            
            if performance_mode != self.performance_mode:
                self.performance_mode = performance_mode
                st.rerun()
            
            # User info (compact)
            st.divider()
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**{st.session_state['username']}**")
                st.caption(st.session_state['user_role'].title())
            
            # Navigation (optimized)
            st.divider()
            st.subheader("Navigation")
            
            nav_items = [
                ("🏠 Dashboard", "dashboard"),
                ("👥 People", "people"),
                ("📁 Gallery", "gallery"),
                ("⭐ Favorites", "favorites"),
                ("🔍 Search", "search"),
                ("📊 Statistics", "statistics"),
                ("⚙️ Settings", "settings"),
                ("📤 Import/Export", "import_export"),
                ("📊 Performance", "performance")  # New performance page
            ]
            
            # Add AI navigation if available
            if EnhancedConfig.AI_ENABLED:
                nav_items.extend([
                    ("🤖 AI Analysis", "ai_analysis"),
                    ("🎨 AI Studio", "ai_studio"),
                    ("⚡ Quick Enhance", "quick_enhance"),
                    ("📊 AI Insights", "ai_insights")
                ])
            
            for label, key in nav_items:
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
                    st.rerun()
            
            # Quick actions with performance indicators
            st.divider()
            st.subheader("Quick Actions")
            
            # Async directory scan
            if st.button("🔄 Scan Directory Async", use_container_width=True,
                        help="Scan directory in background without blocking UI"):
                task_id = self.manager.scan_directory_async()
                st.success(f"Scan started (Task: {task_id})")
                # Note: In production, you'd want to show progress
            
            # Cache management
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.manager.cache.clear()
                st.success("Cache cleared!")
            
            # Resource cleanup
            if st.button("🧹 Cleanup", use_container_width=True):
                self.manager.cleanup_resources()
                st.success("Resources cleaned up!")
            
            # Stats display
            st.divider()
            st.subheader("Stats")
            
            people = self.manager.db.get_all_people()
            total_images = sum(self.manager.get_person_stats(p['person_id'])['image_count'] 
                             for p in people)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("People", len(people))
            with col2:
                st.metric("Images", total_images)
            
            # Cache stats
            cache_stats = self.manager.cache.get_stats()
            st.caption(f"Cache: {cache_stats.get('hit_rate', '0%')} hit rate")
            
            # Performance warning if memory high
            try:
                import psutil
                memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                if memory_mb > EnhancedConfig.MAX_MEMORY_MB * 0.8:  # 80% threshold
                    st.warning(f"High memory: {memory_mb:.1f}MB")
            except:
                pass
    
    def render_performance_page(self):
        """Render performance monitoring and configuration page"""
        self.ui_components.render_performance_dashboard(self.manager)
        
        # Performance configuration
        st.divider()
        st.subheader("Performance Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cache configuration
            st.markdown("**Cache Settings**")
            cache_ttl = st.slider(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=86400,
                value=EnhancedConfig.CACHE_TTL,
                step=300,
                help="How long to cache data before refreshing"
            )
            
            max_cache_size = st.slider(
                "Max Cache Items",
                min_value=100,
                max_value=5000,
                value=EnhancedConfig.MAX_CACHE_SIZE,
                step=100,
                help="Maximum number of items to keep in cache"
            )
        
        with col2:
            # Database configuration
            st.markdown("**Database Settings**")
            pool_size = st.slider(
                "Connection Pool Size",
                min_value=1,
                max_value=20,
                value=EnhancedConfig.DB_CONNECTION_POOL_SIZE,
                step=1,
                help="Number of database connections to pool"
            )
            
            items_per_page = st.slider(
                "Items Per Page",
                min_value=5,
                max_value=100,
                value=EnhancedConfig.ITEMS_PER_PAGE,
                step=5,
                help="Number of items to show per page"
            )
        
        # Apply configuration
        if st.button("Apply Configuration", type="primary"):
            EnhancedConfig.CACHE_TTL = cache_ttl
            EnhancedConfig.MAX_CACHE_SIZE = max_cache_size
            EnhancedConfig.DB_CONNECTION_POOL_SIZE = pool_size
            EnhancedConfig.ITEMS_PER_PAGE = items_per_page
            
            # Update manager configuration
            self.manager.cache.max_memory_items = max_cache_size
            
            st.success("Configuration applied!")
            st.rerun()
        
        # Performance tips
        st.divider()
        st.subheader("Performance Tips")
        
        tips = [
            "✅ Use virtual scrolling for large galleries (>50 items)",
            "✅ Enable lazy loading for images",
            "✅ Use async operations for long-running tasks",
            "✅ Keep cache TTL appropriate for your usage patterns",
            "✅ Monitor memory usage and cleanup regularly",
            "⚠️ Disable AI features if not needed to save memory",
            "⚠️ Reduce image quality for thumbnails if memory is constrained"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")
    
    def render_optimized_gallery_page(self):
        """Render gallery page with performance optimizations"""
        st.title("📁 Gallery (Optimized)")
        
        # Performance controls
        with st.expander("⚡ Performance Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_virtual_scroll = st.checkbox(
                    "Virtual Scrolling", 
                    value=True,
                    help="Use virtual scrolling for better performance with many items"
                )
            
            with col2:
                lazy_load_images = st.checkbox(
                    "Lazy Load Images", 
                    value=True,
                    help="Load images only when they become visible"
                )
            
            with col3:
                items_per_row = st.slider(
                    "Items per Row",
                    min_value=2,
                    max_value=6,
                    value=EnhancedConfig.GRID_COLUMNS,
                    step=1
                )
        
        # Search and filter (original functionality preserved)
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                search_query = st.text_input("Search photos...", key="gallery_search_optimized")
            with col2:
                view_mode = st.selectbox("View", ["Grid", "List"], key="view_mode_optimized")
            with col3:
                sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Rating"], 
                                     key="gallery_sort_optimized")
            with col4:
                items_per_page = st.selectbox("Per Page", [12, 24, 48, 96], 
                                            key="items_per_page_optimized")
        
        # Get entries with optimized method
        selected_person_id = st.session_state.get('selected_person')
        page = st.session_state.get('gallery_page', 1)
        
        # Show loading indicator
        with st.spinner("Loading gallery..."):
            if selected_person_id:
                data = self.manager.get_entries_by_person_optimized(
                    selected_person_id, 
                    page, 
                    search_query,
                    batch_size=int(items_per_page)
                )
            else:
                # Get all entries with pagination
                offset = (page - 1) * int(items_per_page)
                data = self._get_all_entries_paginated_optimized(page, items_per_page, search_query)
        
        # Sort entries
        if sort_by == "Oldest":
            data['entries'].sort(key=lambda x: x.get('created_at', ''))
        elif sort_by == "Rating":
            data['entries'].sort(key=lambda x: x.get('avg_rating', 0), reverse=True)
        else:  # Newest (default)
            data['entries'].sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Render based on performance settings
        if use_virtual_scroll and len(data['entries']) > EnhancedConfig.VIRTUAL_SCROLLING_THRESHOLD:
            # Use virtual scrolling for large datasets
            self.ui_components.render_virtual_scroll_gallery(
                data['entries'],
                items_per_row=items_per_row,
                batch_size=int(items_per_page),
                height=600
            )
        elif view_mode == "Grid":
            # Regular grid with optimized rendering
            cols = st.columns(items_per_row)
            for idx, entry in enumerate(data['entries']):
                with cols[idx % items_per_row]:
                    self.ui_components._render_gallery_item_optimized(entry)
        else:  # List view
            for entry in data['entries']:
                self.ui_components._render_gallery_item_optimized(entry)
        
        # Pagination (original functionality preserved)
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
                    if page > 1 and st.button("◀", key="prev_page_optimized"):
                        st.session_state.gallery_page = page - 1
                        st.rerun()
                
                for idx, page_num in enumerate(page_numbers, 1):
                    with col_btns[idx]:
                        if page_num == "...":
                            st.markdown("...")
                        elif page_num == page:
                            st.markdown(f"**{page_num}**")
                        elif st.button(str(page_num), key=f"page_{page_num}_optimized"):
                            st.session_state.gallery_page = page_num
                            st.rerun()
                
                with col_btns[-1]:
                    if page < data['total_pages'] and st.button("▶", key="next_page_optimized"):
                        st.session_state.gallery_page = page + 1
                        st.rerun()
    
    def _get_all_entries_paginated_optimized(self, page: int, items_per_page: int, 
                                           search_query: str = None) -> Dict:
        """Optimized pagination for all entries"""
        offset = (page - 1) * items_per_page
        
        # Use cached query execution
        if search_query:
            search_pattern = f'%{search_query}%'
            query = '''
            SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                   (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
            FROM album_entries ae
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
            ORDER BY ae.created_at DESC
            LIMIT ? OFFSET ?
            '''
            params = (search_pattern, search_pattern, search_pattern, items_per_page, offset)
        else:
            query = '''
            SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                   (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
            FROM album_entries ae
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            ORDER BY ae.created_at DESC
            LIMIT ? OFFSET ?
            '''
            params = (items_per_page, offset)
        
        entries = self.manager.db.execute_cached_query(query, params, ttl=60)
        
        # Get total count
        if search_query:
            count_query = '''
            SELECT COUNT(*)
            FROM album_entries ae
            WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
            '''
            count_params = (search_pattern, search_pattern, search_pattern)
        else:
            count_query = 'SELECT COUNT(*) FROM album_entries'
            count_params = ()
        
        total_count = self.manager.db.execute_cached_query(count_query, count_params, ttl=300)
        total_count = total_count[0][0] if total_count else 0
        
        total_pages = max(1, math.ceil(total_count / items_per_page))
        
        return {
            'entries': entries,
            'total_count': total_count,
            'total_pages': total_pages,
            'current_page': page
        }
    
    def render_optimized_image_detail_page(self):
        """Render image detail page with optimizations"""
        entry_id = st.session_state.get('selected_image')
        if not entry_id:
            st.error("No image selected")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return
        
        # Get entry with optimized loading
        with st.spinner("Loading image details..."):
            entry = self.manager.get_entry_with_details_optimized(entry_id, preload_images=True)
        
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
            # Display image with optimization
            if entry.get('image_data_url'):
                st.image(
                    entry['image_data_url'],
                    use_column_width=True,
                    output_format="JPEG",
                    clamp=True
                )
            else:
                # Fallback to original method
                if entry.get('filepath'):
                    image_path = EnhancedConfig.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        data_url = OptimizedImageProcessor.get_image_data_url_optimized(
                            image_path,
                            max_size=EnhancedConfig.PREVIEW_SIZE
                        )
                        st.image(data_url, use_column_width=True)
                    else:
                        st.error("Image file not found")
                else:
                    st.error("No image available")
        
        with col2:
            # Image info (original functionality preserved)
            st.title(entry.get('caption', 'Untitled'))
            st.markdown(f"👤 **{entry.get('display_name', 'Unknown')}**")
            
            # Rating
            avg_rating = entry.get('avg_rating', 0)
            rating_count = entry.get('rating_count', 0)
            st.markdown(UIComponents.rating_stars(avg_rating), unsafe_allow_html=True)
            st.caption(f"{rating_count} ratings")
            
            # User rating (original functionality preserved)
            st.subheader("Your Rating")
            user_rating = entry.get('user_rating', 0)
            rating_cols = st.columns(5)
            for i in range(1, 6):
                with rating_cols[i-1]:
                    if st.button(f"{i} ⭐", key=f"rate_{i}_optimized", use_container_width=True):
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
            
            # Metadata in expander for compactness
            with st.expander("📊 Metadata", expanded=False):
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
        
        # AI Analysis section (preserved, with lazy loading)
        if EnhancedConfig.AI_ENABLED and hasattr(self.manager, 'analyze_image_with_ai'):
            st.divider()
            st.subheader("🤖 AI Analysis")
            
            if entry.get('filepath'):
                image_path = EnhancedConfig.DATA_DIR / entry['filepath']
                if image_path.exists():
                    if st.button("Analyze with AI", key="analyze_ai_optimized"):
                        with st.spinner("Analyzing image with AI..."):
                            # Lazy load AI modules if needed
                            if not AI_AVAILABLE:
                                lazy_load_ai_modules()
                            
                            if AI_AVAILABLE:
                                analysis = self.manager.analyze_image_with_ai(image_path)
                                
                                if analysis:
                                    # Use enhanced UI components for analysis display
                                    if hasattr(self.ui_components, 'ai_analysis_panel'):
                                        self.ui_components.ai_analysis_panel(analysis)
                                    else:
                                        # Fallback to original display
                                        st.json(analysis)
        
        # Comments section (original functionality preserved)
        st.divider()
        st.subheader("💬 Comments")
        
        # Add comment form
        with st.form(key="add_comment_form_optimized"):
            comment_text = st.text_area("Add a comment...", height=100, 
                                       max_chars=EnhancedConfig.MAX_COMMENT_LENGTH)
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
    
    def render_main_optimized(self):
        """Main application renderer with performance optimizations"""
        # Render optimized sidebar
        self.render_optimized_sidebar()
        
        # Get current page
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Performance optimization: periodic cleanup
        if 'last_cleanup' not in st.session_state:
            st.session_state.last_cleanup = time.time()
        
        if time.time() - st.session_state.last_cleanup > EnhancedConfig.AUTO_CLEANUP_INTERVAL:
            self.manager.cleanup_resources()
            st.session_state.last_cleanup = time.time()
        
        # Render appropriate page
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'people':
            self.render_people_page()
        elif current_page == 'gallery':
            self.render_optimized_gallery_page()
        elif current_page == 'image_detail':
            self.render_optimized_image_detail_page()
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
        elif current_page == 'ai_analysis':
            self.render_ai_analysis_page()
        elif current_page == 'ai_studio':
            self.render_ai_studio_page()
        elif current_page == 'quick_enhance':
            self.render_quick_enhance_page()
        elif current_page == 'ai_insights':
            self.render_ai_insights_page()
        elif current_page == 'performance':
            self.render_performance_page()
        else:
            self.render_dashboard()
        
        # Optimized footer
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.caption(f"© {datetime.datetime.now().year} {EnhancedConfig.APP_NAME} v{EnhancedConfig.VERSION}")
            
            # Show performance mode indicator
            if self.performance_mode:
                st.caption("⚡ Performance Mode Enabled")
            
            # Show memory usage
            try:
                import psutil
                memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                if memory_mb > EnhancedConfig.MAX_MEMORY_MB:
                    st.warning(f"High memory usage: {memory_mb:.1f}MB")
                else:
                    st.caption(f"Memory: {memory_mb:.1f}MB")
            except:
                pass

# ============================================================================
# PERFORMANCE DECORATORS AND UTILITIES
# ============================================================================

def performance_timer(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Store timing in session state for monitoring
        if 'performance_timings' not in st.session_state:
            st.session_state.performance_timings = {}
        
        func_name = func.__name__
        if func_name not in st.session_state.performance_timings:
            st.session_state.performance_timings[func_name] = {
                'count': 0,
                'total_time': 0,
                'avg_time': 0
            }
        
        timing = st.session_state.performance_timings[func_name]
        timing['count'] += 1
        timing['total_time'] += (end_time - start_time)
        timing['avg_time'] = timing['total_time'] / timing['count']
        
        return result
    return wrapper

def cache_result(ttl_seconds=300, max_size=100):
    """
    Decorator to cache function results with TTL and size limits
    
    Args:
        ttl_seconds: Time to live in seconds
        max_size: Maximum cache size
    """
    def decorator(func):
        # Create cache in session state
        cache_key = f"_cache_{func.__name__}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = OrderedDict()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            arg_str = str(args) + str(sorted(kwargs.items()))
            key_hash = hashlib.md5(arg_str.encode()).hexdigest()
            
            cache = st.session_state[cache_key]
            
            # Check cache
            if key_hash in cache:
                cached_data, timestamp = cache[key_hash]
                if time.time() - timestamp < ttl_seconds:
                    # Move to end (most recently used)
                    cache.move_to_end(key_hash)
                    return cached_data
                else:
                    # Remove expired entry
                    del cache[key_hash]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache[key_hash] = (result, time.time())
            
            # Enforce size limit
            if len(cache) > max_size:
                # Remove oldest entry
                cache.popitem(last=False)
            
            return result
        
        return wrapper
    return decorator

def async_operation(timeout=None):
    """
    Decorator to run function asynchronously
    
    Args:
        timeout: Optional timeout in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Submit to thread pool
            future = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(func, *args, **kwargs)
            
            if timeout:
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    return None
            else:
                return future.result()
        
        return wrapper
    return decorator

# ============================================================================
# MAIN APPLICATION ENTRY POINT WITH PERFORMANCE OPTIONS
# ============================================================================

def main_optimized():
    """Optimized main application entry point"""
    try:
        # Show performance mode selection
        st.sidebar.title("Performance Mode")
        
        mode = st.sidebar.selectbox(
            "Select Mode",
            ["Optimized", "Balanced", "Compatibility"],
            help="Optimized: Best performance, Balanced: Good performance with features, Compatibility: All features"
        )
        
        # Initialize based on mode
        if mode == "Optimized":
            # Use all optimizations
            EnhancedConfig.ITEMS_PER_PAGE = 12
            EnhancedConfig.GRID_COLUMNS = 3
            EnhancedConfig.LAZY_LOADING_ENABLED = True
            EnhancedConfig.VIRTUAL_SCROLLING_THRESHOLD = 30
            
            app = EnhancedPhotoAlbumApp()
            app.render_main_optimized()
            
        elif mode == "Balanced":
            # Balanced optimizations
            EnhancedConfig.ITEMS_PER_PAGE = 20
            EnhancedConfig.GRID_COLUMNS = 4
            EnhancedConfig.LAZY_LOADING_ENABLED = True
            EnhancedConfig.VIRTUAL_SCROLLING_THRESHOLD = 50
            
            app = EnhancedPhotoAlbumApp()
            app.render_main_optimized()
            
        else:  # Compatibility mode
            # Use original app with minimal optimizations
            EnhancedConfig.ITEMS_PER_PAGE = 20
            EnhancedConfig.GRID_COLUMNS = 4
            EnhancedConfig.LAZY_LOADING_ENABLED = False
            
            # Fall back to original PhotoAlbumApp
            from original_app import PhotoAlbumApp
            app = PhotoAlbumApp()
            app.render_main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
        with st.expander("Error Details"):
            st.exception(e)
        
        # Recovery options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try Again"):
                st.rerun()
        with col2:
            if st.button("Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# ============================================================================
# REQUIREMENTS GENERATOR WITH PERFORMANCE LIBRARIES
# ============================================================================

def generate_optimized_requirements():
    """Generate requirements.txt with performance libraries"""
    requirements = """# Core dependencies
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0

# Performance optimizations (optional but recommended)
numba>=0.58.0            # JIT compilation for numerical operations
joblib>=1.3.0           # Disk caching and parallel processing
psutil>=5.9.0           # Memory monitoring

# AI/ML dependencies (optional)
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.19.0
accelerate>=0.21.0

# Image processing (optional)
opencv-python>=4.8.0
scipy>=1.11.0
"""
    return requirements

# ============================================================================
# RUN OPTIMIZED APPLICATION
# ============================================================================

if __name__ == "__main__":
    # Show requirements in sidebar
    with st.sidebar.expander("📋 Requirements"):
        st.code(generate_optimized_requirements(), language="text")
        
        # Show available optimizations
        st.markdown("**Available Optimizations:**")
        
        optimizations = []
        if NUMBA_AVAILABLE:
            optimizations.append("✅ Numba (JIT compilation)")
        else:
            optimizations.append("❌ Numba (install for numerical optimizations)")
        
        if JOBLIB_AVAILABLE:
            optimizations.append("✅ Joblib (caching & parallel processing)")
        else:
            optimizations.append("❌ Joblib (install for disk caching)")
        
        for opt in optimizations:
            st.text(opt)
    
    # Show performance tips
    with st.sidebar.expander("💡 Performance Tips"):
        st.markdown("""
        1. Use **Optimized** mode for large albums
        2. Enable **Virtual Scrolling** for galleries >30 items
        3. Use **Async Scanning** for large directories
        4. Monitor memory usage in Performance page
        5. Clear cache periodically if memory is high
        6. Reduce image quality for thumbnails if needed
        """)
    
    # Run optimized application
    main_optimized()
