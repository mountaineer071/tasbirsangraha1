import streamlit as st
import json
import os
import uuid
import base64
import shutil
import time
import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import hashlib
import re
import mimetypes
from concurrent.futures import ThreadPoolExecutor
import random
from PIL import Image
import io
import imagehash
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import string
import tempfile
import gc

# ======================
# ENUMS & DATA CLASSES
# ======================
class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    FILE = "file"
    SYSTEM = "system"
    AUDIO = "audio"

class UserStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    AWAY = "away"

@dataclass
class User:
    id: str
    username: str
    session_id: str
    status: UserStatus
    last_seen: datetime
    color: str = "#667eea"
    avatar_seed: str = None

@dataclass
class ChatMessage:
    id: str
    session_id: str
    user_id: str
    username: str
    content: str
    type: MessageType
    timestamp: datetime
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    thumbnail_path: Optional[str] = None
    reply_to: Optional[str] = None
    edited: bool = False
    reactions: Dict[str, List[str]] = None
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['type'] = self.type.value
        data['status'] = self.status.value if hasattr(self, 'status') else 'sent'
        return data

# ======================
# ENHANCED CONFIGURATION
# ======================
class EnhancedConfig:
    """Enhanced configuration with monitoring and optimizations"""
    # Dynamic base directory with fallbacks
    BASE_DIR = Path(tempfile.gettempdir()) / "streamlit_temp_chat_pro"
    CHAT_DIR = BASE_DIR / "chats"
    TEMP_UPLOADS = BASE_DIR / "temp_uploads"
    THUMBNAILS = BASE_DIR / "thumbnails"
    AVATARS = BASE_DIR / "avatars"
    METADATA_FILE = BASE_DIR / "metadata.json"
    STATS_FILE = BASE_DIR / "stats.json"
    USER_DB = BASE_DIR / "users.json"
    
    # Security & Limits
    MAX_FILE_SIZE_MB = 100
    MAX_IMAGE_DIMENSION = 2048
    MAX_THUMBNAIL_SIZE = (320, 320)
    
    ALLOWED_EXTENSIONS = {
        'image': ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff'],
        'video': ['mp4', 'mov', 'avi', 'webm', 'mkv', 'flv'],
        'audio': ['mp3', 'wav', 'ogg', 'm4a', 'flac'],
        'document': ['pdf', 'txt', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'],
        'archive': ['zip', 'rar', '7z', 'tar', 'gz']
    }
    
    # Performance
    SESSION_TIMEOUT_MIN = 120
    AUTO_REFRESH_SEC = 2
    MESSAGE_BATCH_SIZE = 50
    CACHE_TTL_SEC = 300
    MAX_CONCURRENT_UPLOADS = 5
    
    # UI
    DEFAULT_AVATAR_COLORS = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    @classmethod
    def init_storage(cls):
        """Initialize all storage with atomic operations"""
        try:
            # Create all directories
            directories = [cls.CHAT_DIR, cls.TEMP_UPLOADS, cls.THUMBNAILS, cls.AVATARS]
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / ".gitkeep").touch(exist_ok=True)
            
            # Initialize metadata with versioning
            if not cls.METADATA_FILE.exists():
                cls._create_initial_metadata()
            
            # Initialize stats
            if not cls.STATS_FILE.exists():
                cls._initialize_stats()
            
            # Initialize user database
            if not cls.USER_DB.exists():
                cls._initialize_user_db()
            
            # Perform cleanup on startup
            cls._perform_startup_cleanup()
            
            # Start background tasks
            cls._start_background_tasks()
            
            st.session_state.setdefault('system_messages', [])
            st.session_state.setdefault('app_version', '2.0.0')
            
        except Exception as e:
            st.error(f"Storage initialization failed: {str(e)}")
            raise
    
    @classmethod
    def _create_initial_metadata(cls):
        """Create initial metadata file with schema version"""
        metadata = {
            "schema_version": 2,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "total_sessions": 0,
            "total_messages": 0,
            "total_files": 0,
            "active_sessions": {},
            "file_index": {},
            "chat_rooms": {
                "general": {
                    "name": "General",
                    "description": "Main chat room",
                    "created_at": datetime.now().isoformat(),
                    "message_count": 0
                }
            }
        }
        cls._atomic_write(cls.METADATA_FILE, metadata)
    
    @classmethod
    def _initialize_stats(cls):
        """Initialize statistics tracking"""
        stats = {
            "start_time": datetime.now().isoformat(),
            "uptime_seconds": 0,
            "messages_sent": 0,
            "files_uploaded": 0,
            "active_users": 0,
            "peak_concurrent": 0,
            "errors": 0,
            "performance": {
                "avg_response_time": 0,
                "cache_hit_rate": 0
            }
        }
        cls._atomic_write(cls.STATS_FILE, stats)
    
    @classmethod
    def _initialize_user_db(cls):
        """Initialize user database"""
        users = {
            "users": {},
            "sessions": {},
            "last_cleanup": datetime.now().isoformat()
        }
        cls._atomic_write(cls.USER_DB, users)
    
    @classmethod
    def _perform_startup_cleanup(cls):
        """Cleanup old sessions and orphaned files"""
        try:
            # Remove sessions older than timeout
            cutoff = datetime.now() - timedelta(minutes=cls.SESSION_TIMEOUT_MIN * 2)
            
            for chat_file in cls.CHAT_DIR.glob("*.json"):
                try:
                    with open(chat_file, 'r') as f:
                        messages = json.load(f)
                    if messages:
                        last_msg_time = datetime.fromisoformat(messages[-1]['timestamp'])
                        if last_msg_time < cutoff:
                            chat_file.unlink()
                except:
                    continue
            
            # Remove orphaned files
            cls._clean_orphaned_files()
            
            # Update metadata
            metadata = cls._read_metadata()
            metadata['last_cleanup'] = datetime.now().isoformat()
            metadata['active_sessions'] = {}
            cls._atomic_write(cls.METADATA_FILE, metadata)
            
        except Exception as e:
            st.warning(f"Cleanup had issues: {str(e)}")
    
    @classmethod
    def _clean_orphaned_files(cls):
        """Remove files not referenced in any chat"""
        try:
            # Get all referenced files from metadata
            metadata = cls._read_metadata()
            referenced_files = set()
            
            for session_data in metadata.get('active_sessions', {}).values():
                referenced_files.update(session_data.get('files', []))
            
            # Scan uploads directory
            for file_path in cls.TEMP_UPLOADS.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(cls.BASE_DIR))
                    if rel_path not in referenced_files:
                        try:
                            file_path.unlink()
                        except:
                            pass
            
        except Exception as e:
            st.warning(f"Orphaned file cleanup failed: {str(e)}")
    
    @classmethod
    def _start_background_tasks(cls):
        """Start background maintenance tasks"""
        if 'background_tasks' not in st.session_state:
            st.session_state.background_tasks = threading.Thread(
                target=cls._background_maintenance,
                daemon=True
            )
            st.session_state.background_tasks.start()
    
    @classmethod
    def _background_maintenance(cls):
        """Background maintenance thread"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                cls._update_stats()
                cls._clean_old_thumbnails()
                cls._compress_old_chats()
            except:
                pass
    
    @classmethod
    def _update_stats(cls):
        """Update runtime statistics"""
        try:
            stats = cls._read_stats()
            stats['uptime_seconds'] = int((datetime.now() - 
                datetime.fromisoformat(stats['start_time'])).total_seconds())
            
            metadata = cls._read_metadata()
            active_count = len(metadata.get('active_sessions', {}))
            stats['active_users'] = active_count
            stats['peak_concurrent'] = max(stats['peak_concurrent'], active_count)
            
            cls._atomic_write(cls.STATS_FILE, stats)
        except:
            pass
    
    @classmethod
    def _clean_old_thumbnails(cls):
        """Remove old thumbnail files"""
        try:
            cutoff = datetime.now() - timedelta(hours=24)
            for thumb in cls.THUMBNAILS.glob("*.jpg"):
                if thumb.stat().st_mtime < cutoff.timestamp():
                    thumb.unlink()
        except:
            pass
    
    @classmethod
    def _compress_old_chats(cls):
        """Compress old chat logs"""
        try:
            cutoff = datetime.now() - timedelta(hours=6)
            for chat_file in cls.CHAT_DIR.glob("*.json"):
                if chat_file.stat().st_mtime < cutoff.timestamp():
                    # Simple compression: remove unnecessary whitespace
                    with open(chat_file, 'r') as f:
                        data = json.load(f)
                    with open(chat_file, 'w') as f:
                        json.dump(data, f, separators=(',', ':'))
        except:
            pass
    
    @classmethod
    def _atomic_write(cls, file_path: Path, data: dict):
        """Atomic file write with retry"""
        temp_path = file_path.with_suffix('.tmp')
        for attempt in range(3):
            try:
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                temp_path.rename(file_path)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(0.1)
    
    @classmethod
    def _read_metadata(cls) -> dict:
        """Read metadata with error handling"""
        try:
            with open(cls.METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    @classmethod
    def _read_stats(cls) -> dict:
        """Read statistics"""
        try:
            with open(cls.STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}

# Initialize storage
EnhancedConfig.init_storage()

# ======================
# ENHANCED CACHE SYSTEM
# ======================
class MemoryCache:
    """LRU cache with TTL for performance"""
    
    def __init__(self, max_size=1000, ttl=300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get item from cache"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                self.hits += 1
                return self.cache[key]
            else:
                self.delete(key)
        self.misses += 1
        return None
    
    def set(self, key, value):
        """Set item in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.delete(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def delete(self, key):
        """Delete item from cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def hit_rate(self):
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

# Initialize cache
message_cache = MemoryCache(max_size=500, ttl=300)
file_cache = MemoryCache(max_size=100, ttl=600)

# ======================
# ENHANCED MANAGERS
# ======================
class EnhancedChatManager:
    """Enhanced chat operations with real-time features"""
    
    _executor = ThreadPoolExecutor(max_workers=EnhancedConfig.MAX_CONCURRENT_UPLOADS)
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(12)
    
    @staticmethod
    def get_session_id() -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = EnhancedChatManager.generate_session_id()
            
            # Register session in metadata
            metadata = EnhancedConfig._read_metadata()
            metadata['active_sessions'][st.session_state.session_id] = {
                'started_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat(),
                'message_count': 0,
                'files': []
            }
            EnhancedConfig._atomic_write(EnhancedConfig.METADATA_FILE, metadata)
            
        return st.session_state.session_id
    
    @staticmethod
    def generate_user_color(username: str) -> str:
        """Generate deterministic color for username"""
        hash_val = hashlib.md5(username.encode()).hexdigest()[:6]
        return f"#{hash_val}"
    
    @staticmethod
    def get_or_create_user() -> User:
        """Get or create user profile"""
        session_id = EnhancedChatManager.get_session_id()
        
        # Check cache first
        cache_key = f"user_{session_id}"
        cached_user = message_cache.get(cache_key)
        if cached_user:
            return User(**cached_user)
        
        # Read user database
        try:
            with open(EnhancedConfig.USER_DB, 'r') as f:
                user_db = json.load(f)
        except:
            user_db = {"users": {}, "sessions": {}}
        
        user_id = st.session_state.get('user_id', str(uuid.uuid4()))
        st.session_state.user_id = user_id
        
        if user_id in user_db['users']:
            user_data = user_db['users'][user_id]
            user = User(
                id=user_id,
                username=user_data['username'],
                session_id=session_id,
                status=UserStatus(user_data.get('status', 'online')),
                last_seen=datetime.fromisoformat(user_data['last_seen']),
                color=user_data.get('color', EnhancedConfig.DEFAULT_AVATAR_COLORS[0]),
                avatar_seed=user_data.get('avatar_seed')
            )
        else:
            # Create new user
            default_name = f"User_{session_id[:8]}"
            username = st.session_state.get('username_input', default_name)
            clean_username = re.sub(r'[^\w\-_\. ]', '', username.strip())[:20]
            clean_username = clean_username if clean_username else default_name
            
            color = EnhancedChatManager.generate_user_color(clean_username)
            avatar_seed = secrets.token_hex(8)
            
            user = User(
                id=user_id,
                username=clean_username,
                session_id=session_id,
                status=UserStatus.ONLINE,
                last_seen=datetime.now(),
                color=color,
                avatar_seed=avatar_seed
            )
            
            # Save to database
            user_db['users'][user_id] = {
                'username': clean_username,
                'created_at': datetime.now().isoformat(),
                'last_seen': user.last_seen.isoformat(),
                'status': user.status.value,
                'color': color,
                'avatar_seed': avatar_seed,
                'total_messages': 0,
                'total_files': 0
            }
            user_db['sessions'][session_id] = user_id
            
            EnhancedConfig._atomic_write(EnhancedConfig.USER_DB, user_db)
        
        # Cache user
        message_cache.set(cache_key, asdict(user))
        return user
    
    @staticmethod
    def update_user_status(status: UserStatus):
        """Update user status"""
        user = EnhancedChatManager.get_or_create_user()
        user.status = status
        user.last_seen = datetime.now()
        
        # Update database
        try:
            with open(EnhancedConfig.USER_DB, 'r') as f:
                user_db = json.load(f)
            
            if user.id in user_db['users']:
                user_db['users'][user.id]['status'] = status.value
                user_db['users'][user.id]['last_seen'] = user.last_seen.isoformat()
                
                EnhancedConfig._atomic_write(EnhancedConfig.USER_DB, user_db)
                message_cache.delete(f"user_{user.session_id}")
        except:
            pass
    
    @staticmethod
    def _get_chat_file(room: str = "general") -> Path:
        """Get chat file for specific room"""
        session_id = EnhancedChatManager.get_session_id()
        safe_room = re.sub(r'[^\w\-_]', '', room.lower())
        return EnhancedConfig.CHAT_DIR / f"chat_{session_id}_{safe_room}.json"
    
    @staticmethod
    def get_available_rooms() -> List[Dict]:
        """Get list of available chat rooms"""
        metadata = EnhancedConfig._read_metadata()
        return [
            {"id": room_id, **room_data}
            for room_id, room_data in metadata.get('chat_rooms', {}).items()
        ]
    
    @staticmethod
    def load_messages(room: str = "general", limit: int = None) -> List[Dict]:
        """Load messages with pagination"""
        cache_key = f"messages_{EnhancedChatManager.get_session_id()}_{room}_{limit}"
        cached = message_cache.get(cache_key)
        if cached:
            return cached
        
        chat_file = EnhancedChatManager._get_chat_file(room)
        if not chat_file.exists():
            return []
        
        try:
            with open(chat_file, 'r') as f:
                messages = json.load(f)
            
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            message_cache.set(cache_key, messages)
            return messages
        except Exception as e:
            st.error(f"Failed to load messages: {str(e)}")
            return []
    
    @staticmethod
    def save_message(content: str, msg_type: MessageType, 
                    file_path: Optional[Path] = None, 
                    file_size: Optional[int] = None,
                    file_type: Optional[str] = None,
                    thumbnail_path: Optional[Path] = None,
                    reply_to: Optional[str] = None,
                    room: str = "general") -> Optional[str]:
        """Save message with enhanced features"""
        if not content.strip() and not file_path:
            return None
        
        user = EnhancedChatManager.get_or_create_user()
        message_id = str(uuid.uuid4())
        
        message = ChatMessage(
            id=message_id,
            session_id=user.session_id,
            user_id=user.id,
            username=user.username,
            content=content.strip() if content else "",
            type=msg_type,
            timestamp=datetime.now(),
            file_path=str(file_path.relative_to(EnhancedConfig.BASE_DIR)) if file_path else None,
            file_size=file_size,
            file_type=file_type,
            thumbnail_path=str(thumbnail_path.relative_to(EnhancedConfig.BASE_DIR)) if thumbnail_path else None,
            reply_to=reply_to,
            reactions={}
        )
        
        # Load existing messages
        messages = EnhancedChatManager.load_messages(room)
        messages.append(message.to_dict())
        
        # Truncate if too many messages
        if len(messages) > 1000:
            messages = messages[-500:]
        
        # Save to file
        chat_file = EnhancedChatManager._get_chat_file(room)
        EnhancedConfig._atomic_write(chat_file, messages)
        
        # Update metadata
        metadata = EnhancedConfig._read_metadata()
        if user.session_id in metadata['active_sessions']:
            metadata['active_sessions'][user.session_id]['last_activity'] = datetime.now().isoformat()
            metadata['active_sessions'][user.session_id]['message_count'] += 1
        
        if room in metadata['chat_rooms']:
            metadata['chat_rooms'][room]['message_count'] += 1
        
        metadata['total_messages'] = metadata.get('total_messages', 0) + 1
        metadata['last_modified'] = datetime.now().isoformat()
        
        EnhancedConfig._atomic_write(EnhancedConfig.METADATA_FILE, metadata)
        
        # Update stats
        EnhancedChatManager._update_message_stats()
        
        # Clear relevant caches
        message_cache.delete(f"messages_{user.session_id}_{room}_*")
        
        return message_id
    
    @staticmethod
    def _update_message_stats():
        """Update message statistics"""
        try:
            stats = EnhancedConfig._read_stats()
            stats['messages_sent'] = stats.get('messages_sent', 0) + 1
            EnhancedConfig._atomic_write(EnhancedConfig.STATS_FILE, stats)
        except:
            pass
    
    @staticmethod
    def add_reaction(message_id: str, reaction: str, room: str = "general"):
        """Add reaction to message"""
        messages = EnhancedChatManager.load_messages(room)
        user = EnhancedChatManager.get_or_create_user()
        
        for msg in messages:
            if msg['id'] == message_id:
                if 'reactions' not in msg:
                    msg['reactions'] = {}
                if reaction not in msg['reactions']:
                    msg['reactions'][reaction] = []
                if user.id not in msg['reactions'][reaction]:
                    msg['reactions'][reaction].append(user.id)
                break
        
        chat_file = EnhancedChatManager._get_chat_file(room)
        EnhancedConfig._atomic_write(chat_file, messages)
        message_cache.delete(f"messages_{user.session_id}_{room}_*")
    
    @staticmethod
    def edit_message(message_id: str, new_content: str, room: str = "general") -> bool:
        """Edit existing message"""
        messages = EnhancedChatManager.load_messages(room)
        user = EnhancedChatManager.get_or_create_user()
        
        for msg in messages:
            if msg['id'] == message_id and msg['user_id'] == user.id:
                msg['content'] = new_content.strip()
                msg['edited'] = True
                msg['edit_timestamp'] = datetime.now().isoformat()
                
                chat_file = EnhancedChatManager._get_chat_file(room)
                EnhancedConfig._atomic_write(chat_file, messages)
                message_cache.delete(f"messages_{user.session_id}_{room}_*")
                return True
        
        return False
    
    @staticmethod
    def delete_message(message_id: str, room: str = "general") -> bool:
        """Delete message (soft delete)"""
        messages = EnhancedChatManager.load_messages(room)
        user = EnhancedChatManager.get_or_create_user()
        
        for i, msg in enumerate(messages):
            if msg['id'] == message_id and msg['user_id'] == user.id:
                messages[i] = {
                    **msg,
                    'deleted': True,
                    'deleted_at': datetime.now().isoformat(),
                    'content': '[Message deleted]',
                    'file_path': None,
                    'thumbnail_path': None
                }
                
                chat_file = EnhancedChatManager._get_chat_file(room)
                EnhancedConfig._atomic_write(chat_file, messages)
                message_cache.delete(f"messages_{user.session_id}_{room}_*")
                return True
        
        return False
    
    @staticmethod
    def get_active_users() -> List[User]:
        """Get list of active users"""
        try:
            with open(EnhancedConfig.USER_DB, 'r') as f:
                user_db = json.load(f)
            
            active_users = []
            cutoff = datetime.now() - timedelta(minutes=5)
            
            for user_id, user_data in user_db['users'].items():
                last_seen = datetime.fromisoformat(user_data['last_seen'])
                if last_seen > cutoff:
                    user = User(
                        id=user_id,
                        username=user_data['username'],
                        session_id=user_data.get('session_id', ''),
                        status=UserStatus(user_data['status']),
                        last_seen=last_seen,
                        color=user_data.get('color', '#667eea'),
                        avatar_seed=user_data.get('avatar_seed')
                    )
                    active_users.append(user)
            
            return active_users
        except:
            return []


class EnhancedFileStorageManager:
    """Enhanced file handling with thumbnails and optimizations"""
    
    @staticmethod
    def validate_file(file) -> Tuple[bool, str, str]:
        """Validate file with detailed error messages"""
        if file.size == 0:
            return False, "Empty file not allowed", ""
        
        max_size = EnhancedConfig.MAX_FILE_SIZE_MB * 1024 * 1024
        if file.size > max_size:
            return False, f"File exceeds {EnhancedConfig.MAX_FILE_SIZE_MB}MB limit", ""
        
        filename = file.name.lower()
        ext = filename.split('.')[-1] if '.' in filename else ''
        
        # Determine file type
        file_type = None
        for category, exts in EnhancedConfig.ALLOWED_EXTENSIONS.items():
            if ext in exts:
                file_type = category
                break
        
        if not file_type:
            allowed = ', '.join(sorted(set(
                ext for cat in EnhancedConfig.ALLOWED_EXTENSIONS.values() for ext in cat
            )))
            return False, f"Invalid file type. Allowed: {allowed}", ""
        
        return True, "", file_type
    
    @staticmethod
    def generate_thumbnail(image_path: Path) -> Optional[Path]:
        """Generate thumbnail for image"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                
                # Resize maintaining aspect ratio
                img.thumbnail(EnhancedConfig.MAX_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                thumb_path = EnhancedConfig.THUMBNAILS / f"{image_path.stem}_thumb.jpg"
                img.save(thumb_path, "JPEG", quality=85, optimize=True)
                
                return thumb_path
        except Exception as e:
            st.warning(f"Thumbnail generation failed: {str(e)}")
            return None
    
    @staticmethod
    def optimize_image(image_path: Path) -> Path:
        """Optimize image for web display"""
        try:
            with Image.open(image_path) as img:
                # Resize if too large
                max_dim = EnhancedConfig.MAX_IMAGE_DIMENSION
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to web-friendly format
                if image_path.suffix.lower() in ['.png', '.bmp', '.tiff']:
                    optimized_path = image_path.with_suffix('.jpg')
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = rgb_img
                    img.save(optimized_path, "JPEG", quality=85, optimize=True)
                    
                    # Remove original if different
                    if optimized_path != image_path:
                        image_path.unlink()
                    
                    return optimized_path
            
            return image_path
        except Exception as e:
            st.warning(f"Image optimization failed: {str(e)}")
            return image_path
    
    @staticmethod
    def save_file(uploaded_file, user_id: str) -> Tuple[Optional[Path], Optional[Path], str]:
        """Save file with optimizations and thumbnails"""
        is_valid, error, file_type = EnhancedFileStorageManager.validate_file(uploaded_file)
        if not is_valid:
            st.error(f"Upload failed: {error}")
            return None, None, file_type
        
        try:
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            clean_name = re.sub(r'[^\w\-_\.]', '_', uploaded_file.name)
            filename = f"{user_id}_{timestamp}_{clean_name}"
            file_path = EnhancedConfig.TEMP_UPLOADS / filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Verify integrity
            if file_path.stat().st_size != len(uploaded_file.getbuffer()):
                file_path.unlink()
                st.error("File corruption detected")
                return None, None, file_type
            
            thumbnail_path = None
            
            # Process based on file type
            if file_type == 'image':
                # Optimize image
                optimized_path = EnhancedFileStorageManager.optimize_image(file_path)
                if optimized_path != file_path:
                    file_path = optimized_path
                
                # Generate thumbnail
                thumbnail_path = EnhancedFileStorageManager.generate_thumbnail(file_path)
            
            # Record in metadata
            EnhancedFileStorageManager._record_file(file_path, thumbnail_path, 
                                                  uploaded_file.name, user_id, file_type)
            
            # Cache file info
            cache_key = f"file_{file_path.name}"
            file_cache.set(cache_key, {
                'path': str(file_path),
                'size': file_path.stat().st_size,
                'type': file_type
            })
            
            return file_path, thumbnail_path, file_type
            
        except Exception as e:
            st.error(f"Save failed: {str(e)}")
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            return None, None, file_type
    
    @staticmethod
    def _record_file(file_path: Path, thumbnail_path: Optional[Path], 
                    original_name: str, user_id: str, file_type: str):
        """Record file in metadata and user database"""
        try:
            # Update metadata
            metadata = EnhancedConfig._read_metadata()
            
            file_record = {
                'path': str(file_path.relative_to(EnhancedConfig.BASE_DIR)),
                'original_name': original_name,
                'user_id': user_id,
                'uploaded_at': datetime.now().isoformat(),
                'size': file_path.stat().st_size,
                'type': file_type,
                'thumbnail': str(thumbnail_path.relative_to(EnhancedConfig.BASE_DIR)) if thumbnail_path else None
            }
            
            metadata.setdefault('file_index', {})[file_path.name] = file_record
            
            # Add to user's session
            session_id = EnhancedChatManager.get_session_id()
            if session_id in metadata['active_sessions']:
                metadata['active_sessions'][session_id].setdefault('files', []).append(
                    file_record['path']
                )
            
            metadata['total_files'] = metadata.get('total_files', 0) + 1
            EnhancedConfig._atomic_write(EnhancedConfig.METADATA_FILE, metadata)
            
            # Update user stats
            EnhancedFileStorageManager._update_user_file_stats(user_id)
            
            # Update global stats
            stats = EnhancedConfig._read_stats()
            stats['files_uploaded'] = stats.get('files_uploaded', 0) + 1
            EnhancedConfig._atomic_write(EnhancedConfig.STATS_FILE, stats)
            
        except Exception as e:
            st.warning(f"File recording failed: {str(e)}")
    
    @staticmethod
    def _update_user_file_stats(user_id: str):
        """Update user's file statistics"""
        try:
            with open(EnhancedConfig.USER_DB, 'r') as f:
                user_db = json.load(f)
            
            if user_id in user_db['users']:
                user_db['users'][user_id]['total_files'] = \
                    user_db['users'][user_id].get('total_files', 0) + 1
                
                EnhancedConfig._atomic_write(EnhancedConfig.USER_DB, user_db)
        except:
            pass
    
    @staticmethod
    def get_file_info(rel_path: str) -> Optional[Dict]:
        """Get file information from metadata"""
        try:
            metadata = EnhancedConfig._read_metadata()
            for file_data in metadata.get('file_index', {}).values():
                if file_data['path'] == rel_path:
                    return file_data
        except:
            pass
        return None
    
    @staticmethod
    def get_session_files(user_id: str) -> List[Dict]:
        """Get files uploaded by user in current session"""
        try:
            metadata = EnhancedConfig._read_metadata()
            user_files = []
            
            for file_data in metadata.get('file_index', {}).values():
                if file_data.get('user_id') == user_id:
                    # Check if file exists
                    abs_path = EnhancedConfig.BASE_DIR / file_data['path']
                    if abs_path.exists():
                        user_files.append(file_data)
            
            return sorted(user_files, 
                         key=lambda x: x['uploaded_at'], 
                         reverse=True)
        except Exception as e:
            st.warning(f"Failed to get session files: {str(e)}")
            return []
    
    @staticmethod
    def get_file_preview_url(file_path: Path) -> Optional[str]:
        """Generate data URL for file preview"""
        try:
            if not file_path.exists():
                return None
            
            mime_type, _ = mimetypes.guess_type(file_path.name)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            with open(file_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode()
            
            return f"data:{mime_type};base64,{data}"
        except:
            return None

# ======================
# ENHANCED UI COMPONENTS
# ======================
class EnhancedChatUI:
    """Complete chat interface with advanced features"""
    
    # Emoji reactions
    REACTION_EMOJIS = ["👍", "❤️", "😂", "😮", "😢", "🔥", "👏", "🎉"]
    
    @staticmethod
    def render_custom_css():
        """Render all custom CSS"""
        st.markdown("""
        <style>
        /* Main theme */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --dark-bg: #0f172a;
            --card-bg: rgba(255, 255, 255, 0.95);
            --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.15);
            --shadow-sm: 0 2px 10px rgba(0, 0, 0, 0.08);
        }
        
        /* Header */
        .main-header {
            background: var(--primary-gradient);
            padding: 1.5rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        .header-title {
            color: white;
            font-weight: 800;
            margin: 0;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header-subtitle {
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        
        /* User badge */
        .user-badge {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.8rem;
            font-weight: 500;
        }
        .user-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--secondary-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
        }
        
        /* Message containers */
        .message-container {
            background: var(--card-bg);
            border-radius: 18px;
            padding: 1.2rem;
            margin: 0.8rem 0;
            box-shadow: var(--shadow-sm);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .message-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
            border-left: 5px solid #3b82f6;
            margin-left: 20%;
        }
        .other-message {
            background: linear-gradient(135deg, #f3f4f6 0%, #f9fafb 100%);
            border-left: 5px solid #9ca3af;
            margin-right: 20%;
        }
        .system-message {
            background: linear-gradient(135deg, #fef3c7 0%, #fef9c3 100%);
            border-left: 5px solid #f59e0b;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            text-align: center;
        }
        
        /* Message header */
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .message-user {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            color: #1f2937;
        }
        .message-time {
            font-size: 0.8rem;
            color: #6b7280;
            font-weight: 500;
        }
        
        /* File previews */
        .file-preview {
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.8rem 0;
            background: rgba(255, 255, 255, 0.5);
            transition: all 0.3s;
        }
        .file-preview:hover {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.05);
        }
        .image-preview {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        /* Reactions */
        .reactions-container {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }
        .reaction {
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 0.2rem 0.6rem;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }
        .reaction:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: #3b82f6;
            transform: scale(1.05);
        }
        .reaction.active {
            background: rgba(59, 130, 246, 0.2);
            border-color: #3b82f6;
        }
        .reaction-count {
            font-size: 0.8rem;
            color: #4b5563;
            font-weight: 500;
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.5rem 1rem;
            background: #f3f4f6;
            border-radius: 20px;
            width: fit-content;
            margin: 0.5rem 0;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            background: #9ca3af;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        
        /* Input area */
        .input-container {
            position: sticky;
            bottom: 0;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-top: 1px solid #e5e7eb;
            margin-top: 2rem;
            z-index: 100;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a1a1a1;
        }
        
        /* Status indicators */
        .status-online { color: #10b981; }
        .status-offline { color: #9ca3af; }
        .status-away { color: #f59e0b; }
        
        /* Loading spinner */
        .loading-spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 2rem auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Room selector */
        .room-tab {
            padding: 0.8rem 1.2rem;
            border-radius: 12px;
            margin: 0.2rem 0;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        .room-tab:hover {
            background: #f3f4f6;
        }
        .room-tab.active {
            background: var(--primary-gradient);
            color: white;
        }
        
        /* Badges */
        .notification-badge {
            background: #ef4444;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            margin-left: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render enhanced app header"""
        user = EnhancedChatManager.get_or_create_user()
        
        st.markdown(f"""
        <div class="main-header">
            <h1 class="header-title">💬 TempChat Pro</h1>
            <p class="header-subtitle">
                🔥 Real-time temporary chat with file sharing • All data auto-deletes on reboot
            </p>
            <div class="user-badge">
                <div class="user-avatar" style="background: {user.color}">
                    {user.username[0].upper()}
                </div>
                <span>{user.username}</span>
                <span class="status-{user.status.value}">•</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_user_profile():
        """Render user profile in sidebar"""
        with st.sidebar:
            user = EnhancedChatManager.get_or_create_user()
            
            col1, col2 = st.columns([1, 3])
            with col1:
                # Avatar with gradient
                avatar_html = f"""
                <div style="
                    width: 60px;
                    height: 60px;
                    background: {user.color};
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    font-weight: bold;
                    color: white;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                ">
                    {user.username[0].upper()}
                </div>
                """
                st.markdown(avatar_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                **{user.username}**  
                <span style="color: {'#10b981' if user.status == UserStatus.ONLINE else '#9ca3af'}; font-size: 0.9rem">
                    ● {user.status.value.capitalize()}
                </span>
                """, unsafe_allow_html=True)
            
            st.divider()
    
    @staticmethod
    def render_room_selector():
        """Render chat room selector"""
        rooms = EnhancedChatManager.get_available_rooms()
        current_room = st.session_state.get('current_room', 'general')
        
        with st.sidebar:
            st.markdown("### 📝 Chat Rooms")
            
            for room in rooms:
                is_active = room['id'] == current_room
                room_name = room.get('name', room['id'].title())
                message_count = room.get('message_count', 0)
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"#{room_name}",
                        key=f"room_{room['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_room = room['id']
                        st.session_state.force_refresh = True
                        st.rerun()
                
                with col2:
                    if message_count > 0:
                        st.markdown(f"<small>{message_count}</small>", unsafe_allow_html=True)
    
    @staticmethod
    def render_active_users():
        """Render active users panel"""
        active_users = EnhancedChatManager.get_active_users()
        current_user = EnhancedChatManager.get_or_create_user()
        
        with st.sidebar:
            st.markdown(f"### 👥 Active Users ({len(active_users)})")
            
            for user in active_users:
                if user.id == current_user.id:
                    continue
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    # User status indicator
                    status_color = {
                        UserStatus.ONLINE: "#10b981",
                        UserStatus.AWAY: "#f59e0b",
                        UserStatus.OFFLINE: "#9ca3af"
                    }.get(user.status, "#9ca3af")
                    
                    st.markdown(f"""
                    <div style="
                        width: 10px;
                        height: 10px;
                        background: {status_color};
                        border-radius: 50%;
                        margin-top: 8px;
                    "></div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    **{user.username}**  
                    <small style="color: #6b7280">
                        {user.last_seen.strftime('%I:%M %p')}
                    </small>
                    """, unsafe_allow_html=True)
            
            if len(active_users) <= 1:
                st.info("👋 You're the first one here! Invite others to join.")
    
    @staticmethod
    def render_message(msg: Dict, current_user_id: str):
        """Render individual message with reactions"""
        is_current_user = msg.get('user_id') == current_user_id
        align = "flex-end" if is_current_user else "flex-start"
        msg_class = "user-message" if is_current_user else "other-message"
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(msg['timestamp'])
            time_str = timestamp.strftime('%I:%M %p')
            date_str = timestamp.strftime('%b %d')
        except:
            time_str = date_str = "Unknown"
        
        # Check if message is deleted
        is_deleted = msg.get('deleted', False)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: {align}; margin: 0.8rem 0;">
            <div class="message-container {msg_class}" style="max-width: 75%;">
                <div class="message-header">
                    <div class="message-user">
                        <div style="
                            width: 24px;
                            height: 24px;
                            border-radius: 50%;
                            background: linear-gradient(135deg, #{hashlib.md5(msg['username'].encode()).hexdigest()[:6]} 0%, #{hashlib.md5(msg['user_id'].encode()).hexdigest()[:6]} 100%);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 0.8rem;
                            color: white;
                            font-weight: bold;
                        ">
                            {msg['username'][0].upper()}
                        </div>
                        {msg['username']}
                    </div>
                    <div class="message-time">
                        {time_str} • {date_str}
                        {msg.get('edited', False) and ' (edited)' or ''}
                    </div>
                </div>
        """, unsafe_allow_html=True)
        
        # Handle different message types
        if is_deleted:
            st.markdown("<em style='color: #9ca3af;'>[Message deleted]</em>", unsafe_allow_html=True)
        else:
            if msg['type'] in ['image', 'video', 'file']:
                EnhancedChatUI._render_file_message(msg)
            else:
                # Text message with markdown support
                st.markdown(msg['content'])
        
        # Render reactions
        reactions = msg.get('reactions', {})
        if reactions:
            EnhancedChatUI._render_reactions(msg['id'], reactions, current_user_id)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Context menu for user's own messages
        if is_current_user and not is_deleted:
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("✏️", key=f"edit_{msg['id']}", help="Edit"):
                    st.session_state.editing_message = msg['id']
                    st.session_state.edit_content = msg['content']
            with col2:
                if st.button("🗑️", key=f"delete_{msg['id']}", help="Delete"):
                    if EnhancedChatManager.delete_message(msg['id'], 
                                                        st.session_state.get('current_room', 'general')):
                        st.session_state.system_messages.append("Message deleted")
                        st.rerun()
    
    @staticmethod
    def _render_file_message(msg: Dict):
        """Render file message with preview"""
        file_info = EnhancedFileStorageManager.get_file_info(msg.get('file_path', ''))
        
        if not file_info:
            st.warning("⚠️ File no longer available")
            return
        
        file_type = file_info.get('type', 'file')
        original_name = file_info.get('original_name', 'Unknown')
        
        if file_type == 'image':
            # Try to load thumbnail
            thumbnail_path = file_info.get('thumbnail')
            if thumbnail_path:
                abs_thumb_path = EnhancedConfig.BASE_DIR / thumbnail_path
                if abs_thumb_path.exists():
                    st.image(str(abs_thumb_path), use_column_width=True)
            
            # Caption
            if msg.get('content'):
                st.caption(msg['content'])
            
            # Download button
            abs_file_path = EnhancedConfig.BASE_DIR / file_info['path']
            if abs_file_path.exists():
                with open(abs_file_path, 'rb') as f:
                    st.download_button(
                        label=f"📥 Download {original_name}",
                        data=f,
                        file_name=original_name,
                        mime="image/jpeg",
                        key=f"dl_img_{msg['id']}"
                    )
        
        elif file_type == 'video':
            abs_file_path = EnhancedConfig.BASE_DIR / file_info['path']
            if abs_file_path.exists():
                st.video(str(abs_file_path))
                if msg.get('content'):
                    st.caption(msg['content'])
        
        else:
            # Generic file
            st.markdown(f"""
            <div class="file-preview">
                <div style="font-size: 3rem; text-align: center; margin-bottom: 0.5rem;">
                    📎
                </div>
                <div style="text-align: center; font-weight: 500; margin-bottom: 0.5rem;">
                    {original_name}
                </div>
                <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
                    {file_info.get('size', 0) / 1024:.1f} KB
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if msg.get('content'):
                st.caption(msg['content'])
    
    @staticmethod
    def _render_reactions(message_id: str, reactions: Dict[str, List[str]], current_user_id: str):
        """Render message reactions"""
        st.markdown('<div class="reactions-container">', unsafe_allow_html=True)
        
        for emoji, user_ids in reactions.items():
            count = len(user_ids)
            user_reacted = current_user_id in user_ids
            
            if st.button(
                f"{emoji} {count}",
                key=f"react_{message_id}_{emoji}",
                help=f"React with {emoji}",
                type="primary" if user_reacted else "secondary"
            ):
                EnhancedChatManager.add_reaction(message_id, emoji, 
                                               st.session_state.get('current_room', 'general'))
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def render_chat_messages():
        """Render all chat messages with virtual scrolling"""
        current_room = st.session_state.get('current_room', 'general')
        user = EnhancedChatManager.get_or_create_user()
        
        # Load messages
        messages = EnhancedChatManager.load_messages(current_room, 
                                                   EnhancedConfig.MESSAGE_BATCH_SIZE)
        
        if not messages:
            st.info("""
            👋 Welcome to the chat!
            
            **Get started:**
            1. Type a message below
            2. Upload files (images, videos, documents)
            3. Invite others to join
            
            All data is temporary and will be cleared on server reboot.
            """)
            return
        
        # Render messages in reverse order (newest at bottom)
        container = st.container()
        with container:
            for msg in messages[-100:]:  # Limit for performance
                EnhancedChatUI.render_message(msg, user.id)
            
            # Auto-scroll to bottom
            st.markdown("""
            <script>
            // Auto-scroll to bottom
            window.scrollTo(0, document.body.scrollHeight);
            </script>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_input_area():
        """Enhanced input area with file upload and emoji support"""
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "📎 Attach files (drag & drop)",
            type=[ext for cat in EnhancedConfig.ALLOWED_EXTENSIONS.values() for ext in cat],
            accept_multiple_files=False,
            key=f"file_upload_{int(time.time())}",
            help=f"Max size: {EnhancedConfig.MAX_FILE_SIZE_MB}MB"
        )
        
        # Message input with emoji picker
        col1, col2, col3 = st.columns([1, 8, 1])
        
        with col1:
            if st.button("😀", use_container_width=True, help="Emoji picker"):
                st.session_state.show_emoji_picker = not st.session_state.get('show_emoji_picker', False)
        
        with col2:
            # Check if editing existing message
            if 'editing_message' in st.session_state:
                message = st.text_area(
                    "Edit message",
                    value=st.session_state.edit_content,
                    key="edit_input",
                    height=100
                )
                
                col_edit1, col_edit2 = st.columns(2)
                with col_edit1:
                    if st.button("💾 Save", use_container_width=True):
                        if EnhancedChatManager.edit_message(
                            st.session_state.editing_message,
                            message,
                            st.session_state.get('current_room', 'general')
                        ):
                            st.session_state.pop('editing_message')
                            st.session_state.pop('edit_content')
                            st.session_state.system_messages.append("Message edited")
                            st.rerun()
                
                with col_edit2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.pop('editing_message')
                        st.session_state.pop('edit_content')
                        st.rerun()
            else:
                # Normal message input
                message = st.text_input(
                    "Type your message...",
                    key=f"chat_input_{int(time.time())}",
                    placeholder="Press Enter to send",
                    label_visibility="collapsed"
                )
        
        with col3:
            send_disabled = not (message.strip() or uploaded_file)
            if st.button("➤", 
                        type="primary", 
                        use_container_width=True,
                        disabled=send_disabled,
                        help="Send message"):
                EnhancedChatUI._handle_send(message, uploaded_file)
                st.rerun()
        
        # Emoji picker
        if st.session_state.get('show_emoji_picker', False):
            st.markdown("---")
            st.markdown("### Emojis")
            emoji_cols = st.columns(8)
            emojis = ["😀", "😂", "🥰", "😎", "😭", "😡", "👍", "❤️",
                     "🔥", "🎉", "👏", "🙏", "🤔", "👀", "💯", "✨"]
            
            for idx, emoji in enumerate(emojis):
                with emoji_cols[idx % 8]:
                    if st.button(emoji, key=f"emoji_{emoji}"):
                        # Add emoji to current message
                        current_msg = st.session_state.get(f"chat_input_{int(time.time())}", "")
                        st.session_state[f"chat_input_{int(time.time())}"] = current_msg + emoji
                        st.session_state.show_emoji_picker = False
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def _handle_send(content: str, uploaded_file):
        """Handle send action"""
        user = EnhancedChatManager.get_or_create_user()
        room = st.session_state.get('current_room', 'general')
        
        file_path = None
        thumbnail_path = None
        file_size = None
        file_type = None
        display_content = content.strip() if content else ""
        
        # Process file if uploaded
        if uploaded_file:
            saved_path, thumb_path, ftype = EnhancedFileStorageManager.save_file(uploaded_file, user.id)
            
            if saved_path:
                file_path = saved_path
                thumbnail_path = thumb_path
                file_size = saved_path.stat().st_size
                file_type = ftype
                
                # Set default content if no text
                if not display_content:
                    display_content = uploaded_file.name
        
        # Determine message type
        if file_path:
            if file_type == 'image':
                msg_type = MessageType.IMAGE
            elif file_type == 'video':
                msg_type = MessageType.VIDEO
            elif file_type == 'audio':
                msg_type = MessageType.AUDIO
            else:
                msg_type = MessageType.FILE
        else:
            msg_type = MessageType.TEXT
        
        # Save message
        if display_content or file_path:
            EnhancedChatManager.save_message(
                content=display_content,
                msg_type=msg_type,
                file_path=file_path,
                file_size=file_size,
                file_type=file_type,
                thumbnail_path=thumbnail_path,
                room=room
            )
            
            # Clear input
            st.session_state.pop(f"chat_input_{int(time.time())}", None)
            st.session_state.pop(f"file_upload_{int(time.time())}", None)


class EnhancedFileGalleryUI:
    """Enhanced file gallery with search and filtering"""
    
    @staticmethod
    def render_gallery():
        """Render file gallery with advanced features"""
        st.header("📁 File Gallery")
        
        user = EnhancedChatManager.get_or_create_user()
        
        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search files", placeholder="Filename or type...")
        with col2:
            file_type_filter = st.selectbox(
                "Type",
                ["All", "Image", "Video", "Audio", "Document", "Archive"]
            )
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Newest", "Oldest", "Name", "Size"]
            )
        
        # Get user's files
        files = EnhancedFileStorageManager.get_session_files(user.id)
        
        if not files:
            st.info("""
            📭 No files shared yet.
            
            **To share files:**
            1. Go to the chat room
            2. Use the upload button or drag & drop
            3. Files will appear here automatically
            """)
            return
        
        # Apply filters
        filtered_files = files
        
        if search_query:
            filtered_files = [
                f for f in filtered_files
                if search_query.lower() in f['original_name'].lower()
            ]
        
        if file_type_filter != "All":
            filtered_files = [
                f for f in filtered_files
                if f.get('type', '').lower() == file_type_filter.lower()
            ]
        
        # Apply sorting
        if sort_by == "Newest":
            filtered_files.sort(key=lambda x: x['uploaded_at'], reverse=True)
        elif sort_by == "Oldest":
            filtered_files.sort(key=lambda x: x['uploaded_at'])
        elif sort_by == "Name":
            filtered_files.sort(key=lambda x: x['original_name'].lower())
        elif sort_by == "Size":
            filtered_files.sort(key=lambda x: x.get('size', 0), reverse=True)
        
        # Display files
        if not filtered_files:
            st.warning("No files match your filters.")
            return
        
        # Grid layout
        cols = st.columns(4)
        
        for idx, file_info in enumerate(filtered_files[:20]):  # Limit for performance
            with cols[idx % 4]:
                EnhancedFileGalleryUI._render_file_card(file_info, idx)
        
        # Show count
        st.caption(f"Showing {len(filtered_files[:20])} of {len(filtered_files)} files")
    
    @staticmethod
    def _render_file_card(file_info: Dict, idx: int):
        """Render file card in gallery"""
        file_path = EnhancedConfig.BASE_DIR / file_info['path']
        original_name = file_info['original_name']
        file_type = file_info.get('type', 'file')
        file_size = file_info.get('size', 0)
        
        # Format size
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        # Card container
        with st.container():
            st.markdown("""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                border: 1px solid #e5e7eb;
                height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
            """, unsafe_allow_html=True)
            
            # File icon based on type
            icons = {
                'image': '🖼️',
                'video': '🎬',
                'audio': '🎵',
                'document': '📄',
                'archive': '📦',
                'file': '📎'
            }
            
            icon = icons.get(file_type, '📎')
            
            # Preview for images
            if file_type == 'image' and file_path.exists():
                thumbnail_path = file_info.get('thumbnail')
                if thumbnail_path:
                    abs_thumb_path = EnhancedConfig.BASE_DIR / thumbnail_path
                    if abs_thumb_path.exists():
                        st.image(str(abs_thumb_path), use_column_width=True)
                else:
                    st.markdown(f"<div style='text-align: center; font-size: 3rem;'>{icon}</div>", 
                              unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: center; font-size: 3rem;'>{icon}</div>", 
                          unsafe_allow_html=True)
            
            # File info
            st.markdown(f"""
            <div style="
                text-align: center;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            ">
                <strong>{original_name[:15]}{'...' if len(original_name) > 15 else ''}</strong>
                <div style="color: #6b7280; font-size: 0.8rem; margin-top: 0.2rem;">
                    {size_str} • {file_type.title()}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    st.download_button(
                        label="Download",
                        data=f,
                        file_name=original_name,
                        mime="application/octet-stream",
                        use_container_width=True,
                        key=f"gallery_dl_{idx}"
                    )
            else:
                st.error("File unavailable")
            
            st.markdown("</div>", unsafe_allow_html=True)


class StatsDashboardUI:
    """Statistics and monitoring dashboard"""
    
    @staticmethod
    def render_dashboard():
        """Render statistics dashboard"""
        st.header("📊 System Dashboard")
        
        # Get stats
        stats = EnhancedConfig._read_stats()
        metadata = EnhancedConfig._read_metadata()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Users",
                len(metadata.get('active_sessions', {})),
                f"Peak: {stats.get('peak_concurrent', 0)}"
            )
        
        with col2:
            st.metric(
                "Total Messages",
                metadata.get('total_messages', 0),
                f"{stats.get('messages_sent', 0)} today"
            )
        
        with col3:
            st.metric(
                "Total Files",
                metadata.get('total_files', 0),
                f"{stats.get('files_uploaded', 0)} today"
            )
        
        with col4:
            uptime = stats.get('uptime_seconds', 0)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            st.metric(
                "Uptime",
                f"{hours}h {minutes}m"
            )
        
        # Charts and visualizations
        st.subheader("📈 Activity Over Time")
        
        # File type distribution
        st.subheader("📦 File Type Distribution")
        
        # Cache performance
        st.subheader("⚡ Performance")
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            hit_rate = message_cache.hit_rate() * 100
            st.metric(
                "Cache Hit Rate",
                f"{hit_rate:.1f}%",
                delta=f"{message_cache.hits:,} hits"
            )
        
        with col_perf2:
            st.metric(
                "Memory Usage",
                f"{len(message_cache.cache)} items",
                delta=f"{len(file_cache.cache)} files cached"
            )
        
        # System info
        st.subheader("🖥️ System Information")
        
        sys_col1, sys_col2 = st.columns(2)
        
        with sys_col1:
            st.markdown("""
            **Storage Path:**  
            `{path}`  
            
            **Session ID:**  
            `{session_id}`  
            
            **App Version:**  
            {version}
            """.format(
                path=EnhancedConfig.BASE_DIR,
                session_id=EnhancedChatManager.get_session_id(),
                version=st.session_state.get('app_version', '2.0.0')
            ))
        
        with sys_col2:
            # Calculate storage usage
            total_size = 0
            for dir_path in [EnhancedConfig.CHAT_DIR, EnhancedConfig.TEMP_UPLOADS, 
                           EnhancedConfig.THUMBNAILS, EnhancedConfig.AVATARS]:
                if dir_path.exists():
                    total_size += sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            
            st.metric(
                "Storage Used",
                f"{total_size / (1024 * 1024):.1f} MB"
            )
            
            st.info("💾 All data is temporary and cleared on server reboot")


# ======================
# MAIN APPLICATION
# ======================
def main():
    """Main application with enhanced features"""
    # Page config
    st.set_page_config(
        page_title="TempChat Pro - Real-time Temporary Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/tempchat',
            'Report a bug': 'https://github.com/yourusername/tempchat/issues',
            'About': """
            # TempChat Pro v2.0
            
            **Temporary, secure chat with file sharing**
            
            All data is automatically cleared on server reboot.
            No persistence, no tracking, just communication.
            """
        }
    )
    
    # Apply custom CSS
    EnhancedChatUI.render_custom_css()
    
    # Initialize session state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'chat'
    if 'current_room' not in st.session_state:
        st.session_state.current_room = 'general'
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    # Sidebar
    with st.sidebar:
        EnhancedChatUI.render_user_profile()
        EnhancedChatUI.render_room_selector()
        
        st.markdown("### 🧭 Navigation")
        
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            if st.button("💬", 
                        use_container_width=True, 
                        help="Chat Room",
                        type="primary" if st.session_state.current_view == 'chat' else "secondary"):
                st.session_state.current_view = 'chat'
                st.rerun()
        
        with nav_col2:
            if st.button("📁", 
                        use_container_width=True, 
                        help="File Gallery",
                        type="primary" if st.session_state.current_view == 'gallery' else "secondary"):
                st.session_state.current_view = 'gallery'
                st.rerun()
        
        with nav_col3:
            if st.button("📊", 
                        use_container_width=True, 
                        help="Dashboard",
                        type="primary" if st.session_state.current_view == 'dashboard' else "secondary"):
                st.session_state.current_view = 'dashboard'
                st.rerun()
        
        st.divider()
        EnhancedChatUI.render_active_users()
        
        st.divider()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Session", use_container_width=True):
            EnhancedChatManager.update_user_status(UserStatus.ONLINE)
            st.session_state.force_refresh = True
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            chat_file = EnhancedChatManager._get_chat_file(st.session_state.current_room)
            if chat_file.exists():
                chat_file.unlink()
                message_cache.clear()
                st.session_state.system_messages.append("Chat history cleared")
                st.rerun()
        
        if st.button("🧹 System Cleanup", use_container_width=True):
            EnhancedConfig._perform_startup_cleanup()
            message_cache.clear()
            file_cache.clear()
            st.session_state.system_messages.append("System cleanup completed")
            st.rerun()
        
        st.divider()
        
        # Status selector
        st.markdown("### 🟢 Status")
        current_user = EnhancedChatManager.get_or_create_user()
        status_options = {
            "🟢 Online": UserStatus.ONLINE,
            "🌙 Away": UserStatus.AWAY,
            "⚫ Offline": UserStatus.OFFLINE
        }
        
        selected_status = st.radio(
            "Set your status",
            list(status_options.keys()),
            index=list(status_options.values()).index(current_user.status),
            label_visibility="collapsed"
        )
        
        if status_options[selected_status] != current_user.status:
            EnhancedChatManager.update_user_status(status_options[selected_status])
            st.rerun()
        
        st.divider()
        
        # Info panel
        st.markdown("""
        ### ℹ️ About TempChat Pro
        
        **Features:**
        • Real-time chat with auto-refresh
        • File sharing (images, videos, documents)
        • Multiple chat rooms
        • Message reactions & editing
        • User status indicators
        • System statistics dashboard
        
        **Security:**
        • All data is temporary
        • Auto-deletes on server reboot
        • Session isolation
        • No external storage
        
        **Limits:**
        • Max file size: {max_size}MB
        • Session timeout: {timeout}min
        • Auto-cleanup every 24h
        
        Made with ❤️ using Streamlit
        """.format(
            max_size=EnhancedConfig.MAX_FILE_SIZE_MB,
            timeout=EnhancedConfig.SESSION_TIMEOUT_MIN
        ))
    
    # Main content area
    EnhancedChatUI.render_header()
    
    # Show system messages
    if st.session_state.system_messages:
        for sys_msg in st.session_state.system_messages[-3:]:  # Show last 3
            st.markdown(f"""
            <div class="system-message">
                ⚙️ {sys_msg}
            </div>
            """, unsafe_allow_html=True)
        
        # Clear old messages
        if len(st.session_state.system_messages) > 3:
            st.session_state.system_messages = st.session_state.system_messages[-3:]
    
    # Render current view
    if st.session_state.current_view == 'chat':
        # Auto-refresh mechanism
        refresh_container = st.empty()
        
        with refresh_container.container():
            EnhancedChatUI.render_chat_messages()
        
        # Input area (always visible)
        EnhancedChatUI.render_input_area()
        
        # Auto-refresh logic
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        refresh_interval = EnhancedConfig.AUTO_REFRESH_SEC
        
        # Faster refresh if messages are coming in
        messages = EnhancedChatManager.load_messages(st.session_state.current_room)
        if messages and len(messages) > 10:
            last_msg_time = datetime.fromisoformat(messages[-1]['timestamp'])
            if (datetime.now() - last_msg_time).total_seconds() < 30:
                refresh_interval = 1  # Faster refresh during active chat
        
        if (current_time - st.session_state.last_refresh > refresh_interval or
            st.session_state.get('force_refresh', False)):
            st.session_state.last_refresh = current_time
            st.session_state.force_refresh = False
            st.rerun()
    
    elif st.session_state.current_view == 'gallery':
        EnhancedFileGalleryUI.render_gallery()
    
    elif st.session_state.current_view == 'dashboard':
        StatsDashboardUI.render_dashboard()
    
    # Footer
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown(f"""
        <div style="color: #6b7280; font-size: 0.8rem;">
            <strong>Session:</strong> {EnhancedChatManager.get_session_id()[:12]}<br>
            <strong>Room:</strong> #{st.session_state.get('current_room', 'general')}
        </div>
        """, unsafe_allow_html=True)
    
    with footer_col2:
        metadata = EnhancedConfig._read_metadata()
        active_sessions = len(metadata.get('active_sessions', {}))
        total_messages = metadata.get('total_messages', 0)
        
        st.markdown(f"""
        <div style="color: #6b7280; font-size: 0.8rem; text-align: center;">
            👥 {active_sessions} active • ✉️ {total_messages} messages
        </div>
        """, unsafe_allow_html=True)
    
    with footer_col3:
        current_time = datetime.now()
        st.markdown(f"""
        <div style="color: #6b7280; font-size: 0.8rem; text-align: right;">
            ⏰ {current_time.strftime('%I:%M %p')}<br>
            📅 {current_time.strftime('%b %d, %Y')}
        </div>
        """, unsafe_allow_html=True)
    
    # Performance monitoring
    if st.session_state.get('debug_mode', False):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🐛 Debug Info")
        st.sidebar.metric("Cache Size", len(message_cache.cache))
        st.sidebar.metric("Cache Hit Rate", f"{message_cache.hit_rate()*100:.1f}%")
        st.sidebar.metric("File Cache", len(file_cache.cache))

# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    try:
        main()
        
        # Cleanup on exit
        import atexit
        
        @atexit.register
        def cleanup_on_exit():
            """Cleanup function called on exit"""
            try:
                # Mark user as offline
                EnhancedChatManager.update_user_status(UserStatus.OFFLINE)
                
                # Update session metadata
                metadata = EnhancedConfig._read_metadata()
                session_id = EnhancedChatManager.get_session_id()
                if session_id in metadata['active_sessions']:
                    metadata['active_sessions'][session_id]['ended_at'] = datetime.now().isoformat()
                    EnhancedConfig._atomic_write(EnhancedConfig.METADATA_FILE, metadata)
            except:
                pass
        
    except Exception as e:
        st.error(f"⚠️ Application Error: {str(e)}")
        st.error("Please refresh the page or restart the application.")
        
        # Log error
        try:
            error_log = EnhancedConfig.BASE_DIR / "error.log"
            with open(error_log, 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {str(e)}\n")
        except:
            pass
        
        if st.button("🔄 Restart Application"):
            st.session_state.clear()
            st.rerun()
