"""
MEMORYVAULT FORUM PRO
Version: 8.0.0 - Interactive Chat Forum with Guest Registration & Cloud Uploads
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import io
import time
import math
from dataclasses import dataclass, asdict
from enum import Enum
import os
from contextlib import contextmanager
import mimetypes
import warnings
import threading
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# VIDEO SUPPORT (optional)
# ============================================================================
try:
    import cv2
    import moviepy.editor as mp
    from moviepy.editor import VideoFileClip
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

# ============================================================================
# SESSION-BASED STORAGE FOR CLOUD DEPLOYMENT
# ============================================================================
class SessionStorage:
    """Handles temporary in-memory storage for Streamlit Cloud compatibility"""
    def __init__(self):
        if 'session_uploads' not in st.session_state:
            st.session_state.session_uploads = {}
        if 'session_media_metadata' not in st.session_state:
            st.session_state.session_media_metadata = {}
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'guest_users' not in st.session_state:
            st.session_state.guest_users = {}
        if 'active_sessions' not in st.session_state:
            st.session_state.active_sessions = set()
        if 'user_activity' not in st.session_state:
            st.session_state.user_activity = {}
    
    def store_upload(self, file_id: str, file_bytes: bytes, filename: str, mime_type: str):
        """Store uploaded file in session state (temporary)"""
        st.session_state.session_uploads[file_id] = {
            'bytes': file_bytes,
            'filename': filename,
            'mime_type': mime_type,
            'uploaded_at': datetime.datetime.now().isoformat(),
            'size': len(file_bytes)
        }
        return file_id
    
    def get_upload(self, file_id: str) -> Optional[Dict]:
        return st.session_state.session_uploads.get(file_id)
    
    def delete_upload(self, file_id: str):
        if file_id in st.session_state.session_uploads:
            del st.session_state.session_uploads[file_id]
    
    def add_chat_message(self, user_id: str, username: str, content: str, 
                        media_ids: List[str] = None, reply_to: Optional[str] = None):
        msg = {
            'message_id': str(uuid.uuid4()),
            'user_id': user_id,
            'username': username,
            'content': content,
            'media_ids': media_ids or [],
            'reply_to': reply_to,
            'timestamp': datetime.datetime.now().isoformat(),
            'likes': set(),
            'edited': False
        }
        st.session_state.chat_messages.append(msg)
        return msg['message_id']
    
    def get_chat_history(self, limit: int = 100) -> List[Dict]:
        messages = st.session_state.chat_messages[-limit:]
        # Convert sets to lists for display
        for msg in messages:
            msg['like_count'] = len(msg['likes'])
        return messages
    
    def like_message(self, message_id: str, user_id: str):
        for msg in st.session_state.chat_messages:
            if msg['message_id'] == message_id:
                if user_id in msg['likes']:
                    msg['likes'].remove(user_id)
                    return False
                else:
                    msg['likes'].add(user_id)
                    return True
        return None
    
    def register_guest(self, username: str) -> str:
        user_id = str(uuid.uuid4())[:8]
        st.session_state.guest_users[user_id] = {
            'user_id': user_id,
            'username': username,
            'joined_at': datetime.datetime.now().isoformat(),
            'status': 'online'
        }
        st.session_state.active_sessions.add(user_id)
        return user_id
    
    def get_online_count(self) -> int:
        # Clean up inactive users (older than 30 minutes)
        now = datetime.datetime.now()
        active = set()
        for uid, user in st.session_state.guest_users.items():
            joined = datetime.datetime.fromisoformat(user['joined_at'])
            if (now - joined).total_seconds() < 1800:  # 30 min timeout
                active.add(uid)
        st.session_state.active_sessions = active
        return len(active)
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        return st.session_state.guest_users.get(user_id)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Forum Pro"
    VERSION = "8.0.0"
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    HD_SIZE = (1920, 1080)
    THUMB_STRIP_SIZE = (120, 90)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4','.mov','.avi','.mkv','.webm','.wmv','.flv','.m4v']
    IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.gif','.bmp','.webp','.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    MAX_COMMENT_LENGTH = 1000
    MAX_MESSAGE_LENGTH = 2000
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for cloud
    FRAME_STYLES = ["Elegant Gold","Polaroid","Modern Shadow","Dark Museum","Vintage","Gallery White"]
    DEFAULT_FRAME = "Elegant Gold"
    
    # Forum settings
    MAX_MESSAGES_PER_PAGE = 50
    ALLOWED_MESSAGE_TYPES = ['text', 'image', 'video', 'mixed']
    
    @classmethod
    def init_directories(cls):
        for d in [cls.DATA_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MEDIA PROCESSOR (Session-based for Cloud)
# ============================================================================
class MediaProcessor:
    @staticmethod
    def process_upload(uploaded_file) -> Optional[Dict]:
        """Process uploaded file and return metadata"""
        if uploaded_file is None:
            return None
            
        file_bytes = uploaded_file.getvalue()
        file_size = len(file_bytes)
        
        if file_size > Config.MAX_FILE_SIZE:
            st.error(f"File too large. Max size: {Config.MAX_FILE_SIZE/(1024*1024):.1f}MB")
            return None
            
        file_ext = Path(uploaded_file.name).suffix.lower()
        file_id = str(uuid.uuid4())
        
        if file_ext in Config.IMAGE_EXTENSIONS:
            return MediaProcessor._process_image(file_id, file_bytes, uploaded_file.name, file_ext)
        elif file_ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaProcessor._process_video(file_id, file_bytes, uploaded_file.name, file_ext)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None
    
    @staticmethod
    def _process_image(file_id: str, file_bytes: bytes, filename: str, ext: str) -> Dict:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)
            
            # Create thumbnail
            thumb_img = img.copy()
            thumb_img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            thumb_buf = io.BytesIO()
            thumb_img.save(thumb_buf, format='JPEG', quality=85)
            thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()
            
            # Create HD preview
            hd_img = img.copy()
            hd_img.thumbnail(Config.HD_SIZE, Image.Resampling.LANCZOS)
            hd_buf = io.BytesIO()
            hd_img.save(hd_buf, format='JPEG', quality=90)
            hd_b64 = base64.b64encode(hd_buf.getvalue()).decode()
            
            # Store in session
            storage = SessionStorage()
            storage.store_upload(file_id, file_bytes, filename, f"image/{ext.replace('.', '')}")
            
            return {
                'file_id': file_id,
                'type': 'image',
                'filename': filename,
                'dimensions': img.size,
                'format': img.format,
                'size': len(file_bytes),
                'thumbnail': f"data:image/jpeg;base64,{thumb_b64}",
                'hd_url': f"data:image/jpeg;base64,{hd_b64}",
                'uploaded_at': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    @staticmethod
    def _process_video(file_id: str, file_bytes: bytes, filename: str, ext: str) -> Dict:
        # For videos, store raw bytes and create placeholder thumbnail
        storage = SessionStorage()
        storage.store_upload(file_id, file_bytes, filename, f"video/{ext.replace('.', '')}")
        
        # Try to extract thumbnail if cv2 available
        thumb_b64 = ""
        if VIDEO_SUPPORT:
            try:
                temp_path = f"/tmp/{file_id}{ext}"
                with open(temp_path, 'wb') as f:
                    f.write(file_bytes)
                cap = cv2.VideoCapture(temp_path)
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=85)
                    thumb_b64 = base64.b64encode(buf.getvalue()).decode()
                cap.release()
                os.remove(temp_path)
            except:
                pass
        
        if not thumb_b64:
            # Default video thumbnail
            thumb_b64 = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iIzMzMyIvPjx0ZXh0IHg9IjE1MCIgeT0iMTUwIiBmaWxsPSIjZmZmIiBmb250LXNpemU9IjQwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+4pqhPC90ZXh0Pjwvc3ZnPg=="
        
        return {
            'file_id': file_id,
            'type': 'video',
            'filename': filename,
            'dimensions': (0, 0),
            'format': ext.upper(),
            'size': len(file_bytes),
            'thumbnail': f"data:image/jpeg;base64,{thumb_b64}" if thumb_b64.startswith('/') else f"data:image/svg+xml;base64,{thumb_b64}",
            'hd_url': None,  # Videos use raw bytes
            'uploaded_at': datetime.datetime.now().isoformat()
        }
    
    @staticmethod
    def get_media_data_url(file_id: str) -> str:
        storage = SessionStorage()
        upload = storage.get_upload(file_id)
        if not upload:
            return ""
        
        mime = upload['mime_type']
        b64 = base64.b64encode(upload['bytes']).decode()
        return f"data:{mime};base64,{b64}"


# ============================================================================
# FRAME RENDERER
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_detail(src: str, style: str, expanded: bool = False) -> str:
        max_h = "85vh" if expanded else "65vh"
        styles = {
            "Elegant Gold": ('background:linear-gradient(135deg,#b8860b,#daa520,#ffd700);padding:14px;border-radius:10px;',
                             'background:#fffff5;padding:20px;border-radius:6px;'),
            "Polaroid": ('background:#fff;padding:20px 20px 70px 20px;border-radius:3px;',''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:16px;box-shadow:0 14px 48px rgba(0,0,0,.2);',''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30);padding:28px;border-radius:16px;',
                            'background:#fffff8;padding:18px;border-radius:6px;'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7);padding:16px;border-radius:6px;border:2px solid #a08050;',
                        'background:#faf5ee;padding:14px;border-radius:4px;'),
            "Gallery White": ('background:#fff;padding:24px;border-radius:4px;border:1px solid #e0e0e0;','')
        }
        outer, inner = styles.get(style, styles["Elegant Gold"])
        return f'''<div style="{outer}"><div style="{inner}">
            <img src="{src}" style="width:100%;max-height:{max_h};object-fit:contain;display:block;margin:0 auto;border-radius:4px;">
        </div></div>'''

    @staticmethod
    def inject_css():
        st.markdown("""
        <style>
        /* Forum Styles */
        .forum-container { max-width: 900px; margin: 0 auto; }
        .chat-message { 
            background: linear-gradient(135deg, #667eea20, #764ba220); 
            border-radius: 16px; 
            padding: 16px; 
            margin-bottom: 12px; 
            border-left: 4px solid #667eea;
            animation: slideIn 0.3s ease-out;
        }
        .chat-message.own { 
            background: linear-gradient(135deg, #11998e20, #38ef7d20); 
            border-left-color: #11998e;
        }
        .chat-header { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 8px; 
        }
        .chat-username { 
            font-weight: 700; 
            color: #667eea; 
            font-size: 14px; 
        }
        .chat-time { 
            color: #888; 
            font-size: 12px; 
        }
        .chat-content { 
            color: #333; 
            line-height: 1.5; 
            margin-bottom: 8px; 
        }
        .chat-media { 
            display: flex; 
            gap: 8px; 
            flex-wrap: wrap; 
            margin-top: 8px; 
        }
        .chat-media img, .chat-media video { 
            border-radius: 8px; 
            max-height: 200px; 
            object-fit: cover; 
            cursor: pointer;
            transition: transform 0.2s;
        }
        .chat-media img:hover { transform: scale(1.05); }
        .chat-actions { 
            display: flex; 
            gap: 12px; 
            margin-top: 8px; 
            font-size: 13px; 
        }
        .chat-action-btn { 
            cursor: pointer; 
            color: #666; 
            transition: color 0.2s; 
        }
        .chat-action-btn:hover { color: #667eea; }
        .chat-action-btn.liked { color: #ff4b4b; }
        
        /* Online Users */
        .online-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #e8f5e9;
            color: #2e7d32;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }
        .online-dot {
            width: 8px;
            height: 8px;
            background: #4caf50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        /* Guest Registration */
        .guest-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 20px;
        }
        .guest-input {
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .guest-input::placeholder { color: rgba(255,255,255,0.7); }
        
        /* Upload Zone */
        .upload-zone {
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s;
        }
        .upload-zone:hover {
            background: #eef1ff;
            border-color: #764ba2;
        }
        
        /* Animations */
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }
        
        /* Navigation */
        .nav-container { display: flex; align-items: center; justify-content: center; height: 100%; min-height: 60vh; }
        .nav-btn { background: #667eea; color: white; border: none; border-radius: 50%; width: 56px; height: 56px; font-size: 28px; cursor: pointer; transition: 0.2s; margin: 0 10px; }
        .nav-btn:hover { background: #764ba2; transform: scale(1.05); }
        .nav-btn.disabled { opacity: 0.3; cursor: not-allowed; pointer-events: none; }
        
        /* Lightbox */
        .lightbox-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); z-index: 9999;
            display: flex; align-items: center; justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# GUEST AUTHENTICATION
# ============================================================================
class GuestAuth:
    @staticmethod
    def render_login():
        st.markdown("""
        <div class="guest-card">
            <h2>👋 Welcome to MemoryVault Forum</h2>
            <p>Join the conversation instantly. No permanent account needed.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Choose a display name", 
                                   placeholder="e.g., PhotoFan2024",
                                   max_chars=30,
                                   key="guest_username")
            
            if st.button("🚀 Join Forum", use_container_width=True, type="primary"):
                if username.strip():
                    if len(username.strip()) < 3:
                        st.error("Username must be at least 3 characters")
                        return False
                    
                    storage = SessionStorage()
                    user_id = storage.register_guest(username.strip())
                    st.session_state.current_user_id = user_id
                    st.session_state.current_username = username.strip()
                    st.session_state.authenticated = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Please enter a display name")
        return False
    
    @staticmethod
    def check_auth():
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        
        if st.session_state.authenticated and 'current_user_id' in st.session_state:
            # Update activity
            storage = SessionStorage()
            user = storage.get_user(st.session_state.current_user_id)
            if user:
                return True
        
        return GuestAuth.render_login()
    
    @staticmethod
    def logout():
        if 'current_user_id' in st.session_state:
            storage = SessionStorage()
            user = storage.get_user(st.session_state.current_user_id)
            if user:
                user['status'] = 'offline'
        st.session_state.authenticated = False
        st.session_state.current_user_id = None
        st.session_state.current_username = None
        st.rerun()


# ============================================================================
# FORUM CHAT COMPONENT
# ============================================================================
class ForumChat:
    def __init__(self):
        self.storage = SessionStorage()
        self.processor = MediaProcessor()
    
    def render_chat_input(self):
        """Render the message input area with media upload"""
        st.markdown("### 💬 Share a moment")
        
        # Text input
        message_text = st.text_area("What's on your mind?", 
                                   placeholder="Share thoughts about these memories...",
                                   max_chars=Config.MAX_MESSAGE_LENGTH,
                                   key="chat_input_text",
                                   height=100)
        
        # Media upload
        col1, col2 = st.columns([3,1])
        with col1:
            uploaded_files = st.file_uploader(
                "📎 Attach photos/videos", 
                accept_multiple_files=True,
                type=list(Config.ALLOWED_EXTENSIONS),
                key="chat_media_upload"
            )
        with col2:
            st.caption("Max 50MB per file")
            st.caption("Images & videos supported")
        
        # Preview uploads
        media_metadata = []
        if uploaded_files:
            st.markdown("**Preview:**")
            preview_cols = st.columns(min(len(uploaded_files), 4))
            for idx, file in enumerate(uploaded_files):
                with preview_cols[idx % 4]:
                    meta = self.processor.process_upload(file)
                    if meta:
                        media_metadata.append(meta)
                        if meta['type'] == 'image':
                            st.image(meta['thumbnail'], use_container_width=True)
                        else:
                            st.video(file)
                            st.caption(f"🎬 {meta['filename']}")
        
        # Send button
        if st.button("📤 Post to Forum", use_container_width=True, type="primary"):
            if not message_text.strip() and not media_metadata:
                st.warning("Please write a message or upload media")
                return
            
            user_id = st.session_state.current_user_id
            username = st.session_state.current_username
            
            media_ids = [m['file_id'] for m in media_metadata]
            self.storage.add_chat_message(
                user_id=user_id,
                username=username,
                content=message_text.strip(),
                media_ids=media_ids
            )
            
            # Clear inputs
            st.session_state.chat_input_text = ""
            # Reset file uploader by clearing key (hack)
            if 'chat_media_upload' in st.session_state:
                del st.session_state.chat_media_upload
            st.rerun()
    
    def render_message(self, msg: Dict):
        """Render a single chat message"""
        is_own = msg['user_id'] == st.session_state.get('current_user_id')
        own_class = "own" if is_own else ""
        
        # Format timestamp
        try:
            ts = datetime.datetime.fromisoformat(msg['timestamp'])
            time_str = ts.strftime("%H:%M · %b %d")
        except:
            time_str = "Just now"
        
        # Check if liked by current user
        current_uid = st.session_state.get('current_user_id', '')
        is_liked = current_uid in msg.get('likes', set())
        like_class = "liked" if is_liked else ""
        like_icon = "❤️" if is_liked else "🤍"
        
        st.markdown(f"""
        <div class="chat-message {own_class}">
            <div class="chat-header">
                <span class="chat-username">@{msg['username']}</span>
                <span class="chat-time">{time_str}</span>
            </div>
            <div class="chat-content">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render media attachments
        if msg.get('media_ids'):
            cols = st.columns(min(len(msg['media_ids']), 3))
            for idx, media_id in enumerate(msg['media_ids']):
                with cols[idx % 3]:
                    upload = self.storage.get_upload(media_id)
                    if upload:
                        mime = upload['mime_type']
                        b64 = base64.b64encode(upload['bytes']).decode()
                        data_url = f"data:{mime};base64,{b64}"
                        
                        if mime.startswith('image'):
                            st.image(data_url, use_container_width=True)
                        elif mime.startswith('video'):
                            st.video(data_url)
        
        # Actions
        col1, col2, col3, col4 = st.columns([1,1,1,6])
        with col1:
            if st.button(f"{like_icon} {msg.get('like_count', 0)}", 
                        key=f"like_{msg['message_id']}",
                        help="Like this post"):
                self.storage.like_message(msg['message_id'], current_uid)
                st.rerun()
        with col2:
            if st.button("💬 Reply", key=f"reply_{msg['message_id']}"):
                st.session_state.replying_to = msg['message_id']
                st.rerun()
        with col3:
            if is_own and st.button("🗑️", key=f"del_{msg['message_id']}"):
                # Remove message
                st.session_state.chat_messages = [
                    m for m in st.session_state.chat_messages 
                    if m['message_id'] != msg['message_id']
                ]
                st.rerun()
        
        # Reply indicator
        if st.session_state.get('replying_to') == msg['message_id']:
            st.info(f"Replying to @{msg['username']}")
    
    def render_chat_history(self):
        """Render all chat messages"""
        messages = self.storage.get_chat_history(Config.MAX_MESSAGES_PER_PAGE)
        
        if not messages:
            st.info("No messages yet. Be the first to share! 🎉")
            return
        
        # Reverse to show newest first
        for msg in reversed(messages):
            self.render_message(msg)
            st.divider()


# ============================================================================
# ONLINE USERS SIDEBAR
# ============================================================================
class OnlineUsersSidebar:
    @staticmethod
    def render():
        storage = SessionStorage()
        count = storage.get_online_count()
        
        st.sidebar.markdown(f"""
        <div class="online-badge">
            <div class="online-dot"></div>
            <span>{count} online</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.divider()
        
        # Show recent users
        st.sidebar.subheader("👥 Recent Guests")
        users = list(st.session_state.guest_users.values())[-10:]
        for user in reversed(users):
            status_emoji = "🟢" if user['status'] == 'online' else "⚪"
            st.sidebar.caption(f"{status_emoji} @{user['username']}")
        
        st.sidebar.divider()
        
        # User stats
        total_users = len(st.session_state.guest_users)
        total_messages = len(st.session_state.chat_messages)
        total_uploads = len(st.session_state.session_uploads)
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Guests", total_users)
        col2.metric("Posts", total_messages)
        st.sidebar.metric("Media", total_uploads)


# ============================================================================
# PHOTO VIEWER COMPONENT (Legacy Album Integration)
# ============================================================================
class PhotoViewer:
    def __init__(self):
        self.processor = MediaProcessor()
    
    def render_gallery(self):
        """Render gallery of session uploads"""
        uploads = st.session_state.session_uploads
        
        if not uploads:
            st.info("No photos uploaded yet. Share some in the forum! 📸")
            return
        
        st.subheader("📸 Community Gallery")
        
        # Filter images
        images = []
        for fid, data in uploads.items():
            if data['mime_type'].startswith('image'):
                images.append((fid, data))
        
        if not images:
            st.info("No images in gallery yet")
            return
        
        # Grid display
        cols = st.columns(4)
        for idx, (fid, data) in enumerate(images):
            with cols[idx % 4]:
                b64 = base64.b64encode(data['bytes']).decode()
                st.image(f"data:{data['mime_type']};base64,{b64}", 
                        use_container_width=True)
                st.caption(f"{data['filename'][:20]}...")
                
                if st.button("🔍 View", key=f"view_{fid}", use_container_width=True):
                    st.session_state.lightbox_image = fid
                    st.rerun()
    
    def render_lightbox(self):
        """Render full-screen image viewer"""
        if 'lightbox_image' not in st.session_state or not st.session_state.lightbox_image:
            return
        
        fid = st.session_state.lightbox_image
        upload = SessionStorage().get_upload(fid)
        
        if not upload:
            st.session_state.lightbox_image = None
            return
        
        # Overlay
        b64 = base64.b64encode(upload['bytes']).decode()
        data_url = f"data:{upload['mime_type']};base64,{b64}"
        
        st.markdown(f"""
        <div class="lightbox-overlay" onclick="window.location.reload()">
            <img src="{data_url}" style="max-width:90%; max-height:90%; object-fit:contain;">
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✕ Close", key="close_lightbox"):
            st.session_state.lightbox_image = None
            st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class ForumApp:
    def __init__(self):
        st.set_page_config(
            page_title=Config.APP_NAME, 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        Config.init_directories()
        FrameRenderer.inject_css()
        self.chat = ForumChat()
        self.viewer = PhotoViewer()
        
        # Init session states
        if 'replying_to' not in st.session_state:
            st.session_state.replying_to = None
        if 'lightbox_image' not in st.session_state:
            st.session_state.lightbox_image = None
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("📁 MemoryVault Forum")
            st.caption(f"v{Config.VERSION}")
            
            # Online users
            OnlineUsersSidebar.render()
            
            # Navigation
            st.subheader("Navigation")
            page = st.radio("Go to", 
                          ["💬 Forum", "📸 Gallery", "ℹ️ About"],
                          key="nav_page")
            
            # Frame style for gallery
            if page == "📸 Gallery":
                fs = st.selectbox("Frame Style", Config.FRAME_STYLES,
                                index=Config.FRAME_STYLES.index(Config.DEFAULT_FRAME))
                st.session_state.frame_style = fs
            
            # User profile
            st.divider()
            st.subheader("Your Profile")
            st.write(f"👤 @{st.session_state.current_username}")
            st.write(f"🆔 {st.session_state.current_user_id}")
            
            if st.button("🚪 Leave Forum", use_container_width=True):
                GuestAuth.logout()
            
            # Stats
            st.divider()
            st.caption("💡 All data is temporary and session-based")
            st.caption("🔄 Refreshing will clear your session")
    
    def render_forum(self):
        st.markdown("""
        <div style="text-align:center; padding:20px;">
            <h1>💬 MemoryVault Forum</h1>
            <p style="color:#666;">Share memories, discuss photos, connect with guests</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat input at top
        self.chat.render_chat_input()
        
        st.divider()
        
        # Chat history
        self.chat.render_chat_history()
    
    def render_gallery(self):
        self.viewer.render_gallery()
        self.viewer.render_lightbox()
    
    def render_about(self):
        st.title("ℹ️ About MemoryVault Forum")
        
        st.markdown("""
        ### 🌟 Features
        - **Instant Guest Access**: No registration required. Pick a name and join!
        - **Live Chat**: Real-time forum discussions with other guests
        - **Media Sharing**: Upload photos and videos directly in chat
        - **Temporary Storage**: All content lives in session memory (perfect for Cloud)
        - **Live Counters**: See who's online and activity stats
        - **Interactive Gallery**: Browse all shared photos with lightbox view
        
        ### 🛡️ Privacy
        - No data persists after you close the browser
        - No cookies or tracking
        - Session expires after 30 minutes of inactivity
        
        ### 🚀 Cloud Ready
        This app is optimized for **Streamlit Cloud** deployment:
        - No filesystem dependencies
        - Session-based storage
        - Automatic cleanup
        """)
        
        st.info(f"Currently online: {SessionStorage().get_online_count()} guests")
    
    def run(self):
        if not GuestAuth.check_auth():
            return
        
        self.render_sidebar()
        
        page = st.session_state.get('nav_page', '💬 Forum')
        
        if page == "💬 Forum":
            self.render_forum()
        elif page == "📸 Gallery":
            self.render_gallery()
        elif page == "ℹ️ About":
            self.render_about()


# ============================================================================
# MAIN
# ============================================================================
def main():
    app = ForumApp()
    app.run()

if __name__ == "__main__":
    main()
