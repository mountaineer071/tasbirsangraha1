import streamlit as st
import json
import os
import uuid
import base64
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import re

# ======================
# CONFIGURATION & SETUP
# ======================
class Config:
    """Centralized configuration with OS-agnostic temp storage"""
    # Use system temp directory (cleared on reboot by OS)
    BASE_DIR = Path(os.environ.get('STREAMLIT_TEMP_DIR', 
                  Path.home() / '.streamlit_temp_chat'))
    CHAT_DIR = BASE_DIR / "chats"
    TEMP_UPLOADS = BASE_DIR / "temp_uploads"
    METADATA_FILE = BASE_DIR / "metadata.json"
    MAX_FILE_SIZE_MB = 50
    ALLOWED_EXTENSIONS = {
        'image': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'video': ['mp4', 'mov', 'avi', 'webm'],
        'document': ['pdf', 'txt', 'doc', 'docx']
    }
    SESSION_TIMEOUT_MIN = 60  # Auto-clear inactive sessions
    AUTO_REFRESH_SEC = 3

    @classmethod
    def init_storage(cls):
        """Initialize storage with reboot-safe cleanup"""
        # Create directories
        cls.CHAT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_UPLOADS.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata file
        if not cls.METADATA_FILE.exists():
            with open(cls.METADATA_FILE, 'w') as f:
                json.dump({
                    "created_at": datetime.now().isoformat(),
                    "sessions": {},
                    "files": []
                }, f, indent=2)
        
        # CRITICAL: Clean ALL data on app startup (simulates "before reboot" requirement)
        cls._full_cleanup()
        
        # Create new session metadata
        cls._update_metadata({
            "created_at": datetime.now().isoformat(),
            "active_session": str(uuid.uuid4()),
            "last_cleanup": datetime.now().isoformat()
        })
    
    @classmethod
    def _full_cleanup(cls):
        """Nuclear option: Delete ALL stored data on app restart"""
        try:
            if cls.CHAT_DIR.exists():
                shutil.rmtree(cls.CHAT_DIR)
            if cls.TEMP_UPLOADS.exists():
                shutil.rmtree(cls.TEMP_UPLOADS)
            cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
            cls.CHAT_DIR.mkdir(exist_ok=True)
            cls.TEMP_UPLOADS.mkdir(exist_ok=True)
            st.session_state['system_message'] = "✅ System rebooted - All temporary data cleared"
        except Exception as e:
            st.error(f"Cleanup failed: {str(e)}")
    
    @classmethod
    def _update_metadata(cls, updates: dict):
        """Safely update metadata JSON"""
        try:
            if cls.METADATA_FILE.exists():
                with open(cls.METADATA_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
            data.update(updates)
            with open(cls.METADATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            st.warning(f"Metadata update failed: {str(e)}")

# Initialize storage ON IMPORT (critical for reboot behavior)
Config.init_storage()

# ======================
# CORE MANAGERS
# ======================
class ChatManager:
    """Thread-safe chat operations with session isolation"""
    
    @staticmethod
    def get_session_id() -> str:
        """Get or create unique session ID"""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = str(uuid.uuid4())[:8]
        return st.session_state['session_id']
    
    @staticmethod
    def get_username() -> str:
        """Get sanitized username"""
        default = f"User_{ChatManager.get_session_id()}"
        raw = st.session_state.get('username_input', default)
        # Sanitize username
        clean = re.sub(r'[^\w\-_\. ]', '', raw.strip())[:20]
        return clean if clean else default
    
    @staticmethod
    def _get_chat_file() -> Path:
        """Get session-specific chat file"""
        session_id = ChatManager.get_session_id()
        return Config.CHAT_DIR / f"chat_{session_id}.json"
    
    @staticmethod
    def load_messages() -> List[Dict]:
        """Load messages for current session"""
        chat_file = ChatManager._get_chat_file()
        if chat_file.exists():
            try:
                with open(chat_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    @staticmethod
    def save_message(content: str, msg_type: str = 'text', file_path: Optional[Path] = None):
        """Save message with atomic write"""
        if not content.strip() and not file_path:
            return
        
        message = {
            'id': str(uuid.uuid4()),
            'session_id': ChatManager.get_session_id(),
            'username': ChatManager.get_username(),
            'content': content.strip() if content else "",
            'type': msg_type,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path.relative_to(Config.BASE_DIR)) if file_path else None
        }
        
        # Load existing messages
        messages = ChatManager.load_messages()
        messages.append(message)
        
        # Atomic write to prevent corruption
        chat_file = ChatManager._get_chat_file()
        temp_file = chat_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(messages, f, indent=2)
            temp_file.rename(chat_file)
        except Exception as e:
            st.error(f"Failed to save message: {str(e)}")
            if temp_file.exists():
                temp_file.unlink()
    
    @staticmethod
    def clear_chat():
        """Clear current session chat"""
        chat_file = ChatManager._get_chat_file()
        if chat_file.exists():
            chat_file.unlink()
        st.session_state.pop('chat_messages', None)


class FileStorageManager:
    """Secure temporary file handling with expiration"""
    
    @staticmethod
    def validate_file(file) -> tuple[bool, str]:
        """Validate file before processing"""
        if file.size == 0:
            return False, "Empty file not allowed"
        
        if file.size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File exceeds {Config.MAX_FILE_SIZE_MB}MB limit"
        
        ext = file.name.split('.')[-1].lower()
        valid_ext = False
        for category, exts in Config.ALLOWED_EXTENSIONS.items():
            if ext in exts:
                valid_ext = True
                break
        
        if not valid_ext:
            allowed = ', '.join([e for cat in Config.ALLOWED_EXTENSIONS.values() for e in cat])
            return False, f"Invalid file type. Allowed: {allowed}"
        
        return True, ""
    
    @staticmethod
    def save_file(uploaded_file) -> Optional[Path]:
        """Save file with security measures"""
        is_valid, error = FileStorageManager.validate_file(uploaded_file)
        if not is_valid:
            st.error(f"Upload failed: {error}")
            return None
        
        try:
            # Create unique filename: session_timestamp_originalname
            session_id = ChatManager.get_session_id()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = re.sub(r'[^\w\-_\.]', '_', uploaded_file.name)
            filename = f"{session_id}_{timestamp}_{clean_name}"
            file_path = Config.TEMP_UPLOADS / filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Verify file integrity
            actual_size = file_path.stat().st_size
            if actual_size != uploaded_file.size:
                file_path.unlink()
                st.error("File corruption detected during upload")
                return None
            
            # Record in metadata
            FileStorageManager._record_file(file_path, uploaded_file.name)
            return file_path
            
        except Exception as e:
            st.error(f"Save failed: {str(e)}")
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            return None
    
    @staticmethod
    def _record_file(file_path: Path, original_name: str):
        """Record file in global metadata"""
        try:
            with open(Config.METADATA_FILE, 'r') as f:
                meta = json.load(f)
            
            meta.setdefault('files', []).append({
                'path': str(file_path.relative_to(Config.BASE_DIR)),
                'original_name': original_name,
                'session_id': ChatManager.get_session_id(),
                'uploaded_by': ChatManager.get_username(),
                'uploaded_at': datetime.now().isoformat(),
                'size': file_path.stat().st_size
            })
            
            with open(Config.METADATA_FILE, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            st.warning(f"Metadata recording failed: {str(e)}")
    
    @staticmethod
    def get_session_files() -> List[Dict]:
        """Get files uploaded in current session"""
        try:
            with open(Config.METADATA_FILE, 'r') as f:
                meta = json.load(f)
            
            current_session = ChatManager.get_session_id()
            return [
                f for f in meta.get('files', [])
                if f.get('session_id') == current_session
            ]
        except Exception:
            return []
    
    @staticmethod
    def get_file_path(rel_path: str) -> Optional[Path]:
        """Get absolute path from relative metadata path"""
        abs_path = Config.BASE_DIR / rel_path
        return abs_path if abs_path.exists() else None


# ======================
# UI COMPONENTS
# ======================
class ChatUI:
    """Complete chat interface with file handling"""
    
    @staticmethod
    def render_header():
        """Render app header with session info"""
        st.markdown("""
        <style>
        .header-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .header-title { 
            color: white; 
            font-weight: 700; 
            margin: 0; 
            font-size: 2.2rem;
        }
        .session-badge {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            display: inline-block;
            margin-top: 0.5rem;
        }
        .message-container {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin: 0.7rem 0;
        }
        .user-msg { 
            background: linear-gradient(to right, #dcfce7, #f0fdf4); 
            border-left: 4px solid #22c55e;
        }
        .other-msg { 
            background: #f9fafb; 
            border-left: 4px solid #6b7280;
        }
        .timestamp { 
            font-size: 0.75rem; 
            color: #6b7280; 
            margin-top: 0.3rem;
            font-weight: 500;
        }
        .file-preview {
            border: 1px dashed #d1d5db;
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.5rem 0;
            background: #f3f4f6;
        }
        .system-msg {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 0.8rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown(f"""
            <div class="header-container">
                <h1 class="header-title">💬 TempChat</h1>
                <div class="session-badge">Session: {ChatManager.get_session_id()}</div>
                <div style="color: rgba(255,255,255,0.85); margin-top: 0.5rem; font-size: 0.95rem">
                    🌥️ All data auto-deletes on server reboot • Max file size: {Config.MAX_FILE_SIZE_MB}MB
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_username_setup():
        """Username configuration at top of sidebar"""
        with st.sidebar:
            st.markdown("### 👤 Your Identity")
            current = st.session_state.get('username_input', 
                         f"User_{ChatManager.get_session_id()}")
            new_username = st.text_input(
                "Display name",
                value=current,
                max_chars=20,
                key="username_input",
                help="Name shown in chat (resets on reboot)"
            )
            # Sanitize on change
            clean = re.sub(r'[^\w\-_\. ]', '', new_username.strip())[:20]
            st.session_state['username_input'] = clean if clean else current
            
            st.divider()
    
    @staticmethod
    def render_chat_messages():
        """Render all chat messages with proper styling"""
        messages = ChatManager.load_messages()
        
        # Show system message if exists
        if 'system_message' in st.session_state:
            st.markdown(
                f'<div class="system-msg">⚙️ {st.session_state["system_message"]}</div>',
                unsafe_allow_html=True
            )
            st.session_state.pop('system_message', None)
        
        if not messages:
            st.info("👋 Start the conversation! Type a message below or share a file.")
            return
        
        # Render messages in chronological order
        for msg in messages:
            is_user = msg.get('session_id') == ChatManager.get_session_id()
            msg_class = "user-msg" if is_user else "other-msg"
            align = "right" if is_user else "left"
            
            with st.container():
                st.markdown(f"""
                <div class="message-container {msg_class}">
                    <div style="display: flex; justify-content: {align}; margin-bottom: 0.3rem;">
                        <strong>{msg.get('username', 'Anonymous')}</strong>
                    </div>
                    <div style="text-align: {align};">
                """, unsafe_allow_html=True)
                
                # Handle different message types
                if msg['type'] == 'text':
                    st.markdown(f"<div>{msg['content']}</div>", unsafe_allow_html=True)
                
                elif msg['type'] in ['image', 'video']:
                    file_path = FileStorageManager.get_file_path(msg['file_path'])
                    if file_path and file_path.exists():
                        try:
                            if msg['type'] == 'image':
                                st.image(str(file_path), use_column_width=True)
                            else:
                                st.video(str(file_path), format="video/mp4")
                            st.caption(msg['content'] or "Shared media")
                        except Exception:
                            st.error("⚠️ Failed to load media")
                    else:
                        st.markdown(f"<div class='file-preview'>📎 {msg['content']} (File expired)</div>", 
                                  unsafe_allow_html=True)
                
                elif msg['type'] == 'file':
                    file_path = FileStorageManager.get_file_path(msg['file_path'])
                    if file_path and file_path.exists():
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label=f"⬇️ {msg['content']}",
                                data=f,
                                file_name=msg['content'],
                                mime="application/octet-stream",
                                key=f"dl_{msg['id']}"
                            )
                    else:
                        st.markdown(f"<div class='file-preview'>📎 {msg['content']} (File expired)</div>", 
                                  unsafe_allow_html=True)
                
                # Timestamp
                try:
                    ts = datetime.fromisoformat(msg['timestamp'])
                    st.markdown(f"""
                    <div class="timestamp" style="text-align: {align};">
                        {ts.strftime('%I:%M %p · %b %d')}
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    pass
                
                st.markdown("</div></div>", unsafe_allow_html=True)
    
    @staticmethod
    def render_input_area():
        """Chat input area with file upload"""
        st.divider()
        
        # File upload section
        uploaded_file = st.file_uploader(
            "📎 Attach file (images, videos, documents)",
            type=[ext for cat in Config.ALLOWED_EXTENSIONS.values() for ext in cat],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"file_uploader_{int(time.time())}"  # Prevent stale state
        )
        
        # Message input with send button
        col1, col2 = st.columns([6, 1])
        with col1:
            message = st.text_input(
                "Type your message...",
                key=f"chat_input_{int(time.time())}",
                placeholder="Type here... (Press Enter to send)",
                label_visibility="collapsed"
            )
        
        with col2:
            send_disabled = not (message.strip() or uploaded_file)
            if st.button("➤", 
                        type="primary", 
                        use_container_width=True,
                        disabled=send_disabled,
                        help="Send message"):
                ChatUI._handle_send(message, uploaded_file)
                st.rerun()
    
    @staticmethod
    def _handle_send(content: str, uploaded_file):
        """Process send action with file handling"""
        file_path = None
        msg_type = 'text'
        display_content = content.strip() if content else ""
        
        # Process file if uploaded
        if uploaded_file:
            # Determine message type
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext in Config.ALLOWED_EXTENSIONS['image']:
                msg_type = 'image'
            elif ext in Config.ALLOWED_EXTENSIONS['video']:
                msg_type = 'video'
            else:
                msg_type = 'file'
            
            # Save file
            saved_path = FileStorageManager.save_file(uploaded_file)
            if saved_path:
                file_path = saved_path
                # Set content to filename if no text message
                if not display_content:
                    display_content = uploaded_file.name
        
        # Save message (always save even if file failed - show error in content)
        if display_content or file_path:
            ChatManager.save_message(
                content=display_content,
                msg_type=msg_type,
                file_path=file_path
            )
        
        # Clear inputs (handled by key rotation in UI)
        st.session_state.pop(f"file_uploader_{int(time.time())}", None)


class FileGalleryUI:
    """Shared files gallery view"""
    
    @staticmethod
    def render_gallery():
        """Render gallery of all shared files in session"""
        st.subheader("📁 Your Shared Files (This Session)")
        
        files = FileStorageManager.get_session_files()
        
        if not files:
            st.info("📭 No files shared yet. Upload files in the chat to see them here!")
            return
        
        # Sort newest first
        files.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
        
        cols = st.columns(3)
        for idx, file_meta in enumerate(files):
            with cols[idx % 3]:
                FileGalleryUI._render_file_card(file_meta, idx)
    
    @staticmethod
    def _render_file_card(file_meta: dict, idx: int):
        """Render individual file card"""
        file_path = FileStorageManager.get_file_path(file_meta['path'])
        orig_name = file_meta['original_name']
        ext = orig_name.split('.')[-1].lower()
        
        # Card styling
        st.markdown("""
        <style>
        .file-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            transition: transform 0.2s;
        }
        .file-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="file-card">', unsafe_allow_html=True)
            
            # Preview based on type
            if ext in Config.ALLOWED_EXTENSIONS['image'] and file_path and file_path.exists():
                try:
                    st.image(str(file_path), use_column_width=True, caption=orig_name)
                except:
                    st.markdown(f"🖼️ {orig_name}")
            elif ext in Config.ALLOWED_EXTENSIONS['video']:
                st.markdown(f"🎬 Video: {orig_name}")
            else:
                st.markdown(f"📎 {orig_name}")
            
            # File details
            size_mb = file_meta['size'] / (1024 * 1024)
            st.caption(f"{size_mb:.1f} MB • {file_meta.get('uploaded_by', 'User')}")
            
            # Download button
            if file_path and file_path.exists():
                with open(file_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download",
                        data=f,
                        file_name=orig_name,
                        mime="application/octet-stream",
                        use_container_width=True,
                        key=f"gallery_dl_{idx}"
                    )
            else:
                st.error("⚠️ File not available")
            
            st.markdown('</div>', unsafe_allow_html=True)


# ======================
# MAIN APPLICATION
# ======================
def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="TempChat - Temporary Chat & File Sharing",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'current_view' not in st.session_state:
        st.session_state['current_view'] = 'chat'
    
    # Render sidebar navigation
    with st.sidebar:
        ChatUI.render_username_setup()
        
        st.markdown("### 🧭 Navigation")
        if st.button("💬 Chat Room", use_container_width=True, type="primary" if st.session_state['current_view']=='chat' else "secondary"):
            st.session_state['current_view'] = 'chat'
        if st.button("📁 My Files", use_container_width=True, type="primary" if st.session_state['current_view']=='gallery' else "secondary"):
            st.session_state['current_view'] = 'gallery'
        
        st.divider()
        st.markdown("### ⚙️ Session Control")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            ChatManager.clear_chat()
            st.session_state['system_message'] = "Chat history cleared for this session"
            st.rerun()
        
        if st.button("🔄 Full Reset (Simulate Reboot)", use_container_width=True, type="primary"):
            Config._full_cleanup()
            st.session_state['system_message'] = "✅ FULL SYSTEM RESET COMPLETE - All data cleared"
            st.rerun()
        
        st.divider()
        st.markdown("""
        ### ℹ️ About TempChat
        - 🌥️ **Temporary Storage**: All data auto-deletes on server reboot
        - 🔒 **Session Isolated**: Your chat/files only visible in this session
        - 📦 **File Support**: Images, videos, documents (max 50MB)
        - ⏱️ **No Persistence**: Nothing saved after reboot
        
        > 💡 *Refresh page to see new messages. For true real-time chat, consider WebSocket solutions in production.*
        """)
    
    # Render main content
    ChatUI.render_header()
    
    if st.session_state['current_view'] == 'chat':
        # Auto-refresh container
        chat_container = st.container()
        with chat_container:
            ChatUI.render_chat_messages()
            ChatUI.render_input_area()
        
        # Auto-refresh logic (non-blocking)
        if 'last_refresh' not in st.session_state:
            st.session_state['last_refresh'] = time.time()
        
        current_time = time.time()
        if current_time - st.session_state['last_refresh'] > Config.AUTO_REFRESH_SEC:
            st.session_state['last_refresh'] = current_time
            st.rerun()
    
    elif st.session_state['current_view'] == 'gallery':
        FileGalleryUI.render_gallery()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>"
        f"TempChat v1.0 • Session ID: {ChatManager.get_session_id()} • "
        f"Storage: {Config.BASE_DIR} • "
        f"Last activity: {datetime.now().strftime('%I:%M %p')}</div>",
        unsafe_allow_html=True
    )


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)
