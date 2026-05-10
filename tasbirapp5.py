"""
MEMORYVAULT FORUM PRO v8.3.0
WhatsApp-style Chat + Upload Timer Selection + Fixed Gallery
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps
import base64
import datetime
import uuid
from typing import Dict, List, Optional
import os
import warnings
import time
import io

warnings.filterwarnings('ignore')

# ============================================================================
# SESSION STORAGE
# ============================================================================
class SessionStorage:
    def __init__(self):
        defaults = {
            'session_uploads': {},
            'chat_messages': [],
            'guest_users': {},
            'active_sessions': set(),
            'message_counter': 0,
            'photo_timers': {},
            'user_upvotes': set(),
            'deleted_messages': set(),
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def store_upload(self, file_id: str, file_bytes: bytes, filename: str, mime_type: str):
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

    def add_chat_message(self, user_id: str, username: str, content: str, 
                        media_ids: List[str] = None, reply_to: Optional[str] = None,
                        timer_duration: int = 300):
        st.session_state.message_counter += 1
        msg_id = f"msg_{st.session_state.message_counter}_{uuid.uuid4().hex[:8]}"

        msg = {
            'message_id': msg_id,
            'user_id': user_id,
            'username': username,
            'content': content,
            'media_ids': media_ids or [],
            'reply_to': reply_to,
            'timestamp': datetime.datetime.now().isoformat(),
            'likes': set(),
            'edited': False,
            'deleted': False
        }
        st.session_state.chat_messages.append(msg)

        if media_ids:
            st.session_state.photo_timers[msg_id] = {
                'created_at': time.time(),
                'base_duration': timer_duration,
                'upvote_extension': 0,
                'total_duration': timer_duration,
                'expires_at': time.time() + timer_duration,
                'media_ids': media_ids.copy(),
                'timer_choice': timer_duration
            }
        return msg_id

    def get_chat_history(self, limit: int = 100) -> List[Dict]:
        messages = []
        now = time.time()

        for msg in reversed(st.session_state.chat_messages[-limit:]):
            if msg['message_id'] in st.session_state.deleted_messages:
                continue

            timer = st.session_state.photo_timers.get(msg['message_id'])
            if timer and now > timer['expires_at']:
                msg = msg.copy()
                msg['media_expired'] = True
                msg['expired_media_ids'] = msg['media_ids']
                msg['media_ids'] = []
            else:
                msg = msg.copy()
                msg['media_expired'] = False
                if timer:
                    msg['time_remaining'] = max(0, int(timer['expires_at'] - now))
                    msg['upvote_count'] = timer['upvote_extension'] // 60
                    msg['timer_choice'] = timer.get('timer_choice', 300)

            msg['like_count'] = len(msg['likes'])
            messages.append(msg)
        return messages

    def upvote_photo_timer(self, message_id: str, user_id: str) -> bool:
        vote_key = (user_id, message_id)
        if vote_key in st.session_state.user_upvotes:
            return False

        timer = st.session_state.photo_timers.get(message_id)
        if not timer or time.time() > timer['expires_at']:
            return False

        st.session_state.user_upvotes.add(vote_key)
        extension = 60
        timer['upvote_extension'] += extension
        timer['total_duration'] += extension
        timer['expires_at'] += extension
        return True

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

    def delete_message(self, message_id: str, user_id: str) -> bool:
        for msg in st.session_state.chat_messages:
            if msg['message_id'] == message_id and msg['user_id'] == user_id:
                st.session_state.deleted_messages.add(message_id)
                timer = st.session_state.photo_timers.get(message_id)
                if timer:
                    timer['expires_at'] = 0
                return True
        return False

    def register_guest(self, username: str) -> str:
        user_id = f"guest_{uuid.uuid4().hex[:12]}"
        st.session_state.guest_users[user_id] = {
            'user_id': user_id,
            'username': username,
            'joined_at': datetime.datetime.now().isoformat(),
            'last_active': time.time(),
            'status': 'online'
        }
        st.session_state.active_sessions.add(user_id)
        return user_id

    def update_activity(self, user_id: str):
        user = st.session_state.guest_users.get(user_id)
        if user:
            user['last_active'] = time.time()
            user['status'] = 'online'

    def get_online_count(self) -> int:
        now = time.time()
        active = set()
        for uid, user in list(st.session_state.guest_users.items()):
            if now - user.get('last_active', 0) < 1800:
                active.add(uid)
            else:
                user['status'] = 'offline'
        st.session_state.active_sessions = active
        return len(active)

    def get_online_users(self) -> List[Dict]:
        now = time.time()
        users = []
        for uid in st.session_state.active_sessions:
            user = st.session_state.guest_users.get(uid)
            if user and now - user.get('last_active', 0) < 1800:
                users.append(user)
        return sorted(users, key=lambda x: x['last_active'], reverse=True)

    def get_user(self, user_id: str) -> Optional[Dict]:
        return st.session_state.guest_users.get(user_id)


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Forum Pro"
    VERSION = "8.3.0"
    MAX_MESSAGE_LENGTH = 2000
    MAX_FILE_SIZE = 50 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4','.mov','.avi','.mkv','.webm','.wmv','.flv','.m4v']
    IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.gif','.bmp','.webp','.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    THUMBNAIL_SIZE = (400, 400)
    MAX_MESSAGES = 200
    TIMER_OPTIONS = {
        "1 minute": 60,
        "3 minutes": 180,
        "5 minutes": 300,
        "10 minutes": 600,
        "30 minutes": 1800,
        "1 hour": 3600,
    }


# ============================================================================
# MEDIA PROCESSOR
# ============================================================================
class MediaProcessor:
    @staticmethod
    def process_upload(uploaded_file) -> Optional[Dict]:
        if uploaded_file is None:
            return None

        file_bytes = uploaded_file.getvalue()
        file_size = len(file_bytes)

        if file_size > Config.MAX_FILE_SIZE:
            st.error(f"File too large. Max: {Config.MAX_FILE_SIZE/(1024*1024):.0f}MB")
            return None

        file_ext = Path(uploaded_file.name).suffix.lower()
        file_id = str(uuid.uuid4())

        if file_ext in Config.IMAGE_EXTENSIONS:
            return MediaProcessor._process_image(file_id, file_bytes, uploaded_file.name, file_ext)
        elif file_ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaProcessor._process_video(file_id, file_bytes, uploaded_file.name, file_ext)
        else:
            st.error(f"Unsupported: {file_ext}")
            return None

    @staticmethod
    def _process_image(file_id, file_bytes, filename, ext):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)

            thumb = img.copy()
            thumb.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format='JPEG', quality=85)
            thumb_b64 = base64.b64encode(buf.getvalue()).decode()

            hd = img.copy()
            hd.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            hd_buf = io.BytesIO()
            hd.save(hd_buf, format='JPEG', quality=90)
            hd_b64 = base64.b64encode(hd_buf.getvalue()).decode()

            storage = SessionStorage()
            storage.store_upload(file_id, file_bytes, filename, f"image/{ext.replace('.', '')}")

            return {
                'file_id': file_id,
                'type': 'image',
                'filename': filename,
                'dimensions': img.size,
                'thumbnail': f"data:image/jpeg;base64,{thumb_b64}",
                'hd_url': f"data:image/jpeg;base64,{hd_b64}",
                'size': file_size
            }
        except Exception as e:
            st.error(f"Image error: {e}")
            return None

    @staticmethod
    def _process_video(file_id, file_bytes, filename, ext):
        storage = SessionStorage()
        storage.store_upload(file_id, file_bytes, filename, f"video/{ext.replace('.', '')}")
        return {
            'file_id': file_id,
            'type': 'video',
            'filename': filename,
            'thumbnail': None,
            'hd_url': None,
            'size': len(file_bytes)
        }


# ============================================================================
# CSS - WhatsApp Style
# ============================================================================
def inject_css():
    st.markdown("""
    <style>
    /* WhatsApp-style chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        background: #e5ddd5;
        border-radius: 12px;
        padding: 10px;
        min-height: 60vh;
    }

    /* Message bubbles */
    .msg-row {
        display: flex;
        margin-bottom: 8px;
        width: 100%;
    }
    .msg-row.own {
        justify-content: flex-end;
    }
    .msg-row.other {
        justify-content: flex-start;
    }

    .msg-bubble {
        max-width: 70%;
        padding: 8px 12px;
        border-radius: 12px;
        position: relative;
        word-wrap: break-word;
    }
    .msg-bubble.own {
        background: #dcf8c6;
        border-bottom-right-radius: 2px;
        margin-left: auto;
    }
    .msg-bubble.other {
        background: #ffffff;
        border-bottom-left-radius: 2px;
        margin-right: auto;
    }

    .msg-sender {
        font-size: 12px;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 2px;
    }
    .msg-text {
        font-size: 14px;
        color: #111;
        line-height: 1.4;
        margin-bottom: 4px;
    }
    .msg-time {
        font-size: 10px;
        color: #888;
        text-align: right;
        margin-top: 2px;
    }
    .msg-media {
        margin: 6px 0;
        border-radius: 8px;
        overflow: hidden;
    }
    .msg-media img {
        max-width: 100%;
        max-height: 300px;
        border-radius: 8px;
        cursor: pointer;
    }

    /* Timer badge inside bubble */
    .timer-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(255,152,0,0.15);
        color: #e65100;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        margin: 4px 0;
    }
    .timer-expired {
        background: rgba(244,67,54,0.15);
        color: #c62828;
    }

    /* Action buttons row */
    .msg-actions {
        display: flex;
        gap: 8px;
        margin-top: 4px;
        font-size: 12px;
    }
    .action-btn {
        background: none;
        border: none;
        cursor: pointer;
        color: #666;
        padding: 2px 6px;
        border-radius: 4px;
        transition: all 0.2s;
    }
    .action-btn:hover {
        background: rgba(0,0,0,0.05);
        color: #333;
    }
    .action-btn.liked {
        color: #ff4b4b;
    }
    .upvote-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 11px;
    }

    /* Online sidebar */
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

    /* Guest login */
    .guest-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        margin: 40px auto;
        max-width: 500px;
    }

    /* Input area */
    .input-area {
        background: #f0f0f0;
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }

    /* Hide default streamlit form margins */
    .stForm {
        border: none !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# GUEST AUTH
# ============================================================================
class GuestAuth:
    @staticmethod
    def render_login():
        st.markdown("""
        <div class="guest-card">
            <h1>👋 MemoryVault Forum</h1>
            <p style="font-size:16px; opacity:0.9;">Join the conversation instantly.<br>No account needed.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Choose display name", 
                                   placeholder="e.g., PhotoFan2024",
                                   max_chars=30,
                                   key="guest_username_input")

            if st.button("🚀 Join Forum", use_container_width=True, type="primary"):
                if username and len(username.strip()) >= 2:
                    storage = SessionStorage()
                    user_id = storage.register_guest(username.strip())
                    st.session_state.current_user_id = user_id
                    st.session_state.current_username = username.strip()
                    st.session_state.authenticated = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Name must be 2+ characters")
        return False

    @staticmethod
    def check_auth():
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if st.session_state.authenticated and 'current_user_id' in st.session_state:
            storage = SessionStorage()
            storage.update_activity(st.session_state.current_user_id)
            return True
        return GuestAuth.render_login()

    @staticmethod
    def logout():
        st.session_state.authenticated = False
        st.session_state.current_user_id = None
        st.session_state.current_username = None
        st.rerun()


# ============================================================================
# FORUM CHAT - WhatsApp Style
# ============================================================================
class ForumChat:
    def __init__(self):
        self.storage = SessionStorage()
        self.processor = MediaProcessor()

    def render_chat_input(self):
        st.markdown("<div class='input-area'>", unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            message_text = st.text_area("Message", 
                                       placeholder="Type a message...",
                                       max_chars=Config.MAX_MESSAGE_LENGTH,
                                       height=60,
                                       label_visibility="collapsed")

            col1, col2 = st.columns([3,1])
            with col1:
                uploaded_files = st.file_uploader(
                    "📎 Photos/Videos", 
                    accept_multiple_files=True,
                    type=list(Config.ALLOWED_EXTENSIONS),
                    label_visibility="collapsed"
                )
            with col2:
                timer_choice = st.selectbox(
                    "⏱️ Timer",
                    options=list(Config.TIMER_OPTIONS.keys()),
                    index=2,  # Default 5 minutes
                    help="How long photos stay visible"
                )

            submitted = st.form_submit_button("📤 Send", use_container_width=True, type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if not message_text.strip() and not uploaded_files:
                st.warning("Please write a message or upload media")
                return

            media_metadata = []
            if uploaded_files:
                for file in uploaded_files:
                    meta = self.processor.process_upload(file)
                    if meta:
                        media_metadata.append(meta)

            user_id = st.session_state.current_user_id
            username = st.session_state.current_username
            media_ids = [m['file_id'] for m in media_metadata]
            timer_seconds = Config.TIMER_OPTIONS[timer_choice]

            self.storage.add_chat_message(
                user_id=user_id,
                username=username,
                content=message_text.strip(),
                media_ids=media_ids,
                timer_duration=timer_seconds
            )

            st.success("Sent! ✅")
            time.sleep(0.3)
            st.rerun()

    def format_time_remaining(self, seconds: int) -> str:
        if seconds <= 0:
            return "Expired"
        mins, secs = divmod(seconds, 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def render_message(self, msg: Dict):
        is_own = msg['user_id'] == st.session_state.get('current_user_id')
        row_class = "own" if is_own else "other"
        bubble_class = "own" if is_own else "other"
        current_uid = st.session_state.get('current_user_id', '')

        try:
            ts = datetime.datetime.fromisoformat(msg['timestamp'])
            time_str = ts.strftime("%H:%M")
        except:
            time_str = "now"

        is_liked = current_uid in msg.get('likes', set())
        like_icon = "❤️" if is_liked else "🤍"

        # Timer HTML
        timer_html = ""
        if msg.get('media_ids') and not msg.get('media_expired'):
            remaining = msg.get('time_remaining', 0)
            upvotes = msg.get('upvote_count', 0)
            timer_choice = msg.get('timer_choice', 300)
            timer_label = [k for k, v in Config.TIMER_OPTIONS.items() if v == timer_choice]
            timer_label = timer_label[0] if timer_label else "5 min"

            timer_html = f"""
            <div class="timer-badge">
                ⏱️ {self.format_time_remaining(remaining)} / {timer_label}
                {f" • ⬆️ +{upvotes}m" if upvotes > 0 else ""}
            </div>
            """
        elif msg.get('media_expired'):
            timer_html = """
            <div class="timer-badge timer-expired">
                ⏱️ Photos expired
            </div>
            """

        # Media HTML
        media_html = ""
        if msg.get('media_ids') and not msg.get('media_expired'):
            for media_id in msg['media_ids']:
                upload = self.storage.get_upload(media_id)
                if upload:
                    mime = upload['mime_type']
                    b64 = base64.b64encode(upload['bytes']).decode()
                    data_url = f"data:{mime};base64,{b64}"

                    if mime.startswith('image'):
                        media_html += f'<div class="msg-media"><img src="{data_url}"></div>'
                    elif mime.startswith('video'):
                        media_html += f'<div class="msg-media">[Video: {upload["filename"]}]</div>'

        # Actions
        actions_html = f"""
        <div class="msg-actions">
            <span class="action-btn {'liked' if is_liked else ''}">{like_icon} {msg.get('like_count', 0)}</span>
        """

        if msg.get('media_ids') and not msg.get('media_expired') and not is_own:
            actions_html += '<span class="action-btn upvote-btn">⬆️ +1m</span>'

        if is_own:
            actions_html += '<span class="action-btn">🗑️</span>'

        actions_html += '</div>'

        # Full message HTML
        st.markdown(f"""
        <div class="msg-row {row_class}">
            <div class="msg-bubble {bubble_class}">
                <div class="msg-sender">@{msg['username']}</div>
                <div class="msg-text">{msg['content']}</div>
                {timer_html}
                {media_html}
                <div class="msg-time">{time_str} {'✓✓' if is_own else ''}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Streamlit buttons below HTML (for actual interactivity)
        cols = st.columns([1,1,1,1,6])
        with cols[0]:
            if st.button(f"{like_icon} {msg.get('like_count', 0)}", 
                        key=f"like_{msg['message_id']}",
                        help="Like"):
                self.storage.like_message(msg['message_id'], current_uid)
                st.rerun()

        if msg.get('media_ids') and not msg.get('media_expired') and not is_own:
            with cols[1]:
                if st.button("⬆️ +1m", 
                            key=f"upvote_{msg['message_id']}",
                            help="Extend timer"):
                    success = self.storage.upvote_photo_timer(msg['message_id'], current_uid)
                    if success:
                        st.success("+1 minute! ✅")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.info("Already upvoted")

        with cols[2]:
            if st.button("💬", key=f"reply_{msg['message_id']}", help="Reply"):
                st.session_state.replying_to = msg['message_id']
                st.session_state.replying_to_user = msg['username']
                st.rerun()

        with cols[3]:
            if is_own:
                if st.button("🗑️", key=f"del_{msg['message_id']}", help="Delete"):
                    self.storage.delete_message(msg['message_id'], current_uid)
                    st.rerun()

        if st.session_state.get('replying_to') == msg['message_id']:
            st.info(f"Replying to @{msg['username']}")

    def render_chat_history(self):
        messages = self.storage.get_chat_history(Config.MAX_MESSAGES)

        if not messages:
            st.info("No messages yet. Start the conversation! 🎉")
            return

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in messages:
            self.render_message(msg)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# ONLINE SIDEBAR
# ============================================================================
class OnlineSidebar:
    @staticmethod
    def render():
        storage = SessionStorage()
        count = storage.get_online_count()
        users = storage.get_online_users()

        st.sidebar.markdown(f"""
        <div class="online-badge">
            <div class="online-dot"></div>
            <span>{count} online now</span>
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.divider()

        st.sidebar.subheader("👥 Active Guests")
        for user in users[:15]:
            st.sidebar.caption(f"🟢 @{user['username']}")

        if len(users) > 15:
            st.sidebar.caption(f"...and {len(users)-15} more")

        st.sidebar.divider()

        total_guests = len(st.session_state.guest_users)
        total_msgs = len(st.session_state.chat_messages)
        total_uploads = len(st.session_state.session_uploads)
        active_timers = sum(1 for t in st.session_state.photo_timers.values() 
                          if time.time() < t['expires_at'])

        col1, col2 = st.sidebar.columns(2)
        col1.metric("Guests", total_guests)
        col2.metric("Posts", total_msgs)

        col3, col4 = st.sidebar.columns(2)
        col3.metric("Media", total_uploads)
        col4.metric("Active", active_timers)

        st.sidebar.divider()
        st.sidebar.caption("⏳ Photos auto-delete based on timer")
        st.sidebar.caption("⬆️ Upvotes extend by +1 minute")


# ============================================================================
# GALLERY - FIXED
# ============================================================================
class Gallery:
    @staticmethod
    def render():
        uploads = st.session_state.session_uploads

        if not uploads:
            st.info("No photos shared yet. Post in the forum! 📸")
            return

        st.subheader("📸 Community Gallery")

        # Get all non-expired images from active timers
        active_media_ids = set()
        now = time.time()
        for timer in st.session_state.photo_timers.values():
            if now < timer['expires_at']:
                for mid in timer.get('media_ids', []):
                    active_media_ids.add(mid)

        images = []
        for fid, data in uploads.items():
            if data['mime_type'].startswith('image') and fid in active_media_ids:
                images.append((fid, data))

        if not images:
            st.info("No active photos. Check the forum for new uploads! ⏳")
            return

        # Show count
        st.caption(f"Showing {len(images)} active photo(s)")

        cols = st.columns(min(len(images), 4))
        for idx, (fid, data) in enumerate(images):
            with cols[idx % 4]:
                b64 = base64.b64encode(data['bytes']).decode()
                st.image(f"data:{data['mime_type']};base64,{b64}", use_container_width=True)
                st.caption(f"{data['filename'][:20]}...")


# ============================================================================
# MAIN APP
# ============================================================================
class ForumApp:
    def __init__(self):
        st.set_page_config(page_title=Config.APP_NAME, layout="wide")
        inject_css()
        self.chat = ForumChat()
        self.storage = SessionStorage()

        if 'replying_to' not in st.session_state:
            st.session_state.replying_to = None
        if 'replying_to_user' not in st.session_state:
            st.session_state.replying_to_user = None

    def render_sidebar(self):
        with st.sidebar:
            st.title("📁 MemoryVault")
            st.caption(f"v{Config.VERSION}")

            OnlineSidebar.render()

            st.subheader("Navigation")
            page = st.radio("Go to", ["💬 Forum", "📸 Gallery", "ℹ️ About"], key="nav_page")

            st.divider()
            st.subheader("Your Profile")
            st.write(f"👤 @{st.session_state.get('current_username', 'Guest')}")
            st.write(f"🆔 {st.session_state.get('current_user_id', '')[:16]}...")

            if st.button("🚪 Leave", use_container_width=True):
                GuestAuth.logout()

            st.divider()
            st.caption("💡 Session-based: data clears on refresh")
            st.caption("⏱️ Choose photo timer at upload")

    def render_forum(self):
        st.markdown("""
        <div style="text-align:center; padding:10px 0 20px;">
            <h1>💬 MemoryVault Forum</h1>
            <p style="color:#666; font-size:14px;">
                WhatsApp-style chat • Choose photo timer • Upvote to extend
            </p>
        </div>
        """, unsafe_allow_html=True)

        self.chat.render_chat_input()
        st.divider()
        self.chat.render_chat_history()

    def render_gallery(self):
        Gallery.render()

    def render_about(self):
        st.title("ℹ️ About MemoryVault Forum")
        st.markdown("""
        ### 🌟 v8.3.0 Features

        **WhatsApp-Style Chat:**
        - Your messages appear on the **right** (green bubbles)
        - Others appear on the **left** (white bubbles)
        - Read receipts (✓✓) on own messages

        **Photo Timer Selection:**
        - Choose timer **before** uploading: 1min, 3min, 5min, 10min, 30min, 1hour
        - Timer shown on each photo
        - Photos auto-hide when timer expires

        **Community Upvotes:**
        - Click ⬆️ to extend any photo by +1 minute
        - Each user can upvote once per photo

        **Live Features:**
        - Real-time online counter
        - Active guest list
        - Gallery shows only non-expired photos

        **Privacy:**
        - Zero persistence
        - Session-only storage
        - Perfect for Streamlit Cloud
        """)

        st.info(f"📊 Currently online: {self.storage.get_online_count()} guests")

    def run(self):
        if not GuestAuth.check_auth():
            return

        self.render_sidebar()

        page = st.session_state.get('nav_page', '💬 Forum')
        if page == "💬 Forum":
            self.render_forum()
        elif page == "📸 Gallery":
            self.render_gallery()
        else:
            self.render_about()


# ============================================================================
# MAIN
# ============================================================================
def main():
    app = ForumApp()
    app.run()

if __name__ == "__main__":
    main()
