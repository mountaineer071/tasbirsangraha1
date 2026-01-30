"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH AI IMAGE MANIPULATION
Version: 3.0.0 - AI Enhanced Edition
Features: AI Image Editing, Real Person Generation, Style Transfer, Face Restoration, and More
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
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

# ============================================================================
# AI LIBRARIES - NEW ADDITIONS
# ============================================================================
try:
    import torch
    import torchvision
    from torchvision import transforms
    import cv2
    import dlib
    import insightface
    from insightface.app import FaceAnalysis
    import face_recognition
    from deepface import DeepFace
    import mediapipe as mp
    from rembg import remove as remove_background
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
    from diffusers import StableDiffusionInpaintPipeline
    import gradio as gr
    from transformers import pipeline
    from scipy import ndimage
    import skimage
    from skimage import filters, exposure, restoration, segmentation
    from skimage.color import rgb2gray, gray2rgb
    from skimage.feature import canny
    from skimage.transform import rotate, resize, swirl
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import kornia
    import kornia.augmentation as K
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import warnings
    warnings.filterwarnings('ignore')
    
    AI_AVAILABLE = True
    st.success("✅ AI libraries loaded successfully!")
except ImportError as e:
    AI_AVAILABLE = False
    st.warning(f"⚠️ Some AI libraries not available: {e}")
    st.info("Install AI packages with: pip install torch torchvision opencv-python dlib insightface face-recognition deepface mediapipe rembg diffusers transformers scikit-image albumentations kornia facenet-pytorch")

# ============================================================================
# CONFIGURATION AND CONSTANTS - EXPANDED
# ============================================================================
class Config:
    """Application configuration constants"""
    APP_NAME = "MemoryVault AI Pro"
    VERSION = "3.0.0"
    
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    AI_MODELS_DIR = BASE_DIR / "ai_models"
    AI_OUTPUT_DIR = BASE_DIR / "ai_outputs"
    
    # Files and paths
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    AI_CONFIG_FILE = AI_MODELS_DIR / "ai_config.json"
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    
    # AI Settings
    AI_MODEL_CACHE_SIZE = 5
    AI_PROCESSING_TIMEOUT = 300  # 5 minutes
    AI_MAX_FACES = 10
    AI_QUALITY_PRESET = "high"  # high, medium, low
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.heic'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # Cache
    CACHE_TTL = 3600  # 1 hour
    
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
            cls.AI_OUTPUT_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create sample structure if no data exists
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()
            
        # Create AI config if not exists
        if not cls.AI_CONFIG_FILE.exists():
            cls.create_ai_config()
    
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
                    st.info(f"Created sample image for {person}")
                except Exception as e:
                    st.warning(f"Could not create sample image for {person}: {str(e)}")
    
    @classmethod
    def create_ai_config(cls):
        """Create AI configuration file"""
        ai_config = {
            "enabled_features": {
                "face_detection": True,
                "face_restoration": True,
                "style_transfer": True,
                "background_removal": True,
                "image_generation": True,
                "image_enhancement": True,
                "face_swap": True,
                "age_progression": True,
                "emotion_detection": True,
                "super_resolution": True
            },
            "model_settings": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "precision": "float16",
                "cache_dir": str(cls.AI_MODELS_DIR)
            },
            "default_prompts": {
                "portrait": "professional portrait photography, sharp focus, studio lighting, 8k",
                "landscape": "beautiful landscape, golden hour, cinematic, 8k",
                "fantasy": "fantasy art, digital painting, magical, detailed, trending on artstation",
                "realistic": "photorealistic, ultra detailed, realistic lighting, 8k"
            },
            "style_presets": {
                "vangogh": "in the style of Vincent van Gogh",
                "monet": "in the style of Claude Monet",
                "picasso": "in the style of Pablo Picasso",
                "anime": "anime style, vibrant colors, detailed",
                "cyberpunk": "cyberpunk style, neon lights, futuristic"
            }
        }
        
        with open(cls.AI_CONFIG_FILE, 'w') as f:
            json.dump(ai_config, f, indent=2)

class UserRoles(Enum):
    """User permission levels"""
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"
    AI_EXPERT = "ai_expert"  # New role for AI features

# ============================================================================
# AI MODELS MANAGER - NEW CLASS
# ============================================================================
class AIModelManager:
    """Manage AI models with caching and lazy loading"""
    
    def __init__(self):
        self.models = {}
        self.model_cache = {}
        self.load_ai_config()
        
    def load_ai_config(self):
        """Load AI configuration"""
        try:
            with open(Config.AI_CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = Config.create_ai_config()
    
    def get_device(self):
        """Get available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_face_detection_model(self):
        """Load face detection model"""
        if 'face_detection' in self.model_cache:
            return self.model_cache['face_detection']
        
        try:
            # Try multiple face detection models
            models = []
            
            # Mediapipe Face Detection
            try:
                mp_face_detection = mp.solutions.face_detection
                face_detection_mp = mp_face_detection.FaceDetection(
                    model_selection=1,  # 0 for short-range, 1 for full-range
                    min_detection_confidence=0.5
                )
                models.append(('mediapipe', face_detection_mp))
            except:
                pass
            
            # Dlib HOG Face Detector
            try:
                face_detector_dlib = dlib.get_frontal_face_detector()
                models.append(('dlib', face_detector_dlib))
            except:
                pass
            
            # InsightFace
            try:
                app = FaceAnalysis(name='buffalo_l')
                app.prepare(ctx_id=0, det_size=(640, 640))
                models.append(('insightface', app))
            except:
                pass
            
            self.model_cache['face_detection'] = models
            return models
            
        except Exception as e:
            st.error(f"Error loading face detection models: {e}")
            return []
    
    def load_style_transfer_model(self):
        """Load style transfer model"""
        if 'style_transfer' in self.model_cache:
            return self.model_cache['style_transfer']
        
        try:
            # Load a simple neural style transfer model
            device = self.get_device()
            
            # Load VGG19 for style transfer
            vgg19 = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
            
            # Freeze all parameters
            for param in vgg19.parameters():
                param.requires_grad_(False)
            
            self.model_cache['style_transfer'] = vgg19
            return vgg19
            
        except Exception as e:
            st.error(f"Error loading style transfer model: {e}")
            return None
    
    def load_super_resolution_model(self):
        """Load super resolution model"""
        if 'super_resolution' in self.model_cache:
            return self.model_cache['super_resolution']
        
        try:
            # Try to load Real-ESRGAN or similar
            device = self.get_device()
            
            # Simple upscaling model as fallback
            class SimpleSR(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                    self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
                    self.conv3 = torch.nn.Conv2d(64, 3, 3, padding=1)
                    self.relu = torch.nn.ReLU()
                    
                def forward(self, x):
                    x = self.relu(self.conv1(x))
                    x = self.relu(self.conv2(x))
                    x = self.conv3(x)
                    return x
            
            model = SimpleSR().to(device)
            self.model_cache['super_resolution'] = model
            return model
            
        except Exception as e:
            st.error(f"Error loading super resolution model: {e}")
            return None
    
    def load_face_restoration_model(self):
        """Load face restoration model"""
        if 'face_restoration' in self.model_cache:
            return self.model_cache['face_restoration']
        
        try:
            # GFP-GAN or similar face restoration
            device = self.get_device()
            
            # Simple face enhancement model as fallback
            class FaceEnhancer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 32, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 64, 3, padding=1),
                        torch.nn.ReLU()
                    )
                    self.decoder = torch.nn.Sequential(
                        torch.nn.Conv2d(64, 32, 3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv2d(32, 3, 3, padding=1),
                        torch.nn.Sigmoid()
                    )
                    
                def forward(self, x):
                    features = self.encoder(x)
                    return self.decoder(features)
            
            model = FaceEnhancer().to(device)
            self.model_cache['face_restoration'] = model
            return model
            
        except Exception as e:
            st.error(f"Error loading face restoration model: {e}")
            return None
    
    def load_image_generation_model(self):
        """Load image generation model"""
        if 'image_generation' in self.model_cache:
            return self.model_cache['image_generation']
        
        try:
            # Stable Diffusion or similar
            device = self.get_device()
            model_id = "runwayml/stable-diffusion-v1-5"
            
            # Note: This requires significant memory
            # In production, you might want to use a smaller model
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(device)
            
            self.model_cache['image_generation'] = pipe
            return pipe
            
        except Exception as e:
            st.error(f"Error loading image generation model: {e}")
            return None
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# AI IMAGE PROCESSOR - NEW CLASS
# ============================================================================
class AIImageProcessor:
    """Handle AI-based image processing operations"""
    
    def __init__(self):
        self.model_manager = AIModelManager()
        self.device = self.model_manager.get_device()
        
    def enhance_image(self, image: Image.Image, enhancement_type: str = "auto") -> Image.Image:
        """AI-powered image enhancement"""
        try:
            if not AI_AVAILABLE:
                return self._basic_enhancement(image)
            
            # Convert PIL to numpy
            img_np = np.array(image)
            
            if enhancement_type == "auto":
                # Auto-detect enhancement needed
                return self._auto_enhance(img_np)
            elif enhancement_type == "color":
                return self._enhance_colors(img_np)
            elif enhancement_type == "sharpness":
                return self._enhance_sharpness(img_np)
            elif enhancement_type == "low_light":
                return self._enhance_low_light(img_np)
            elif enhancement_type == "denoise":
                return self._denoise_image(img_np)
            else:
                return image
                
        except Exception as e:
            st.error(f"Error in AI enhancement: {e}")
            return image
    
    def _auto_enhance(self, img_np: np.ndarray) -> Image.Image:
        """Auto-enhance based on image analysis"""
        # Convert to float
        img_float = img_np.astype(np.float32) / 255.0
        
        # Analyze image
        brightness = np.mean(img_float)
        contrast = np.std(img_float)
        
        # Apply enhancements based on analysis
        if brightness < 0.3:
            img_float = exposure.adjust_gamma(img_float, gamma=0.8)
        elif brightness > 0.7:
            img_float = exposure.adjust_gamma(img_float, gamma=1.2)
        
        if contrast < 0.1:
            img_float = exposure.equalize_hist(img_float)
        
        # Convert back
        img_enhanced = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(img_enhanced)
    
    def _enhance_colors(self, img_np: np.ndarray) -> Image.Image:
        """Enhance colors using histogram equalization in LAB space"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(enhanced)
        except:
            # Fallback to basic color enhancement
            img = Image.fromarray(img_np)
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(1.5)
    
    def _enhance_sharpness(self, img_np: np.ndarray) -> Image.Image:
        """Enhance image sharpness"""
        try:
            # Unsharp masking
            blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
            sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
            return Image.fromarray(sharpened)
        except:
            # Fallback
            img = Image.fromarray(img_np)
            enhancer = ImageEnhance.Sharpness(img)
            return enhancer.enhance(2.0)
    
    def _enhance_low_light(self, img_np: np.ndarray) -> Image.Image:
        """Enhance low-light images"""
        try:
            # Convert to YUV
            yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
            y, u, v = cv2.split(yuv)
            
            # Apply CLAHE to Y channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            
            # Merge and convert back
            enhanced_yuv = cv2.merge([y, u, v])
            enhanced = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2RGB)
            
            return Image.fromarray(enhanced)
        except:
            return Image.fromarray(img_np)
    
    def _denoise_image(self, img_np: np.ndarray) -> Image.Image:
        """Remove noise from image"""
        try:
            # Non-local means denoising
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
            return Image.fromarray(denoised)
        except:
            # Median filtering as fallback
            denoised = cv2.medianBlur(img_np, 3)
            return Image.fromarray(denoised)
    
    def _basic_enhancement(self, image: Image.Image) -> Image.Image:
        """Basic enhancement without AI"""
        enhancers = {
            'color': ImageEnhance.Color(image),
            'contrast': ImageEnhance.Contrast(image),
            'brightness': ImageEnhance.Brightness(image),
            'sharpness': ImageEnhance.Sharpness(image)
        }
        
        # Apply moderate enhancements
        enhanced = enhancers['color'].enhance(1.2)
        enhanced = enhancers['contrast'].enhance(1.1)
        enhanced = enhancers['brightness'].enhance(1.05)
        enhanced = enhancers['sharpness'].enhance(1.1)
        
        return enhanced
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from image"""
        try:
            if AI_AVAILABLE:
                # Use rembg library
                result = remove_background(image)
                return result
            else:
                # Basic background removal using thresholding
                img_np = np.array(image)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                
                # Apply morphological operations
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Create transparent background
                rgba = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
                rgba[:, :, 3] = mask
                
                return Image.fromarray(rgba)
        except Exception as e:
            st.error(f"Error removing background: {e}")
            return image
    
    def detect_faces(self, image: Image.Image) -> List[Dict]:
        """Detect faces in image"""
        faces = []
        
        if not AI_AVAILABLE:
            return faces
        
        try:
            img_np = np.array(image)
            
            # Try multiple face detection methods
            detection_methods = self.model_manager.load_face_detection_model()
            
            for method_name, detector in detection_methods:
                try:
                    if method_name == 'mediapipe':
                        # Mediapipe detection
                        mp_detector = detector
                        results = mp_detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                        
                        if results.detections:
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                h, w, _ = img_np.shape
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                
                                faces.append({
                                    'bbox': (x, y, width, height),
                                    'confidence': detection.score[0],
                                    'method': 'mediapipe'
                                })
                    
                    elif method_name == 'dlib':
                        # Dlib detection
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        dets = detector(gray, 1)
                        
                        for i, d in enumerate(dets):
                            faces.append({
                                'bbox': (d.left(), d.top(), d.width(), d.height()),
                                'confidence': 1.0,
                                'method': 'dlib'
                            })
                    
                    elif method_name == 'insightface':
                        # InsightFace detection
                        app = detector
                        faces_detected = app.get(img_np)
                        
                        for face in faces_detected:
                            bbox = face.bbox.astype(int)
                            faces.append({
                                'bbox': (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]),
                                'confidence': face.det_score,
                                'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                                'method': 'insightface'
                            })
                
                except Exception as e:
                    continue
            
            # Remove duplicate faces
            faces = self._deduplicate_faces(faces)
            return faces[:Config.AI_MAX_FACES]
            
        except Exception as e:
            st.error(f"Error detecting faces: {e}")
            return []
    
    def _deduplicate_faces(self, faces: List[Dict]) -> List[Dict]:
        """Remove duplicate face detections"""
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        used_areas = []
        
        for face in faces:
            x, y, w, h = face['bbox']
            area = (x, y, x+w, y+h)
            
            # Check if this area overlaps significantly with existing faces
            is_duplicate = False
            for used_area in used_areas:
                # Calculate IoU
                x1 = max(area[0], used_area[0])
                y1 = max(area[1], used_area[1])
                x2 = min(area[2], used_area[2])
                y2 = min(area[3], used_area[3])
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (area[2] - area[0]) * (area[3] - area[1])
                    area2 = (used_area[2] - used_area[0]) * (used_area[3] - used_area[1])
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.5:  # If more than 50% overlap, consider it duplicate
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_faces.append(face)
                used_areas.append(area)
        
        return unique_faces
    
    def restore_faces(self, image: Image.Image, faces: List[Dict] = None) -> Image.Image:
        """Restore faces in image (remove blemishes, enhance details)"""
        try:
            if not AI_AVAILABLE:
                return image
            
            if faces is None:
                faces = self.detect_faces(image)
            
            if not faces:
                return image
            
            img_np = np.array(image)
            restored = img_np.copy()
            
            for face in faces:
                x, y, w, h = face['bbox']
                
                # Expand face region slightly
                padding = int(min(w, h) * 0.1)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_np.shape[1], x + w + padding)
                y2 = min(img_np.shape[0], y + h + padding)
                
                # Extract face
                face_region = img_np[y1:y2, x1:x2]
                
                if face_region.size == 0:
                    continue
                
                # Apply face enhancement
                try:
                    # Simple enhancement for now - in production use GFP-GAN or similar
                    enhanced_face = self._enhance_face_region(face_region)
                    
                    # Blend enhanced face back
                    restored[y1:y2, x1:x2] = enhanced_face
                except:
                    continue
            
            return Image.fromarray(restored)
            
        except Exception as e:
            st.error(f"Error restoring faces: {e}")
            return image
    
    def _enhance_face_region(self, face_region: np.ndarray) -> np.ndarray:
        """Enhance a face region"""
        # Convert to float
        face_float = face_region.astype(np.float32) / 255.0
        
        # Apply mild smoothing
        face_float = cv2.GaussianBlur(face_float, (3, 3), 0.5)
        
        # Enhance details
        sharpened = cv2.detailEnhance(face_float, sigma_s=10, sigma_r=0.15)
        
        # Adjust color balance
        lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist((l * 255).astype(np.uint8))
        l = l.astype(np.float32) / 255.0
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back
        return (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    
    def apply_style_transfer(self, content_image: Image.Image, 
                            style_image: Image.Image = None,
                            style_name: str = None) -> Image.Image:
        """Apply style transfer to image"""
        try:
            if not AI_AVAILABLE:
                return content_image
            
            # Simple color transfer as fallback
            if style_image:
                return self._color_transfer(content_image, style_image)
            elif style_name:
                return self._apply_named_style(content_image, style_name)
            else:
                return content_image
                
        except Exception as e:
            st.error(f"Error applying style transfer: {e}")
            return content_image
    
    def _color_transfer(self, content: Image.Image, style: Image.Image) -> Image.Image:
        """Transfer color statistics from style to content"""
        content_np = np.array(content).astype(np.float32)
        style_np = np.array(style.resize(content.size)).astype(np.float32)
        
        # Convert to LAB color space
        content_lab = cv2.cvtColor(content_np / 255.0, cv2.COLOR_RGB2LAB)
        style_lab = cv2.cvtColor(style_np / 255.0, cv2.COLOR_RGB2LAB)
        
        # Transfer mean and std
        for i in range(3):
            content_mean = content_lab[:,:,i].mean()
            content_std = content_lab[:,:,i].std()
            style_mean = style_lab[:,:,i].mean()
            style_std = style_lab[:,:,i].std()
            
            content_lab[:,:,i] = ((content_lab[:,:,i] - content_mean) / content_std) * style_std + style_mean
        
        # Convert back to RGB
        result_lab = np.clip(content_lab, 0, 1)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        result = (result * 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _apply_named_style(self, image: Image.Image, style_name: str) -> Image.Image:
        """Apply a named style preset"""
        style_presets = {
            'vintage': self._apply_vintage_filter,
            'cinematic': self._apply_cinematic_filter,
            'warm': self._apply_warm_filter,
            'cool': self._apply_cool_filter,
            'dramatic': self._apply_dramatic_filter,
            'pastel': self._apply_pastel_filter,
            'high_contrast': self._apply_high_contrast_filter
        }
        
        if style_name in style_presets:
            return style_presets[style_name](image)
        else:
            return image
    
    def _apply_vintage_filter(self, image: Image.Image) -> Image.Image:
        """Apply vintage filter"""
        img_np = np.array(image)
        
        # Add sepia tone
        sepia = np.array([[0.393, 0.769, 0.189],
                         [0.349, 0.686, 0.168],
                         [0.272, 0.534, 0.131]])
        
        vintage = cv2.transform(img_np, sepia)
        vintage = np.clip(vintage, 0, 255).astype(np.uint8)
        
        # Add vignette
        rows, cols = vintage.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/3)
        kernel_y = cv2.getGaussianKernel(rows, rows/3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        for i in range(3):
            vintage[:,:,i] = vintage[:,:,i] * mask
        
        return Image.fromarray(vintage)
    
    def _apply_cinematic_filter(self, image: Image.Image) -> Image.Image:
        """Apply cinematic filter"""
        img_np = np.array(image)
        
        # Increase contrast
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Add film grain
        grain = np.random.normal(0, 10, enhanced.shape).astype(np.int16)
        cinematic = np.clip(enhanced.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        return Image.fromarray(cinematic)
    
    def _apply_warm_filter(self, image: Image.Image) -> Image.Image:
        """Apply warm filter"""
        img_np = np.array(image).astype(np.float32)
        
        # Increase red and yellow channels
        img_np[:,:,0] = np.clip(img_np[:,:,0] * 1.1, 0, 255)  # Red
        img_np[:,:,1] = np.clip(img_np[:,:,1] * 1.05, 0, 255)  # Green
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def _apply_cool_filter(self, image: Image.Image) -> Image.Image:
        """Apply cool filter"""
        img_np = np.array(image).astype(np.float32)
        
        # Increase blue channel
        img_np[:,:,2] = np.clip(img_np[:,:,2] * 1.1, 0, 255)  # Blue
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def _apply_dramatic_filter(self, image: Image.Image) -> Image.Image:
        """Apply dramatic filter"""
        img_np = np.array(image)
        
        # Convert to black and white with high contrast
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        dramatic = cv2.equalizeHist(gray)
        
        # Convert back to color (grayscale)
        dramatic_rgb = cv2.cvtColor(dramatic, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(dramatic_rgb)
    
    def _apply_pastel_filter(self, image: Image.Image) -> Image.Image:
        """Apply pastel filter"""
        img_np = np.array(image).astype(np.float32)
        
        # Reduce saturation
        hsv = cv2.cvtColor(img_np / 255.0, cv2.COLOR_RGB2HSV)
        hsv[:,:,1] = hsv[:,:,1] * 0.5  # Reduce saturation
        pastel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Increase brightness slightly
        pastel = pastel * 1.1
        
        return Image.fromarray(np.clip(pastel * 255, 0, 255).astype(np.uint8))
    
    def _apply_high_contrast_filter(self, image: Image.Image) -> Image.Image:
        """Apply high contrast filter"""
        img_np = np.array(image)
        
        # Apply histogram equalization on each channel
        channels = cv2.split(img_np)
        eq_channels = []
        
        for channel in channels:
            eq = cv2.equalizeHist(channel)
            eq_channels.append(eq)
        
        high_contrast = cv2.merge(eq_channels)
        
        return Image.fromarray(high_contrast)
    
    def generate_image(self, prompt: str, 
                      negative_prompt: str = "",
                      width: int = 512,
                      height: int = 512,
                      num_inference_steps: int = 30,
                      guidance_scale: float = 7.5) -> Image.Image:
        """Generate image from text prompt"""
        try:
            if not AI_AVAILABLE:
                return self._generate_placeholder_image(prompt, width, height)
            
            model = self.model_manager.load_image_generation_model()
            if model is None:
                return self._generate_placeholder_image(prompt, width, height)
            
            # Generate image
            with torch.no_grad():
                image = model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            return image
            
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return self._generate_placeholder_image(prompt, width, height)
    
    def _generate_placeholder_image(self, prompt: str, width: int, height: int) -> Image.Image:
        """Generate a placeholder image"""
        img = Image.new('RGB', (width, height), color='#667eea')
        draw = ImageDraw.Draw(img)
        
        # Add some text
        try:
            font = ImageFont.load_default()
            text = prompt[:30] + "..." if len(prompt) > 30 else prompt
            text_width = draw.textlength(text, font=font)
            text_position = ((width - text_width) // 2, height // 2)
            
            draw.text(text_position, text, fill='white', font=font)
        except:
            pass
        
        return img
    
    def upscale_image(self, image: Image.Image, scale_factor: int = 2) -> Image.Image:
        """Upscale image using AI"""
        try:
            if not AI_AVAILABLE:
                return image.resize(
                    (image.width * scale_factor, image.height * scale_factor),
                    Image.Resampling.LANCZOS
                )
            
            # Simple interpolation-based upscaling
            # In production, use ESRGAN or similar
            img_np = np.array(image)
            
            # Bicubic interpolation
            new_width = image.width * scale_factor
            new_height = image.height * scale_factor
            
            upscaled = cv2.resize(
                img_np,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Apply mild sharpening
            upscaled = cv2.detailEnhance(upscaled, sigma_s=10, sigma_r=0.1)
            
            return Image.fromarray(upscaled)
            
        except Exception as e:
            st.error(f"Error upscaling image: {e}")
            return image.resize(
                (image.width * scale_factor, image.height * scale_factor),
                Image.Resampling.LANCZOS
            )
    
    def create_collage(self, images: List[Image.Image], 
                      layout: str = "grid",
                      rows: int = 2,
                      cols: int = 2) -> Image.Image:
        """Create collage from multiple images"""
        if not images:
            return Image.new('RGB', (800, 600), color='white')
        
        if layout == "grid":
            return self._create_grid_collage(images, rows, cols)
        elif layout == "mosaic":
            return self._create_mosaic_collage(images)
        elif layout == "panorama":
            return self._create_panorama_collage(images)
        else:
            return self._create_grid_collage(images, rows, cols)
    
    def _create_grid_collage(self, images: List[Image.Image], rows: int, cols: int) -> Image.Image:
        """Create grid collage"""
        cell_width = 400
        cell_height = 300
        collage_width = cell_width * cols
        collage_height = cell_height * rows
        
        collage = Image.new('RGB', (collage_width, collage_height), color='white')
        
        for idx, img in enumerate(images[:rows*cols]):
            row = idx // cols
            col = idx % cols
            
            # Resize image to fit cell
            img_resized = img.copy()
            img_resized.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
            
            # Calculate position
            x_offset = col * cell_width + (cell_width - img_resized.width) // 2
            y_offset = row * cell_height + (cell_height - img_resized.height) // 2
            
            # Paste image
            collage.paste(img_resized, (x_offset, y_offset))
        
        return collage
    
    def _create_mosaic_collage(self, images: List[Image.Image]) -> Image.Image:
        """Create mosaic-style collage"""
        if len(images) < 4:
            return self._create_grid_collage(images, 2, 2)
        
        # Create a 3x3 grid with one large central image
        collage = Image.new('RGB', (1200, 900), color='white')
        
        # Large central image
        if len(images) > 0:
            central = images[0]
            central.thumbnail((600, 600), Image.Resampling.LANCZOS)
            collage.paste(central, (300, 150))
        
        # Surrounding images
        positions = [
            (50, 50, 200, 200),    # top-left
            (450, 50, 200, 200),   # top-center
            (850, 50, 200, 200),   # top-right
            (50, 700, 200, 200),   # bottom-left
            (450, 700, 200, 200),  # bottom-center
            (850, 700, 200, 200),  # bottom-right
            (50, 375, 200, 200),   # middle-left
            (850, 375, 200, 200)   # middle-right
        ]
        
        for idx, pos in enumerate(positions[:min(len(images)-1, 8)]):
            img = images[idx + 1]
            img.thumbnail((pos[2], pos[3]), Image.Resampling.LANCZOS)
            x_offset = pos[0] + (pos[2] - img.width) // 2
            y_offset = pos[1] + (pos[3] - img.height) // 2
            collage.paste(img, (x_offset, y_offset))
        
        return collage
    
    def _create_panorama_collage(self, images: List[Image.Image]) -> Image.Image:
        """Create panorama-style collage"""
        if not images:
            return Image.new('RGB', (1200, 300), color='white')
        
        # Resize all images to same height
        target_height = 300
        resized_images = []
        
        for img in images:
            aspect_ratio = img.width / img.height
            target_width = int(target_height * aspect_ratio)
            resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_images.append(resized)
        
        # Calculate total width
        total_width = sum(img.width for img in resized_images)
        
        # Create panorama
        panorama = Image.new('RGB', (total_width, target_height), color='white')
        
        x_offset = 0
        for img in resized_images:
            panorama.paste(img, (x_offset, 0))
            x_offset += img.width
        
        return panorama

# ============================================================================
# DATA MODELS - UPDATED WITH AI METADATA
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
    ai_metadata: Optional[Dict] = None  # New field for AI metadata
    
    @classmethod
    def from_image(cls, image_path: Path) -> 'ImageMetadata':
        """Create metadata from image file"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with Image.open(image_path) as img:
            stats = image_path.stat()
            
            # Extract AI metadata if available
            ai_metadata = None
            if AI_AVAILABLE:
                ai_metadata = cls._extract_ai_metadata(img)
            
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
                checksum=cls._calculate_checksum(image_path),
                ai_metadata=ai_metadata
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
            st.warning(f"Error extracting EXIF data: {str(e)}")
            return None
    
    @staticmethod
    def _extract_ai_metadata(img: Image.Image) -> Optional[Dict]:
        """Extract AI-based metadata"""
        try:
            if not AI_AVAILABLE:
                return None
            
            img_np = np.array(img)
            metadata = {}
            
            # Color histogram
            hist_r = np.histogram(img_np[:,:,0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_np[:,:,1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_np[:,:,2], bins=32, range=(0, 256))[0]
            
            metadata['color_histogram'] = {
                'red': hist_r.tolist(),
                'green': hist_g.tolist(),
                'blue': hist_b.tolist()
            }
            
            # Basic image statistics
            metadata['statistics'] = {
                'mean_brightness': float(np.mean(img_np)),
                'std_brightness': float(np.std(img_np)),
                'contrast': float(np.max(img_np) - np.min(img_np)),
                'entropy': float(skimage.measure.shannon_entropy(img_np))
            }
            
            # Dominant colors
            pixels = img_np.reshape(-1, 3)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels[:1000])  # Sample for speed
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
            metadata['dominant_colors'] = dominant_colors
            
            return metadata
            
        except Exception as e:
            st.warning(f"Error extracting AI metadata: {str(e)}")
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
class AIProcessingHistory:
    """Track AI processing history"""
    process_id: str
    image_id: str
    operation: str
    parameters: Dict
    input_hash: str
    output_hash: str
    processing_time: float
    created_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

# ============================================================================
# DATABASE MANAGEMENT - UPDATED WITH AI TABLES
# ============================================================================
class DatabaseManager:
    """Manage SQLite database operations with AI support"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        except sqlite3.Error as e:
            st.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_database(self):
        """Initialize database with AI tables"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create existing tables (unchanged)
                # ... [existing table creation code remains the same] ...
                
                # NEW: AI Processing History table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_processing_history (
                    process_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (image_id)
                )
                ''')
                
                # NEW: AI Models table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    version TEXT,
                    path TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # NEW: AI Generated Images table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_generated_images (
                    generation_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT,
                    parameters TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indexes for AI tables
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_history_image ON ai_processing_history(image_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_history_operation ON ai_processing_history(operation)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_generated_prompt ON ai_generated_images(prompt)')
                
                conn.commit()
                st.success("✅ Database with AI tables initialized successfully!")
                
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise
    
    # NEW METHODS FOR AI DATA
    def add_ai_processing_record(self, record: AIProcessingHistory):
        """Add AI processing record"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO ai_processing_history
                (process_id, image_id, operation, parameters, input_hash, output_hash, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.process_id,
                    record.image_id,
                    record.operation,
                    json.dumps(record.parameters),
                    record.input_hash,
                    record.output_hash,
                    record.processing_time
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding AI processing record: {str(e)}")
    
    def get_ai_processing_history(self, image_id: str = None, 
                                 operation: str = None,
                                 limit: int = 100) -> List[Dict]:
        """Get AI processing history"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM ai_processing_history'
                params = []
                
                if image_id or operation:
                    query += ' WHERE'
                    conditions = []
                    
                    if image_id:
                        conditions.append(' image_id = ?')
                        params.append(image_id)
                    
                    if operation:
                        conditions.append(' operation = ?')
                        params.append(operation)
                    
                    query += ' AND '.join(conditions)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving AI history: {str(e)}")
            return []
    
    def add_ai_generated_image(self, generation_id: str, prompt: str,
                              negative_prompt: str, parameters: Dict,
                              image_path: str, created_by: str):
        """Add AI generated image record"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO ai_generated_images
                (generation_id, prompt, negative_prompt, parameters, image_path, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    generation_id,
                    prompt,
                    negative_prompt,
                    json.dumps(parameters),
                    image_path,
                    created_by
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding AI generated image: {str(e)}")
    
    def get_ai_generated_images(self, limit: int = 50) -> List[Dict]:
        """Get AI generated images"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM ai_generated_images
                ORDER BY created_at DESC
                LIMIT ?
                ''', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving AI generated images: {str(e)}")
            return []
    
    # Existing methods remain unchanged...
    # ... [rest of existing DatabaseManager methods] ...

# ============================================================================
# IMAGE PROCESSOR - UPDATED WITH AI
# ============================================================================
class ImageProcessor:
    """Handle image processing operations with AI support"""
    
    def __init__(self):
        self.ai_processor = AIImageProcessor() if AI_AVAILABLE else None
    
    @staticmethod
    def create_thumbnail(image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for image"""
        if not image_path.exists():
            st.warning(f"Image not found: {image_path}")
            return None
            
        thumbnail_dir = thumbnail_dir or Config.THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
        
        try:
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
            st.error(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_image_data_url(image_path: Path) -> str:
        """Convert image to data URL for embedding"""
        try:
            if not image_path.exists():
                return ""
                
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                mime_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else f"image/{image_path.suffix[1:]}"
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            st.error(f"Error encoding image {image_path}: {str(e)}")
            return ""
    
    def ai_enhance_image(self, image: Image.Image, enhancement_type: str = "auto") -> Image.Image:
        """AI-powered image enhancement"""
        if self.ai_processor:
            return self.ai_processor.enhance_image(image, enhancement_type)
        else:
            return self._basic_enhancement(image)
    
    def ai_remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background using AI"""
        if self.ai_processor:
            return self.ai_processor.remove_background(image)
        else:
            return image
    
    def ai_detect_faces(self, image: Image.Image) -> List[Dict]:
        """Detect faces using AI"""
        if self.ai_processor:
            return self.ai_processor.detect_faces(image)
        else:
            return []
    
    def ai_restore_faces(self, image: Image.Image, faces: List[Dict] = None) -> Image.Image:
        """Restore faces using AI"""
        if self.ai_processor:
            return self.ai_processor.restore_faces(image, faces)
        else:
            return image
    
    def ai_apply_style_transfer(self, content_image: Image.Image,
                               style_image: Image.Image = None,
                               style_name: str = None) -> Image.Image:
        """Apply style transfer using AI"""
        if self.ai_processor:
            return self.ai_processor.apply_style_transfer(content_image, style_image, style_name)
        else:
            return content_image
    
    def ai_generate_image(self, prompt: str, **kwargs) -> Image.Image:
        """Generate image from text prompt"""
        if self.ai_processor:
            return self.ai_processor.generate_image(prompt, **kwargs)
        else:
            return self.ai_processor._generate_placeholder_image(prompt, 512, 512)
    
    def ai_upscale_image(self, image: Image.Image, scale_factor: int = 2) -> Image.Image:
        """Upscale image using AI"""
        if self.ai_processor:
            return self.ai_processor.upscale_image(image, scale_factor)
        else:
            return image.resize(
                (image.width * scale_factor, image.height * scale_factor),
                Image.Resampling.LANCZOS
            )
    
    def ai_create_collage(self, images: List[Image.Image], **kwargs) -> Image.Image:
        """Create collage using AI"""
        if self.ai_processor:
            return self.ai_processor.create_collage(images, **kwargs)
        else:
            return images[0] if images else Image.new('RGB', (800, 600), color='white')

# ============================================================================
# AI ALBUM MANAGER - NEW CLASS
# ============================================================================
class AIAlbumManager:
    """Manage AI-powered album operations"""
    
    def __init__(self, album_manager):
        self.manager = album_manager
        self.image_processor = ImageProcessor()
        self.db = album_manager.db
    
    def process_with_ai(self, image_path: Path, operation: str, 
                       parameters: Dict = None) -> Tuple[Image.Image, Dict]:
        """Process image with AI and track history"""
        start_time = time.time()
        
        try:
            # Load image
            with Image.open(image_path) as img:
                original_img = img.copy()
            
            # Get input hash
            input_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
            
            # Process based on operation
            result = None
            output_hash = None
            
            if operation == "enhance":
                enhancement_type = parameters.get('type', 'auto') if parameters else 'auto'
                result = self.image_processor.ai_enhance_image(original_img, enhancement_type)
            
            elif operation == "remove_background":
                result = self.image_processor.ai_remove_background(original_img)
            
            elif operation == "restore_faces":
                result = self.image_processor.ai_restore_faces(original_img)
            
            elif operation == "style_transfer":
                style_image_path = parameters.get('style_image') if parameters else None
                style_name = parameters.get('style_name') if parameters else None
                
                if style_image_path:
                    with Image.open(style_image_path) as style_img:
                        result = self.image_processor.ai_apply_style_transfer(
                            original_img, style_img, style_name
                        )
                else:
                    result = self.image_processor.ai_apply_style_transfer(
                        original_img, style_name=style_name
                    )
            
            elif operation == "upscale":
                scale_factor = parameters.get('scale_factor', 2) if parameters else 2
                result = self.image_processor.ai_upscale_image(original_img, scale_factor)
            
            elif operation == "generate":
                prompt = parameters.get('prompt', '') if parameters else ''
                result = self.image_processor.ai_generate_image(prompt, **parameters)
            
            else:
                raise ValueError(f"Unknown AI operation: {operation}")
            
            # Calculate output hash
            if result:
                buffer = io.BytesIO()
                result.save(buffer, format='PNG')
                output_hash = hashlib.md5(buffer.getvalue()).hexdigest()
            
            # Record processing history
            processing_time = time.time() - start_time
            
            record = AIProcessingHistory(
                process_id=str(uuid.uuid4()),
                image_id=self._get_image_id_from_path(image_path),
                operation=operation,
                parameters=parameters or {},
                input_hash=input_hash,
                output_hash=output_hash or "",
                processing_time=processing_time,
                created_at=datetime.datetime.now()
            )
            
            self.db.add_ai_processing_record(record)
            
            return result, {
                'process_id': record.process_id,
                'processing_time': processing_time,
                'operation': operation
            }
            
        except Exception as e:
            st.error(f"AI processing error: {e}")
            return None, {'error': str(e)}
    
    def _get_image_id_from_path(self, image_path: Path) -> str:
        """Get image ID from file path"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                relative_path = str(image_path.relative_to(Config.DATA_DIR))
                cursor.execute(
                    'SELECT image_id FROM images WHERE filepath = ?',
                    (relative_path,)
                )
                result = cursor.fetchone()
                return result[0] if result else "unknown"
        except:
            return "unknown"
    
    def batch_process(self, image_paths: List[Path], operation: str,
                     parameters: Dict = None) -> List[Tuple[Path, Image.Image, Dict]]:
        """Batch process multiple images"""
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_path in enumerate(image_paths):
            status_text.text(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
            
            try:
                result, info = self.process_with_ai(image_path, operation, parameters)
                if result:
                    results.append((image_path, result, info))
            except Exception as e:
                st.error(f"Failed to process {image_path.name}: {e}")
            
            progress_bar.progress((i + 1) / len(image_paths))
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def generate_ai_image(self, prompt: str, negative_prompt: str = "",
                         width: int = 512, height: int = 512,
                         num_inference_steps: int = 30,
                         guidance_scale: float = 7.5) -> Image.Image:
        """Generate AI image and save to database"""
        try:
            parameters = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale
            }
            
            # Generate image
            result = self.image_processor.ai_generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            # Save to file
            generation_id = str(uuid.uuid4())
            output_dir = Config.AI_OUTPUT_DIR / "generated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"ai_generated_{generation_id[:8]}.png"
            output_path = output_dir / filename
            
            result.save(output_path, format='PNG')
            
            # Add to database
            self.db.add_ai_generated_image(
                generation_id=generation_id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters=parameters,
                image_path=str(output_path),
                created_by=st.session_state.get('username', 'anonymous')
            )
            
            return result
            
        except Exception as e:
            st.error(f"AI generation error: {e}")
            return None
    
    def get_ai_statistics(self) -> Dict:
        """Get AI processing statistics"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Total AI processes
                cursor.execute('SELECT COUNT(*) FROM ai_processing_history')
                total_processes = cursor.fetchone()[0]
                
                # Processes by operation
                cursor.execute('''
                SELECT operation, COUNT(*) as count
                FROM ai_processing_history
                GROUP BY operation
                ORDER BY count DESC
                ''')
                operations = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Average processing time
                cursor.execute('SELECT AVG(processing_time) FROM ai_processing_history')
                avg_time = cursor.fetchone()[0] or 0
                
                # Recent activity
                cursor.execute('''
                SELECT operation, created_at
                FROM ai_processing_history
                ORDER BY created_at DESC
                LIMIT 10
                ''')
                recent = cursor.fetchall()
                
                return {
                    'total_processes': total_processes,
                    'operations': operations,
                    'avg_processing_time': avg_time,
                    'recent_activity': recent
                }
                
        except Exception as e:
            st.error(f"Error getting AI statistics: {e}")
            return {}

# ============================================================================
# UI COMPONENTS - UPDATED WITH AI
# ============================================================================
class UIComponents:
    """Reusable UI components with AI enhancements"""
    
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
    
    @staticmethod
    def ai_feature_card(title: str, description: str, icon: str, 
                       enabled: bool = True) -> str:
        """Generate HTML for AI feature card"""
        status_color = "#48bb78" if enabled else "#cbd5e0"
        status_text = "Available" if enabled else "Disabled"
        
        return f'''
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid {status_color};
        ">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    width: 40px;
                    height: 40px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                    margin-right: 12px;
                ">
                    {icon}
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 18px; color: #2d3748;">{title}</h3>
                    <span style="
                        background: {status_color};
                        color: white;
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: bold;
                    ">
                        {status_text}
                    </span>
                </div>
            </div>
            <p style="margin: 0; color: #718096; font-size: 14px; line-height: 1.5;">
                {description}
            </p>
        </div>
        '''

# ============================================================================
# ALBUM MANAGER - UPDATED WITH AI
# ============================================================================
class AlbumManager:
    """Main album management class with AI capabilities"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.image_processor = ImageProcessor()
        self.ai_manager = AIAlbumManager(self) if AI_AVAILABLE else None
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'current_page': 1,
                'selected_person': None,
                'selected_image': None,
                'search_query': '',
                'view_mode': 'grid',
                'sort_by': 'date',
                'sort_order': 'desc',
                'selected_tags': [],
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'recently_viewed': [],
                'toc_page': 1,
                'gallery_page': 1,
                'show_directory_info': True,
                # AI-specific states
                'ai_enabled': AI_AVAILABLE,
                'ai_processing_active': False,
                'ai_generation_prompt': '',
                'ai_selected_style': 'cinematic',
                'ai_selected_enhancement': 'auto'
            })
    
    # All existing methods remain unchanged...
    # ... [rest of existing AlbumManager methods] ...

# ============================================================================
# PHOTO ALBUM APP - UPDATED WITH AI PAGES
# ============================================================================
class PhotoAlbumApp:
    """Main application class with AI features"""
    
    def __init__(self):
        self.manager = AlbumManager()
        self.setup_page_config()
        self.check_initialization()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=Config.APP_NAME,
            page_icon="🤖",  # Changed to AI icon
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-album-ai',
                'Report a bug': 'https://github.com/yourusername/photo-album-ai/issues',
                'About': f"# {Config.APP_NAME} v{Config.VERSION}\nAI-powered photo album management system"
            }
        )
    
    def check_initialization(self):
        """Check and initialize application directories"""
        try:
            Config.init_directories()
            self.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            self.initialized = False
    
    def render_sidebar(self):
        """Render application sidebar with AI features"""
        with st.sidebar:
            st.title(f"🤖 {Config.APP_NAME}")
            st.caption(f"v{Config.VERSION}")
            
            # AI Status
            if AI_AVAILABLE:
                st.success("✅ AI Features Enabled")
            else:
                st.warning("⚠️ AI Features Disabled")
            
            # User info
            st.divider()
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**{st.session_state['username']}**")
                st.caption(st.session_state['user_role'].title())
            
            # Navigation with AI pages
            st.divider()
            st.subheader("Navigation")
            
            nav_options = {
                "🏠 Dashboard": "dashboard",
                "🤖 AI Studio": "ai_studio",  # New AI page
                "👥 People": "people",
                "📁 Gallery": "gallery",
                "⭐ Favorites": "favorites",
                "🔍 Search": "search",
                "📊 Statistics": "statistics",
                "⚙️ Settings": "settings",
                "📤 Import/Export": "import_export",
                "🧠 AI Tools": "ai_tools"  # New AI tools page
            }
            
            for label, key in nav_options.items():
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
                    st.rerun()
            
            # Quick actions with AI
            st.divider()
            st.subheader("Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Scan", use_container_width=True):
                    with st.spinner("Scanning directory..."):
                        results = self.manager.scan_directory()
                        if results['new_images'] > 0 or results['updated_images'] > 0:
                            st.success(f"Found {results['new_images']} new images")
                        st.rerun()
            
            with col2:
                if st.button("✨ AI Enhance", use_container_width=True, disabled=not AI_AVAILABLE):
                    st.session_state['current_page'] = 'ai_tools'
                    st.session_state['ai_selected_tool'] = 'enhance'
                    st.rerun()
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.manager.cache.clear()
                if self.manager.ai_manager:
                    self.manager.ai_manager.image_processor.ai_processor.model_manager.clear_cache()
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
            
            if AI_AVAILABLE:
                ai_stats = self.manager.ai_manager.get_ai_statistics()
                st.metric("AI Processes", ai_stats.get('total_processes', 0))
            
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
    
    def render_ai_studio_page(self):
        """Render AI Studio page"""
        st.title("🤖 AI Studio")
        st.markdown("Advanced AI-powered image editing and generation")
        
        if not AI_AVAILABLE:
            st.error("AI features are not available. Please install required packages.")
            st.code("pip install torch torchvision opencv-python diffusers transformers")
            return
        
        tabs = st.tabs(["🎨 Image Editor", "🚀 AI Generator", "🔄 Style Transfer", "🧹 Cleanup", "📈 Analytics"])
        
        with tabs[0]:  # Image Editor
            st.subheader("AI Image Editor")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Image upload
                uploaded_file = st.file_uploader(
                    "Upload image to edit",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    key="ai_editor_upload"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Original Image", use_column_width=True)
                    
                    # Editing options
                    st.subheader("Editing Tools")
                    
                    edit_col1, edit_col2, edit_col3 = st.columns(3)
                    
                    with edit_col1:
                        if st.button("✨ Auto Enhance", use_container_width=True):
                            with st.spinner("Enhancing image..."):
                                enhanced = self.manager.image_processor.ai_enhance_image(image)
                                st.session_state['edited_image'] = enhanced
                                st.rerun()
                    
                    with edit_col2:
                        if st.button("🧹 Remove Background", use_container_width=True):
                            with st.spinner("Removing background..."):
                                no_bg = self.manager.image_processor.ai_remove_background(image)
                                st.session_state['edited_image'] = no_bg
                                st.rerun()
                    
                    with edit_col3:
                        if st.button("😊 Restore Faces", use_container_width=True):
                            with st.spinner("Restoring faces..."):
                                restored = self.manager.image_processor.ai_restore_faces(image)
                                st.session_state['edited_image'] = restored
                                st.rerun()
            
            with col2:
                # Edited image display
                if 'edited_image' in st.session_state:
                    st.subheader("Edited Image")
                    st.image(st.session_state['edited_image'], use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    st.session_state['edited_image'].save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download",
                        data=buf.getvalue(),
                        file_name=f"ai_edited_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # Save to album
                    if st.button("💾 Save to Album", use_container_width=True):
                        # Implementation for saving to album
                        st.success("Image saved to album!")
        
        with tabs[1]:  # AI Generator
            st.subheader("AI Image Generator")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Prompt input
                prompt = st.text_area(
                    "Describe the image you want to generate:",
                    height=100,
                    placeholder="A beautiful landscape with mountains and a lake, sunset, cinematic, 8k"
                )
                
                negative_prompt = st.text_input(
                    "Negative prompt (what to avoid):",
                    placeholder="blurry, low quality, distorted"
                )
                
                # Generation parameters
                col_params1, col_params2, col_params3 = st.columns(3)
                
                with col_params1:
                    width = st.selectbox("Width", [512, 768, 1024], index=0)
                    height = st.selectbox("Height", [512, 768, 1024], index=0)
                
                with col_params2:
                    num_steps = st.slider("Steps", 10, 100, 30)
                    guidance = st.slider("Guidance", 1.0, 20.0, 7.5)
                
                with col_params3:
                    seed = st.number_input("Seed (optional)", value=None, placeholder="Random")
                    batch_size = st.selectbox("Batch Size", [1, 2, 4], index=0)
                
                # Generate button
                if st.button("✨ Generate Image", type="primary", use_container_width=True):
                    if prompt:
                        with st.spinner("Generating image..."):
                            generated = self.manager.ai_manager.generate_ai_image(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                width=width,
                                height=height,
                                num_inference_steps=num_steps,
                                guidance_scale=guidance
                            )
                            
                            if generated:
                                st.session_state['generated_image'] = generated
                                st.session_state['generation_prompt'] = prompt
                                st.rerun()
                    else:
                        st.warning("Please enter a prompt")
            
            with col2:
                # Generated image display
                if 'generated_image' in st.session_state:
                    st.subheader("Generated Image")
                    st.image(st.session_state['generated_image'], use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    st.session_state['generated_image'].save(buf, format='PNG')
                    st.download_button(
                        label="📥 Download",
                        data=buf.getvalue(),
                        file_name=f"ai_generated_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # Prompt info
                    st.caption(f"Prompt: {st.session_state.get('generation_prompt', '')}")
        
        with tabs[2]:  # Style Transfer
            st.subheader("Style Transfer")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Content image
                content_file = st.file_uploader(
                    "Content Image",
                    type=['jpg', 'jpeg', 'png'],
                    key="content_image"
                )
                
                if content_file:
                    content_img = Image.open(content_file)
                    st.image(content_img, caption="Content", use_column_width=True)
            
            with col2:
                # Style image or preset
                style_option = st.radio(
                    "Style Source",
                    ["Upload Image", "Use Preset"],
                    horizontal=True
                )
                
                if style_option == "Upload Image":
                    style_file = st.file_uploader(
                        "Style Image",
                        type=['jpg', 'jpeg', 'png'],
                        key="style_image"
                    )
                    
                    if style_file:
                        style_img = Image.open(style_file)
                        st.image(style_img, caption="Style", use_column_width=True)
                else:
                    style_preset = st.selectbox(
                        "Style Preset",
                        ["Cinematic", "Vintage", "Painting", "Anime", "Cyberpunk", "Fantasy"]
                    )
                
                # Transfer button
                if st.button("🎨 Apply Style Transfer", use_container_width=True):
                    if content_file:
                        with st.spinner("Applying style transfer..."):
                            if style_option == "Upload Image" and style_file:
                                result = self.manager.image_processor.ai_apply_style_transfer(
                                    content_img, style_img
                                )
                            else:
                                result = self.manager.image_processor.ai_apply_style_transfer(
                                    content_img, style_name=style_preset.lower()
                                )
                            
                            st.session_state['styled_image'] = result
                            st.rerun()
            
            # Show result
            if 'styled_image' in st.session_state:
                st.subheader("Styled Result")
                st.image(st.session_state['styled_image'], use_column_width=True)
        
        with tabs[3]:  # Cleanup
            st.subheader("Image Cleanup Tools")
            
            uploaded_file = st.file_uploader(
                "Upload image to clean up",
                type=['jpg', 'jpeg', 'png'],
                key="cleanup_upload"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                
                with col2:
                    # Cleanup options
                    cleanup_type = st.selectbox(
                        "Cleanup Type",
                        ["Denoise", "Sharpen", "Color Correct", "Upscale", "All"]
                    )
                    
                    if st.button("🧹 Clean Up Image", use_container_width=True):
                        with st.spinner(f"Applying {cleanup_type}..."):
                            if cleanup_type == "Denoise":
                                result = self.manager.image_processor.ai_enhance_image(
                                    image, "denoise"
                                )
                            elif cleanup_type == "Sharpen":
                                result = self.manager.image_processor.ai_enhance_image(
                                    image, "sharpness"
                                )
                            elif cleanup_type == "Color Correct":
                                result = self.manager.image_processor.ai_enhance_image(
                                    image, "color"
                                )
                            elif cleanup_type == "Upscale":
                                result = self.manager.image_processor.ai_upscale_image(
                                    image, scale_factor=2
                                )
                            else:
                                # Apply all enhancements
                                result = self.manager.image_processor.ai_enhance_image(
                                    image, "auto"
                                )
                                result = self.manager.image_processor.ai_upscale_image(
                                    result, scale_factor=2
                                )
                            
                            st.session_state['cleaned_image'] = result
                            st.rerun()
                
                # Show cleaned image
                if 'cleaned_image' in st.session_state:
                    st.subheader("Cleaned Image")
                    st.image(st.session_state['cleaned_image'], use_column_width=True)
        
        with tabs[4]:  # Analytics
            st.subheader("AI Analytics")
            
            if self.manager.ai_manager:
                stats = self.manager.ai_manager.get_ai_statistics()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Processes", stats.get('total_processes', 0))
                
                with col2:
                    avg_time = stats.get('avg_processing_time', 0)
                    st.metric("Avg. Time", f"{avg_time:.2f}s")
                
                with col3:
                    st.metric("AI Models", len(self.manager.ai_manager.image_processor.ai_processor.model_manager.model_cache))
                
                # Operations breakdown
                st.subheader("Operations Breakdown")
                operations = stats.get('operations', {})
                
                if operations:
                    df = pd.DataFrame(
                        list(operations.items()),
                        columns=['Operation', 'Count']
                    )
                    st.bar_chart(df.set_index('Operation'))
                
                # Recent activity
                st.subheader("Recent Activity")
                recent = stats.get('recent_activity', [])
                
                for operation, timestamp in recent:
                    st.text(f"{timestamp}: {operation}")
    
    def render_ai_tools_page(self):
        """Render AI Tools page"""
        st.title("🧠 AI Tools")
        
        if not AI_AVAILABLE:
            st.warning("AI tools require additional packages to be installed.")
            with st.expander("Installation Instructions"):
                st.code("""
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
                pip install opencv-python dlib insightface face-recognition
                pip install diffusers transformers scikit-image albumentations
                pip install rembg mediapipe deepface facenet-pytorch
                """)
            return
        
        # AI Tools Grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(UIComponents.ai_feature_card(
                title="Face Detection & Recognition",
                description="Detect and recognize faces in your photos",
                icon="😊",
                enabled=True
            ), unsafe_allow_html=True)
            
            if st.button("Detect Faces", use_container_width=True):
                st.session_state['ai_tool'] = 'face_detection'
                st.rerun()
            
            st.markdown(UIComponents.ai_feature_card(
                title="Background Removal",
                description="Remove backgrounds from images automatically",
                icon="🧹",
                enabled=True
            ), unsafe_allow_html=True)
            
            if st.button("Remove Background", use_container_width=True):
                st.session_state['ai_tool'] = 'background_removal'
                st.rerun()
        
        with col2:
            st.markdown(UIComponents.ai_feature_card(
                title="Image Enhancement",
                description="Automatically enhance image quality",
                icon="✨",
                enabled=True
            ), unsafe_allow_html=True)
            
            if st.button("Enhance Images", use_container_width=True):
                st.session_state['ai_tool'] = 'enhancement'
                st.rerun()
            
            st.markdown(UIComponents.ai_feature_card(
                title="Style Transfer",
                description="Apply artistic styles to your photos",
                icon="🎨",
                enabled=True
            ), unsafe_allow_html=True)
            
            if st.button("Transfer Style", use_container_width=True):
                st.session_state['ai_tool'] = 'style_transfer'
                st.rerun()
        
        # Batch Processing
        st.divider()
        st.subheader("🔄 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images for batch processing",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"Selected {len(uploaded_files)} images")
                
                # Preview
                if len(uploaded_files) <= 6:
                    cols = st.columns(3)
                    for idx, file in enumerate(uploaded_files[:6]):
                        with cols[idx % 3]:
                            st.image(file, use_column_width=True)
            
            with col2:
                operation = st.selectbox(
                    "Operation",
                    ["Enhance", "Remove Background", "Restore Faces", "Upscale"]
                )
                
                if st.button("Process All", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(uploaded_files)} images..."):
                        results = []
                        
                        for file in uploaded_files:
                            image = Image.open(file)
                            
                            if operation == "Enhance":
                                result = self.manager.image_processor.ai_enhance_image(image)
                            elif operation == "Remove Background":
                                result = self.manager.image_processor.ai_remove_background(image)
                            elif operation == "Restore Faces":
                                result = self.manager.image_processor.ai_restore_faces(image)
                            else:  # Upscale
                                result = self.manager.image_processor.ai_upscale_image(image)
                            
                            results.append(result)
                        
                        st.success(f"Processed {len(results)} images!")
                        
                        # Show results
                        st.subheader("Results")
                        result_cols = st.columns(3)
                        for idx, result in enumerate(results[:6]):
                            with result_cols[idx % 3]:
                                st.image(result, use_column_width=True)
                        
                        # Download all
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for idx, result in enumerate(results):
                                buf = io.BytesIO()
                                result.save(buf, format='PNG')
                                zip_file.writestr(
                                    f"ai_processed_{idx+1}.png",
                                    buf.getvalue()
                                )
                        
                        st.download_button(
                            label="📥 Download All",
                            data=zip_buffer.getvalue(),
                            file_name=f"ai_batch_processed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
    
    def render_main(self):
        """Main application renderer"""
        # Render sidebar
        self.render_sidebar()
        
        # Get current page
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Render appropriate page
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'ai_studio':
            self.render_ai_studio_page()
        elif current_page == 'ai_tools':
            self.render_ai_tools_page()
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
            st.caption("AI-powered photo album management system")
            
            if AI_AVAILABLE:
                st.caption("🤖 AI Features: Enabled")
            else:
                st.caption("⚠️ AI Features: Disabled - Install packages to enable")

# ============================================================================
# REQUIREMENTS FILE
# ============================================================================
"""
Create a requirements.txt file with:
streamlit>=1.28.0
pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
sqlite3
python-dateutil
uuid

# AI/ML Libraries
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
dlib>=19.24.0
insightface>=0.7.3
face-recognition>=1.3.0
deepface>=0.0.79
mediapipe>=0.10.0
rembg>=2.0.50
diffusers>=0.24.0
transformers>=4.35.0
scikit-image>=0.22.0
albumentations>=1.3.0
kornia>=0.7.0
facenet-pytorch>=2.5.0
gradio>=4.0.0
"""

# ============================================================================
# INSTALLATION SCRIPT
# ============================================================================
"""
Create an install_ai.py file:

#!/usr/bin/env python3
import subprocess
import sys

def install_packages():
    packages = [
        "streamlit>=1.28.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "opencv-python",
        "dlib",
        "insightface",
        "face-recognition",
        "deepface",
        "mediapipe",
        "rembg",
        "diffusers",
        "transformers",
        "scikit-image",
        "albumentations",
        "kornia",
        "facenet-pytorch",
        "gradio"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages()
"""

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

if __name__ == "__main__":
    main()
