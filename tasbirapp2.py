# ============================================================================
# IMPORTS - ADDITIONAL MODULES FOR AI FEATURES
# ============================================================================
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
import replicate
import openai
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from PIL import ImageFilter, ImageEnhance
import cv2
import numpy as np
from io import BytesIO
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AI MODELS CONFIGURATION
# ============================================================================
class AIConfig:
    """Configuration for AI models"""
    
    # Model paths and names
    CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
    TAG_MODEL = "microsoft/beit-large-patch16-224"
    CLIP_MODEL = "openai/clip-vit-large-patch14"
    DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
    INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
    
    # API Keys (store in secrets or environment variables)
    OPENAI_API_KEY = None
    REPLICATE_API_TOKEN = None
    HUGGINGFACE_TOKEN = None
    
    # Model settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRECISION = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Cache settings for AI models
    AI_CACHE_TTL = 3600 * 24  # 24 hours
    
    @classmethod
    def load_secrets(cls):
        """Load API keys from Streamlit secrets or environment variables"""
        try:
            import os
            cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            cls.REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN") or st.secrets.get("REPLICATE_API_TOKEN")
            cls.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or st.secrets.get("HUGGINGFACE_TOKEN")
        except:
            pass

# ============================================================================
# AI MODEL MANAGER WITH CACHING
# ============================================================================
class AIModelManager:
    """Manage AI models with caching and lazy loading"""
    
    def __init__(self):
        self.models_loaded = False
        self.model_cache = {}
        self.device = AIConfig.DEVICE
        AIConfig.load_secrets()
        
    def _get_cache_key(self, model_name, task):
        """Generate cache key for model"""
        return f"{model_name}_{task}"
    
    def load_model(self, model_name, task="default"):
        """Load model with caching"""
        cache_key = self._get_cache_key(model_name, task)
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            with st.spinner(f"Loading {model_name}..."):
                if model_name == "caption":
                    processor = BlipProcessor.from_pretrained(AIConfig.CAPTION_MODEL)
                    model = BlipForConditionalGeneration.from_pretrained(
                        AIConfig.CAPTION_MODEL,
                        torch_dtype=AIConfig.PRECISION
                    ).to(self.device)
                    self.model_cache[cache_key] = (processor, model)
                    
                elif model_name == "clip":
                    processor = CLIPProcessor.from_pretrained(AIConfig.CLIP_MODEL)
                    model = CLIPModel.from_pretrained(AIConfig.CLIP_MODEL).to(self.device)
                    self.model_cache[cache_key] = (processor, model)
                    
                elif model_name == "diffusion":
                    pipe = StableDiffusionPipeline.from_pretrained(
                        AIConfig.DIFFUSION_MODEL,
                        torch_dtype=AIConfig.PRECISION,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                    self.model_cache[cache_key] = pipe
                    
                elif model_name == "inpaint":
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        AIConfig.INPAINT_MODEL,
                        torch_dtype=AIConfig.PRECISION,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    self.model_cache[cache_key] = pipe
                    
                elif model_name == "img2img":
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        AIConfig.DIFFUSION_MODEL,
                        torch_dtype=AIConfig.PRECISION,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    self.model_cache[cache_key] = pipe
                    
            return self.model_cache[cache_key]
            
        except Exception as e:
            st.error(f"Error loading model {model_name}: {str(e)}")
            return None
    
    def unload_model(self, model_name, task="default"):
        """Unload model from cache"""
        cache_key = self._get_cache_key(model_name, task)
        if cache_key in self.model_cache:
            del self.model_cache[cache_key]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def clear_all_models(self):
        """Clear all loaded models"""
        self.model_cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================================
# LLM-BASED IMAGE ENHANCEMENT
# ============================================================================
class LLMImageEnhancer:
    """LLM-based image analysis and enhancement"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for LLM features"""
        if 'llm_captions' not in st.session_state:
            st.session_state.llm_captions = {}
        if 'llm_tags' not in st.session_state:
            st.session_state.llm_tags = {}
        if 'llm_descriptions' not in st.session_state:
            st.session_state.llm_descriptions = {}
        if 'llm_analysis_cache' not in st.session_state:
            st.session_state.llm_analysis_cache = {}
    
    def generate_caption(self, image: Image.Image, max_length: int = 30) -> str:
        """Generate caption for image using BLIP model"""
        try:
            # Create cache key
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"caption_{image_hash}_{max_length}"
            
            # Check cache
            if cache_key in st.session_state.llm_analysis_cache:
                return st.session_state.llm_analysis_cache[cache_key]
            
            # Load model
            processor, model = self.model_manager.load_model("caption")
            if processor is None or model is None:
                return "Could not load caption model"
            
            # Prepare image
            inputs = processor(image, return_tensors="pt").to(self.model_manager.device)
            
            # Generate caption
            with torch.no_grad():
                out = model.generate(**inputs, max_length=max_length, num_beams=5)
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # Cache result
            st.session_state.llm_analysis_cache[cache_key] = caption
            st.session_state.llm_captions[image_hash] = caption
            
            return caption
            
        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            return "Error generating caption"
    
    def generate_detailed_description(self, image: Image.Image) -> str:
        """Generate detailed description using CLIP and text generation"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"description_{image_hash}"
            
            if cache_key in st.session_state.llm_analysis_cache:
                return st.session_state.llm_analysis_cache[cache_key]
            
            # Get initial caption
            caption = self.generate_caption(image, max_length=50)
            
            # Use CLIP to get image features
            processor, model = self.model_manager.load_model("clip")
            if processor is None or model is None:
                return caption
            
            inputs = processor(images=image, return_tensors="pt").to(self.model_manager.device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # Generate detailed description based on features
            description_prompts = [
                f"Describe this image in detail: {caption}",
                f"What can you see in this image? {caption}",
                f"Provide a comprehensive description of this photo: {caption}"
            ]
            
            # Simulate enhanced description (in production, use GPT or similar)
            enhanced_desc = caption
            if "person" in caption.lower():
                enhanced_desc += ". The photo shows a person in the scene."
            if "landscape" in caption.lower() or "nature" in caption.lower():
                enhanced_desc += " This appears to be an outdoor scene with natural elements."
            
            # Cache result
            st.session_state.llm_analysis_cache[cache_key] = enhanced_desc
            st.session_state.llm_descriptions[image_hash] = enhanced_desc
            
            return enhanced_desc
            
        except Exception as e:
            st.error(f"Error generating description: {str(e)}")
            return "Error generating description"
    
    def generate_tags(self, image: Image.Image, num_tags: int = 10) -> List[str]:
        """Generate tags for image"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"tags_{image_hash}_{num_tags}"
            
            if cache_key in st.session_state.llm_analysis_cache:
                return st.session_state.llm_analysis_cache[cache_key]
            
            # Get caption
            caption = self.generate_caption(image)
            
            # Extract keywords from caption
            words = caption.lower().split()
            
            # Common photo tags
            common_tags = [
                'photo', 'image', 'picture', 'photography',
                'color', 'black and white', 'vintage', 'modern',
                'portrait', 'landscape', 'urban', 'nature',
                'day', 'night', 'sunset', 'sunrise',
                'indoor', 'outdoor', 'travel', 'memory'
            ]
            
            # Generate tags based on caption
            tags = set()
            
            # Add words from caption
            for word in words:
                if len(word) > 3 and word not in ['this', 'that', 'with', 'from']:
                    tags.add(word)
            
            # Add relevant common tags
            for common_tag in common_tags:
                if any(word in common_tag for word in words[:5]):
                    tags.add(common_tag)
            
            # Limit number of tags
            tags_list = list(tags)[:num_tags]
            
            # Cache result
            st.session_state.llm_analysis_cache[cache_key] = tags_list
            st.session_state.llm_tags[image_hash] = tags_list
            
            return tags_list
            
        except Exception as e:
            st.error(f"Error generating tags: {str(e)}")
            return []
    
    def analyze_image_composition(self, image: Image.Image) -> Dict:
        """Analyze image composition and provide feedback"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"composition_{image_hash}"
            
            if cache_key in st.session_state.llm_analysis_cache:
                return st.session_state.llm_analysis_cache[cache_key]
            
            analysis = {
                'brightness': 'optimal',
                'contrast': 'good',
                'sharpness': 'acceptable',
                'color_balance': 'balanced',
                'composition': 'standard',
                'suggestions': []
            }
            
            # Basic image analysis
            img_array = np.array(image)
            
            # Brightness analysis
            brightness = np.mean(img_array)
            if brightness < 50:
                analysis['brightness'] = 'dark'
                analysis['suggestions'].append('Consider increasing brightness')
            elif brightness > 200:
                analysis['brightness'] = 'bright'
                analysis['suggestions'].append('Consider decreasing brightness')
            
            # Contrast analysis
            contrast = np.std(img_array)
            if contrast < 30:
                analysis['contrast'] = 'low'
                analysis['suggestions'].append('Consider increasing contrast')
            
            # Color balance analysis
            if len(img_array.shape) == 3:  # Color image
                r, g, b = np.mean(img_array, axis=(0, 1))
                if abs(r - g) > 30 or abs(r - b) > 30:
                    analysis['color_balance'] = 'unbalanced'
                    analysis['suggestions'].append('Consider adjusting color balance')
            
            # Rule of thirds analysis (simplified)
            height, width = img_array.shape[:2]
            analysis['composition'] = f"{width}x{height}"
            
            # Cache result
            st.session_state.llm_analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing composition: {str(e)}")
            return {}
    
    def generate_ai_caption_variants(self, image: Image.Image, num_variants: int = 3) -> List[str]:
        """Generate multiple caption variants"""
        try:
            base_caption = self.generate_caption(image)
            
            variants = [base_caption]
            
            # Create variants by modifying the base caption
            if num_variants > 1:
                # Short version
                short_words = base_caption.split()[:8]
                variants.append(' '.join(short_words) + '...')
                
                # Creative version
                creative = base_caption.replace('a ', 'an artistic ').replace('the ', 'the stunning ')
                variants.append(creative)
                
                # Descriptive version
                descriptive = f"A captivating photo showing {base_caption.lower()}"
                variants.append(descriptive)
            
            return variants[:num_variants]
            
        except Exception as e:
            st.error(f"Error generating caption variants: {str(e)}")
            return [base_caption] if 'base_caption' in locals() else ["No caption available"]

# ============================================================================
# DIFFUSION-BASED IMAGE MODIFICATION
# ============================================================================
class DiffusionImageStyler:
    """Diffusion model-based image styling and modification"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for diffusion features"""
        if 'diffusion_results' not in st.session_state:
            st.session_state.diffusion_results = {}
        if 'style_presets' not in st.session_state:
            st.session_state.style_presets = self._get_default_presets()
        if 'generation_history' not in st.session_state:
            st.session_state.generation_history = []
        if 'current_modification' not in st.session_state:
            st.session_state.current_modification = None
    
    def _get_default_presets(self):
        """Get default style presets"""
        return {
            'realistic': {
                'prompt_suffix': 'photorealistic, 8k, detailed, sharp focus',
                'negative_prompt': 'blurry, cartoon, painting, drawing',
                'guidance_scale': 7.5,
                'num_inference_steps': 30
            },
            'painting': {
                'prompt_suffix': 'oil painting, masterpiece, art gallery',
                'negative_prompt': 'photograph, photo, realistic',
                'guidance_scale': 8.0,
                'num_inference_steps': 40
            },
            'sketch': {
                'prompt_suffix': 'pencil sketch, drawing, black and white',
                'negative_prompt': 'color, painting, photograph',
                'guidance_scale': 9.0,
                'num_inference_steps': 25
            },
            'fantasy': {
                'prompt_suffix': 'fantasy art, magical, dreamlike, surreal',
                'negative_prompt': 'realistic, ordinary, mundane',
                'guidance_scale': 10.0,
                'num_inference_steps': 50
            },
            'vintage': {
                'prompt_suffix': 'vintage photograph, 1950s, retro style',
                'negative_prompt': 'modern, digital, hd',
                'guidance_scale': 7.0,
                'num_inference_steps': 35
            }
        }
    
    def apply_style_transfer(self, image: Image.Image, style_name: str, 
                           strength: float = 0.7) -> Optional[Image.Image]:
        """Apply style transfer using diffusion"""
        try:
            # Create cache key
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"style_{image_hash}_{style_name}_{strength}"
            
            # Check cache
            if cache_key in st.session_state.diffusion_results:
                return st.session_state.diffusion_results[cache_key]
            
            # Get style preset
            if style_name not in st.session_state.style_presets:
                st.error(f"Style '{style_name}' not found")
                return None
            
            preset = st.session_state.style_presets[style_name]
            
            # Generate prompt from image
            llm_enhancer = LLMImageEnhancer(self.model_manager)
            caption = llm_enhancer.generate_caption(image)
            
            # Create final prompt
            base_prompt = f"{caption}, {preset['prompt_suffix']}"
            
            # Load img2img pipeline
            pipe = self.model_manager.load_model("img2img")
            if pipe is None:
                return None
            
            # Prepare image
            init_image = image.convert("RGB").resize((512, 512))
            
            # Generate
            with torch.autocast(self.model_manager.device):
                result = pipe(
                    prompt=base_prompt,
                    negative_prompt=preset['negative_prompt'],
                    image=init_image,
                    strength=strength,
                    guidance_scale=preset['guidance_scale'],
                    num_inference_steps=preset['num_inference_steps']
                ).images[0]
            
            # Resize back to original dimensions
            result = result.resize(image.size)
            
            # Cache result
            st.session_state.diffusion_results[cache_key] = result
            st.session_state.generation_history.append({
                'type': 'style_transfer',
                'style': style_name,
                'original_hash': image_hash,
                'timestamp': datetime.datetime.now()
            })
            
            return result
            
        except Exception as e:
            st.error(f"Error applying style transfer: {str(e)}")
            return None
    
    def text_guided_edit(self, image: Image.Image, prompt: str, 
                        negative_prompt: str = "", strength: float = 0.6) -> Optional[Image.Image]:
        """Edit image based on text prompt"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"edit_{image_hash}_{hash(prompt)}_{strength}"
            
            if cache_key in st.session_state.diffusion_results:
                return st.session_state.diffusion_results[cache_key]
            
            # Load img2img pipeline
            pipe = self.model_manager.load_model("img2img")
            if pipe is None:
                return None
            
            # Prepare image
            init_image = image.convert("RGB").resize((512, 512))
            
            # Generate
            with torch.autocast(self.model_manager.device):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=30
                ).images[0]
            
            result = result.resize(image.size)
            
            # Cache result
            st.session_state.diffusion_results[cache_key] = result
            st.session_state.generation_history.append({
                'type': 'text_edit',
                'prompt': prompt,
                'original_hash': image_hash,
                'timestamp': datetime.datetime.now()
            })
            
            return result
            
        except Exception as e:
            st.error(f"Error in text-guided edit: {str(e)}")
            return None
    
    def inpaint_object(self, image: Image.Image, mask: Image.Image, 
                      prompt: str = "", strength: float = 0.75) -> Optional[Image.Image]:
        """Inpaint object using diffusion"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"inpaint_{image_hash}_{hash(prompt)}_{strength}"
            
            if cache_key in st.session_state.diffusion_results:
                return st.session_state.diffusion_results[cache_key]
            
            # Load inpainting pipeline
            pipe = self.model_manager.load_model("inpaint")
            if pipe is None:
                return None
            
            # Prepare images
            init_image = image.convert("RGB").resize((512, 512))
            mask_image = mask.convert("L").resize((512, 512))
            
            # Generate prompt if not provided
            if not prompt:
                llm_enhancer = LLMImageEnhancer(self.model_manager)
                caption = llm_enhancer.generate_caption(image)
                prompt = f"{caption}, filled area blends naturally"
            
            # Generate
            with torch.autocast(self.model_manager.device):
                result = pipe(
                    prompt=prompt,
                    image=init_image,
                    mask_image=mask_image,
                    strength=strength,
                    guidance_scale=7.5,
                    num_inference_steps=50
                ).images[0]
            
            result = result.resize(image.size)
            
            # Cache result
            st.session_state.diffusion_results[cache_key] = result
            st.session_state.generation_history.append({
                'type': 'inpainting',
                'prompt': prompt,
                'original_hash': image_hash,
                'timestamp': datetime.datetime.now()
            })
            
            return result
            
        except Exception as e:
            st.error(f"Error in inpainting: {str(e)}")
            return None
    
    def generate_variations(self, image: Image.Image, num_variations: int = 4,
                          variation_strength: float = 0.3) -> List[Image.Image]:
        """Generate variations of an image"""
        try:
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"variations_{image_hash}_{num_variations}_{variation_strength}"
            
            if cache_key in st.session_state.diffusion_results:
                return st.session_state.diffusion_results[cache_key]
            
            # Load model
            pipe = self.model_manager.load_model("img2img")
            if pipe is None:
                return []
            
            # Generate caption for prompt
            llm_enhancer = LLMImageEnhancer(self.model_manager)
            caption = llm_enhancer.generate_caption(image)
            
            # Prepare image
            init_image = image.convert("RGB").resize((512, 512))
            
            variations = []
            
            # Generate variations
            with torch.autocast(self.model_manager.device):
                for i in range(num_variations):
                    # Add slight variation to prompt
                    variation_prompt = f"{caption}, variation {i+1}"
                    
                    result = pipe(
                        prompt=variation_prompt,
                        image=init_image,
                        strength=variation_strength,
                        guidance_scale=7.0,
                        num_inference_steps=25,
                        generator=torch.Generator(device=self.model_manager.device).manual_seed(i)
                    ).images[0]
                    
                    result = result.resize(image.size)
                    variations.append(result)
            
            # Cache result
            st.session_state.diffusion_results[cache_key] = variations
            st.session_state.generation_history.append({
                'type': 'variations',
                'count': num_variations,
                'original_hash': image_hash,
                'timestamp': datetime.datetime.now()
            })
            
            return variations
            
        except Exception as e:
            st.error(f"Error generating variations: {str(e)}")
            return []
    
    def upscale_image(self, image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
        """Upscale image using ESRGAN-like approach"""
        try:
            # Simple upscaling for demo (in production, use ESRGAN or similar)
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            return image.resize(new_size, Image.Resampling.LANCZOS)
            
        except Exception as e:
            st.error(f"Error upscaling image: {str(e)}")
            return image
    
    def create_custom_style(self, name: str, prompt_suffix: str, 
                          negative_prompt: str = "", guidance_scale: float = 7.5,
                          num_steps: int = 30):
        """Create custom style preset"""
        st.session_state.style_presets[name] = {
            'prompt_suffix': prompt_suffix,
            'negative_prompt': negative_prompt,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_steps
        }

# ============================================================================
# ENHANCED IMAGE PROCESSOR WITH AI FEATURES
# ============================================================================
class AIImageProcessor(ImageProcessor):
    """Extended image processor with AI capabilities"""
    
    def __init__(self):
        super().__init__()
        self.model_manager = AIModelManager()
        self.llm_enhancer = LLMImageEnhancer(self.model_manager)
        self.diffusion_styler = DiffusionImageStyler(self.model_manager)
    
    @staticmethod
    def apply_basic_filters(image: Image.Image, filter_type: str, intensity: float = 1.0) -> Image.Image:
        """Apply basic image filters"""
        try:
            if filter_type == 'blur':
                return image.filter(ImageFilter.GaussianBlur(radius=intensity))
            elif filter_type == 'sharpen':
                return image.filter(ImageFilter.SHARPEN)
            elif filter_type == 'edge_enhance':
                return image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == 'contrast':
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(intensity)
            elif filter_type == 'brightness':
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(intensity)
            elif filter_type == 'saturation':
                enhancer = ImageEnhance.Color(image)
                return enhancer.enhance(intensity)
            elif filter_type == 'grayscale':
                return image.convert('L').convert('RGB')
            elif filter_type == 'sepia':
                # Apply sepia tone
                sepia_filter = np.array([
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131]
                ])
                img_array = np.array(image)
                sepia_img = np.dot(img_array[..., :3], sepia_filter.T)
                sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
                return Image.fromarray(sepia_img)
            else:
                return image
                
        except Exception as e:
            st.error(f"Error applying filter {filter_type}: {str(e)}")
            return image
    
    def analyze_image_with_llm(self, image: Image.Image) -> Dict:
        """Comprehensive LLM analysis of image"""
        analysis = {}
        
        # Generate caption
        analysis['caption'] = self.llm_enhancer.generate_caption(image)
        
        # Generate description
        analysis['description'] = self.llm_enhancer.generate_detailed_description(image)
        
        # Generate tags
        analysis['tags'] = self.llm_enhancer.generate_tags(image)
        
        # Analyze composition
        analysis['composition'] = self.llm_enhancer.analyze_image_composition(image)
        
        # Generate caption variants
        analysis['caption_variants'] = self.llm_enhancer.generate_ai_caption_variants(image)
        
        return analysis
    
    def apply_ai_style(self, image: Image.Image, style_type: str, **kwargs) -> Optional[Image.Image]:
        """Apply AI-based style to image"""
        return self.diffusion_styler.apply_style_transfer(image, style_type, **kwargs)
    
    def edit_with_prompt(self, image: Image.Image, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Edit image using text prompt"""
        return self.diffusion_styler.text_guided_edit(image, prompt, **kwargs)
    
    def generate_variations(self, image: Image.Image, num_variations: int = 4) -> List[Image.Image]:
        """Generate AI variations of image"""
        return self.diffusion_styler.generate_variations(image, num_variations)
    
    def auto_enhance_image(self, image: Image.Image) -> Image.Image:
        """Auto-enhance image using AI analysis"""
        try:
            # Analyze image
            analysis = self.analyze_image_with_llm(image)
            
            # Apply enhancements based on analysis
            enhanced = image.copy()
            
            if analysis['composition'].get('brightness') == 'dark':
                enhanced = self.apply_basic_filters(enhanced, 'brightness', 1.3)
            
            if analysis['composition'].get('contrast') == 'low':
                enhanced = self.apply_basic_filters(enhanced, 'contrast', 1.2)
            
            if analysis['composition'].get('sharpness') == 'acceptable':
                enhanced = self.apply_basic_filters(enhanced, 'sharpen')
            
            return enhanced
            
        except Exception as e:
            st.error(f"Error in auto-enhancement: {str(e)}")
            return image

# ============================================================================
# ENHANCED ALBUM MANAGER WITH AI FEATURES
# ============================================================================
class EnhancedAlbumManager(AlbumManager):
    """Enhanced album manager with AI capabilities"""
    
    def __init__(self):
        super().__init__()
        self.ai_processor = AIImageProcessor()
        
        # Initialize AI session state
        self._init_ai_session_state()
    
    def _init_ai_session_state(self):
        """Initialize AI-related session state"""
        if 'ai_analysis_results' not in st.session_state:
            st.session_state.ai_analysis_results = {}
        if 'ai_modifications' not in st.session_state:
            st.session_state.ai_modifications = {}
        if 'ai_suggestions' not in st.session_state:
            st.session_state.ai_suggestions = {}
        if 'batch_processing_queue' not in st.session_state:
            st.session_state.batch_processing_queue = []
    
    def analyze_image_with_ai(self, image_path: Path) -> Dict:
        """Analyze image with AI and cache results"""
        try:
            # Check cache
            image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
            cache_key = f"ai_analysis_{image_hash}"
            
            if cache_key in st.session_state.ai_analysis_results:
                return st.session_state.ai_analysis_results[cache_key]
            
            # Load and analyze image
            with Image.open(image_path) as img:
                analysis = self.ai_processor.analyze_image_with_llm(img)
                
                # Add file info
                analysis['file_info'] = {
                    'path': str(image_path),
                    'size': image_path.stat().st_size,
                    'dimensions': img.size
                }
                
                # Cache results
                st.session_state.ai_analysis_results[cache_key] = analysis
                
                return analysis
                
        except Exception as e:
            st.error(f"Error analyzing image with AI: {str(e)}")
            return {}
    
    def apply_ai_modification(self, image_path: Path, modification_type: str, 
                            **kwargs) -> Optional[Path]:
        """Apply AI modification to image and save result"""
        try:
            # Load image
            with Image.open(image_path) as img:
                modified = None
                
                if modification_type == 'style_transfer':
                    style = kwargs.get('style', 'realistic')
                    strength = kwargs.get('strength', 0.7)
                    modified = self.ai_processor.apply_ai_style(img, style, strength=strength)
                    
                elif modification_type == 'text_edit':
                    prompt = kwargs.get('prompt', '')
                    if prompt:
                        modified = self.ai_processor.edit_with_prompt(img, prompt)
                        
                elif modification_type == 'auto_enhance':
                    modified = self.ai_processor.auto_enhance_image(img)
                    
                elif modification_type == 'filter':
                    filter_type = kwargs.get('filter_type', 'sharpen')
                    intensity = kwargs.get('intensity', 1.0)
                    modified = self.ai_processor.apply_basic_filters(img, filter_type, intensity)
                
                if modified:
                    # Save modified image
                    modified_path = image_path.parent / f"{image_path.stem}_modified_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{image_path.suffix}"
                    modified.save(modified_path)
                    
                    # Update cache
                    image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
                    cache_key = f"modification_{image_hash}_{modification_type}"
                    st.session_state.ai_modifications[cache_key] = str(modified_path)
                    
                    # Add to album
                    self._add_modified_image_to_album(modified_path, image_path)
                    
                    return modified_path
            
            return None
            
        except Exception as e:
            st.error(f"Error applying AI modification: {str(e)}")
            return None
    
    def _add_modified_image_to_album(self, modified_path: Path, original_path: Path):
        """Add modified image to album database"""
        try:
            # Get original entry
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT ae.*, p.folder_name 
                FROM album_entries ae
                JOIN images i ON ae.image_id = i.image_id
                JOIN people p ON ae.person_id = p.person_id
                WHERE i.filepath LIKE ?
                ''', (f'%{original_path.name}',))
                
                original_entry = cursor.fetchone()
                
                if original_entry:
                    # Create metadata for modified image
                    metadata = ImageMetadata.from_image(modified_path)
                    thumbnail_path = self.image_processor.create_thumbnail(modified_path)
                    thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                    
                    # Add to database
                    self.db.add_image(metadata, thumbnail_path_str)
                    
                    # Create album entry
                    album_entry = AlbumEntry(
                        entry_id=str(uuid.uuid4()),
                        image_id=metadata.image_id,
                        person_id=original_entry['person_id'],
                        caption=f"[AI Modified] {original_entry['caption']}",
                        description=f"AI-modified version of original photo. {original_entry.get('description', '')}",
                        location=original_entry.get('location', ''),
                        date_taken=datetime.datetime.now(),
                        tags=json.loads(original_entry['tags']) if original_entry['tags'] else [] + ['ai-modified'],
                        privacy_level=original_entry.get('privacy_level', 'public'),
                        created_by=st.session_state['username'],
                        created_at=datetime.datetime.now(),
                        updated_at=datetime.datetime.now()
                    )
                    
                    self.db.add_album_entry(album_entry)
                    
        except Exception as e:
            st.warning(f"Could not add modified image to album: {str(e)}")
    
    def batch_analyze_images(self, image_paths: List[Path]) -> Dict:
        """Batch analyze multiple images"""
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_path in enumerate(image_paths):
            try:
                status_text.text(f"Analyzing {image_path.name}...")
                analysis = self.analyze_image_with_ai(image_path)
                results[str(image_path)] = analysis
                
                # Update progress
                progress = (i + 1) / len(image_paths)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.warning(f"Error analyzing {image_path.name}: {str(e)}")
                results[str(image_path)] = {'error': str(e)}
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def get_ai_suggestions(self, image_path: Path) -> List[Dict]:
        """Get AI suggestions for image improvement"""
        try:
            analysis = self.analyze_image_with_ai(image_path)
            
            suggestions = []
            
            # Composition suggestions
            comp_suggestions = analysis.get('composition', {}).get('suggestions', [])
            for suggestion in comp_suggestions:
                suggestions.append({
                    'type': 'composition',
                    'suggestion': suggestion,
                    'priority': 'medium'
                })
            
            # Style transfer suggestions
            caption = analysis.get('caption', '').lower()
            if 'portrait' in caption or 'person' in caption:
                suggestions.append({
                    'type': 'style',
                    'suggestion': 'Try portrait enhancement or painting style',
                    'priority': 'low'
                })
            
            if 'landscape' in caption or 'nature' in caption:
                suggestions.append({
                    'type': 'style',
                    'suggestion': 'Consider fantasy or painting style for landscape',
                    'priority': 'medium'
                })
            
            # Enhancement suggestions
            suggestions.append({
                'type': 'enhancement',
                'suggestion': 'Try auto-enhance for quick improvements',
                'priority': 'high'
            })
            
            # Tag suggestions
            tags = analysis.get('tags', [])
            if len(tags) < 5:
                suggestions.append({
                    'type': 'metadata',
                    'suggestion': 'Add more descriptive tags for better searchability',
                    'priority': 'low'
                })
            
            return suggestions
            
        except Exception as e:
            st.error(f"Error getting AI suggestions: {str(e)}")
            return []

# ============================================================================
# ENHANCED UI COMPONENTS WITH AI FEATURES
# ============================================================================
class EnhancedUIComponents(UIComponents):
    """Enhanced UI components with AI features"""
    
    @staticmethod
    def ai_analysis_panel(analysis: Dict):
        """Display AI analysis results"""
        with st.expander("🤖 AI Analysis", expanded=True):
            if analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Caption**")
                    st.info(analysis.get('caption', 'No caption generated'))
                    
                    st.markdown("**🏷️ Suggested Tags**")
                    tags = analysis.get('tags', [])
                    if tags:
                        st.markdown(UIComponents.tag_badges(tags[:10]), unsafe_allow_html=True)
                    else:
                        st.text("No tags generated")
                
                with col2:
                    st.markdown("**📊 Composition Analysis**")
                    comp = analysis.get('composition', {})
                    if comp:
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Brightness", comp.get('brightness', 'N/A'))
                            st.metric("Contrast", comp.get('contrast', 'N/A'))
                        with cols[1]:
                            st.metric("Sharpness", comp.get('sharpness', 'N/A'))
                            st.metric("Color Balance", comp.get('color_balance', 'N/A'))
                    
                    # Suggestions
                    suggestions = comp.get('suggestions', [])
                    if suggestions:
                        st.markdown("**💡 Suggestions**")
                        for suggestion in suggestions:
                            st.text(f"• {suggestion}")
                
                # Caption variants
                variants = analysis.get('caption_variants', [])
                if len(variants) > 1:
                    st.markdown("**🔄 Caption Variants**")
                    for i, variant in enumerate(variants[:3]):
                        st.text_area(f"Variant {i+1}", variant, height=50, key=f"variant_{i}")
    
    @staticmethod
    def ai_modification_interface(image_path: Path, album_manager: EnhancedAlbumManager):
        """Interface for AI image modifications"""
        st.subheader("✨ AI Image Modifications")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Style Transfer", "Text Edit", "Auto Enhance", "Batch Process"])
        
        with tab1:
            st.markdown("**🎨 Apply Artistic Style**")
            
            # Style selection
            style_presets = st.session_state.get('style_presets', {})
            style_names = list(style_presets.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                selected_style = st.selectbox("Select Style", style_names, key="style_select")
                strength = st.slider("Style Strength", 0.1, 1.0, 0.7, 0.1, key="style_strength")
            
            with col2:
                if selected_style and style_presets.get(selected_style):
                    preset = style_presets[selected_style]
                    st.markdown(f"**Preset Details:**")
                    st.text(f"Prompt: {preset.get('prompt_suffix', '')}")
                    st.text(f"Guidance: {preset.get('guidance_scale', 7.5)}")
            
            if st.button("Apply Style Transfer", key="apply_style"):
                with st.spinner(f"Applying {selected_style} style..."):
                    result_path = album_manager.apply_ai_modification(
                        image_path, 
                        'style_transfer',
                        style=selected_style,
                        strength=strength
                    )
                    
                    if result_path:
                        st.success(f"Style applied! Saved as {result_path.name}")
                        st.image(str(result_path), caption=f"{selected_style.capitalize()} Style", use_column_width=True)
                        
                        # Offer download
                        with open(result_path, 'rb') as f:
                            st.download_button(
                                label="Download Modified Image",
                                data=f,
                                file_name=result_path.name,
                                mime="image/jpeg"
                            )
        
        with tab2:
            st.markdown("**📝 Edit with Text Prompt**")
            
            prompt = st.text_area(
                "Describe the changes you want",
                placeholder="e.g., 'make it look like a painting', 'add more vibrant colors', 'convert to black and white with color highlights'",
                height=100,
                key="edit_prompt"
            )
            
            negative_prompt = st.text_input(
                "What to avoid",
                placeholder="e.g., 'blurry, distorted, unrealistic'",
                key="negative_prompt"
            )
            
            strength = st.slider("Edit Strength", 0.1, 0.9, 0.6, 0.1, key="edit_strength")
            
            if st.button("Apply Edit", key="apply_edit") and prompt:
                with st.spinner("Applying text-guided edit..."):
                    result_path = album_manager.apply_ai_modification(
                        image_path,
                        'text_edit',
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        strength=strength
                    )
                    
                    if result_path:
                        st.success("Edit applied successfully!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(str(image_path), caption="Original", use_column_width=True)
                        with col2:
                            st.image(str(result_path), caption="Edited", use_column_width=True)
        
        with tab3:
            st.markdown("**⚡ Auto Enhance**")
            st.markdown("""
            Automatically enhances your image based on AI analysis:
            - Adjusts brightness and contrast
            - Improves sharpness
            - Optimizes colors
            """)
            
            if st.button("Run Auto Enhancement", key="auto_enhance"):
                with st.spinner("Analyzing and enhancing image..."):
                    # First analyze
                    analysis = album_manager.analyze_image_with_ai(image_path)
                    
                    # Then enhance
                    result_path = album_manager.apply_ai_modification(
                        image_path,
                        'auto_enhance'
                    )
                    
                    if result_path:
                        st.success("Enhancement complete!")
                        
                        # Show comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(str(image_path), caption="Original", use_column_width=True)
                        with col2:
                            st.image(str(result_path), caption="Enhanced", use_column_width=True)
                        
                        # Show what was improved
                        if analysis.get('composition'):
                            st.markdown("**📊 Improvements Applied:**")
                            comp = analysis['composition']
                            improvements = []
                            
                            if comp.get('brightness') == 'dark':
                                improvements.append("• Increased brightness")
                            if comp.get('contrast') == 'low':
                                improvements.append("• Enhanced contrast")
                            if comp.get('sharpness') == 'acceptable':
                                improvements.append("• Improved sharpness")
                            
                            if improvements:
                                for imp in improvements:
                                    st.text(imp)
                            else:
                                st.text("Image was already well balanced")
        
        with tab4:
            st.markdown("**🔀 Batch Processing**")
            
            # Get all images in the same directory
            image_dir = image_path.parent
            all_images = [
                f for f in image_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
            ]
            
            if len(all_images) > 1:
                st.markdown(f"Found {len(all_images)} images in this folder")
                
                # Select images for batch processing
                selected_images = st.multiselect(
                    "Select images for batch processing",
                    [img.name for img in all_images],
                    default=[image_path.name],
                    key="batch_images"
                )
                
                # Select operation
                operation = st.selectbox(
                    "Operation",
                    ["Analyze All", "Auto Enhance All", "Apply Style to All"],
                    key="batch_operation"
                )
                
                if operation == "Apply Style to All":
                    batch_style = st.selectbox("Style", list(style_presets.keys()), key="batch_style")
                
                if st.button("Run Batch Processing", key="run_batch"):
                    # Process selected images
                    selected_paths = [image_dir / name for name in selected_images]
                    
                    with st.spinner(f"Processing {len(selected_paths)} images..."):
                        if operation == "Analyze All":
                            results = album_manager.batch_analyze_images(selected_paths)
                            
                            # Display summary
                            st.success(f"Analyzed {len(results)} images")
                            
                            # Show some insights
                            all_tags = []
                            for result in results.values():
                                if isinstance(result, dict):
                                    tags = result.get('tags', [])
                                    all_tags.extend(tags)
                            
                            if all_tags:
                                from collections import Counter
                                tag_counts = Counter(all_tags)
                                st.markdown("**Most Common Tags:**")
                                for tag, count in tag_counts.most_common(10):
                                    st.text(f"{tag}: {count} images")
                        
                        elif operation == "Auto Enhance All":
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            enhanced_count = 0
                            for i, img_path in enumerate(selected_paths):
                                status_text.text(f"Processing {img_path.name}...")
                                result = album_manager.apply_ai_modification(img_path, 'auto_enhance')
                                if result:
                                    enhanced_count += 1
                                
                                progress = (i + 1) / len(selected_paths)
                                progress_bar.progress(progress)
                            
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"Enhanced {enhanced_count} of {len(selected_paths)} images")
                        
                        elif operation == "Apply Style to All":
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            styled_count = 0
                            for i, img_path in enumerate(selected_paths):
                                status_text.text(f"Applying {batch_style} style to {img_path.name}...")
                                result = album_manager.apply_ai_modification(
                                    img_path,
                                    'style_transfer',
                                    style=batch_style
                                )
                                if result:
                                    styled_count += 1
                                
                                progress = (i + 1) / len(selected_paths)
                                progress_bar.progress(progress)
                            
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"Applied {batch_style} style to {styled_count} images")
            else:
                st.info("No other images found in this folder for batch processing")

# ============================================================================
# ENHANCED PHOTO ALBUM APP WITH AI FEATURES
# ============================================================================
class EnhancedPhotoAlbumApp(PhotoAlbumApp):
    """Enhanced photo album app with AI features"""
    
    def __init__(self):
        super().__init__()
        self.manager = EnhancedAlbumManager()
        self.ui_components = EnhancedUIComponents()
        
        # Add AI pages to navigation
        self._extend_navigation()
    
    def _extend_navigation(self):
        """Extend navigation with AI features"""
        # This will be called in render_sidebar
        pass
    
    def render_sidebar(self):
        """Render enhanced sidebar with AI options"""
        super().render_sidebar()
        
        # Add AI section to sidebar
        with st.sidebar:
            st.divider()
            st.subheader("🤖 AI Features")
            
            ai_options = {
                "🔍 AI Analysis": "ai_analysis",
                "🎨 AI Studio": "ai_studio",
                "⚡ Quick Enhance": "quick_enhance",
                "📊 AI Insights": "ai_insights"
            }
            
            for label, key in ai_options.items():
                if st.button(label, use_container_width=True, key=f"ai_nav_{key}"):
                    st.session_state['current_page'] = key
                    st.rerun()
    
    def render_ai_analysis_page(self):
        """Render AI analysis page"""
        st.title("🔍 AI Image Analysis")
        
        # Image selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get all people for selection
            people = self.manager.db.get_all_people()
            person_options = {p['display_name']: p['person_id'] for p in people}
            
            selected_person = st.selectbox(
                "Select Person",
                list(person_options.keys()),
                key="ai_analysis_person"
            )
        
        with col2:
            if selected_person:
                person_id = person_options[selected_person]
                
                # Get person's images
                entries = self.manager.get_entries_by_person(person_id, page=1)
                image_options = {}
                for entry in entries.get('entries', []):
                    image_options[entry.get('caption', 'Untitled')] = entry.get('image_id')
                
                selected_image_caption = st.selectbox(
                    "Select Image",
                    list(image_options.keys()),
                    key="ai_analysis_image"
                )
        
        # Display and analyze selected image
        if selected_person and selected_image_caption:
            image_id = image_options[selected_image_caption]
            
            # Get entry details
            entry = self.manager.get_entry_with_details(image_id)
            
            if entry:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display image
                    if entry.get('image_data_url'):
                        st.image(entry['image_data_url'], caption=entry.get('caption'), use_column_width=True)
                    
                    # Image info
                    st.markdown(f"**📏 Dimensions:** {entry.get('dimensions', 'N/A')}")
                    st.markdown(f"**📂 Format:** {entry.get('format', 'N/A')}")
                    st.markdown(f"**🗓️ Date:** {entry.get('created_at', 'N/A')}")
                
                with col2:
                    # Run AI analysis
                    if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                        # Get image path
                        if entry.get('filepath'):
                            image_path = Config.DATA_DIR / entry['filepath']
                            
                            if image_path.exists():
                                with st.spinner("Analyzing image with AI..."):
                                    # Perform AI analysis
                                    analysis = self.manager.analyze_image_with_ai(image_path)
                                    
                                    # Display results
                                    self.ui_components.ai_analysis_panel(analysis)
                                    
                                    # Get suggestions
                                    suggestions = self.manager.get_ai_suggestions(image_path)
                                    
                                    if suggestions:
                                        st.markdown("**💡 AI Suggestions**")
                                        for suggestion in suggestions:
                                            with st.container(border=True):
                                                st.markdown(f"**{suggestion['type'].title()}**")
                                                st.text(suggestion['suggestion'])
                                                st.caption(f"Priority: {suggestion['priority']}")
                            else:
                                st.error("Image file not found")
                        else:
                            st.error("No image path available")
                    else:
                        st.info("Click 'Analyze with AI' to start analysis")
            
            # Show AI modification interface
            st.divider()
            st.subheader("✨ Apply AI Modifications")
            
            if entry and entry.get('filepath'):
                image_path = Config.DATA_DIR / entry['filepath']
                if image_path.exists():
                    self.ui_components.ai_modification_interface(image_path, self.manager)
    
    def render_ai_studio_page(self):
        """Render AI studio for creative modifications"""
        st.title("🎨 AI Studio")
        
        tab1, tab2, tab3 = st.tabs(["Creative Styles", "Text to Image", "Advanced Tools"])
        
        with tab1:
            st.subheader("Creative Style Transfer")
            
            # Image upload or selection
            image_source = st.radio(
                "Image Source",
                ["Select from Album", "Upload New Image"],
                horizontal=True,
                key="studio_image_source"
            )
            
            image = None
            image_path = None
            
            if image_source == "Select from Album":
                # Get recent images
                recent_entries = self.manager.get_recent_entries(limit=10)
                
                if recent_entries:
                    image_options = {e['caption']: e['entry_id'] for e in recent_entries}
                    selected_caption = st.selectbox("Select Image", list(image_options.keys()))
                    
                    if selected_caption:
                        entry_id = image_options[selected_caption]
                        entry = self.manager.get_entry_with_details(entry_id)
                        
                        if entry and entry.get('filepath'):
                            image_path = Config.DATA_DIR / entry['filepath']
                            if image_path.exists():
                                image = Image.open(image_path)
            else:
                uploaded_file = st.file_uploader(
                    "Upload an image",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    key="studio_upload"
                )
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    image_path = Path(uploaded_file.name)
            
            if image:
                # Display original
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                
                with col2:
                    # Style selection
                    style_presets = st.session_state.get('style_presets', {})
                    
                    # Custom style creator
                    with st.expander("🎨 Create Custom Style"):
                        custom_name = st.text_input("Style Name", key="custom_style_name")
                        custom_prompt = st.text_input("Prompt Suffix", key="custom_prompt")
                        custom_negative = st.text_input("Negative Prompt", key="custom_negative")
                        custom_guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5, key="custom_guidance")
                        custom_steps = st.slider("Steps", 10, 100, 30, key="custom_steps")
                        
                        if st.button("Save Custom Style", key="save_custom_style"):
                            if custom_name and custom_prompt:
                                self.manager.ai_processor.diffusion_styler.create_custom_style(
                                    custom_name,
                                    custom_prompt,
                                    custom_negative,
                                    custom_guidance,
                                    custom_steps
                                )
                                st.success(f"Style '{custom_name}' saved!")
                                st.rerun()
                    
                    # Select style
                    all_styles = list(style_presets.keys())
                    selected_style = st.selectbox("Select Style", all_styles, key="studio_style")
                    
                    # Style parameters
                    strength = st.slider("Strength", 0.1, 1.0, 0.7, 0.1, key="studio_strength")
                    num_variations = st.slider("Number of Variations", 1, 4, 1, key="studio_variations")
                    
                    if st.button("Generate Styled Images", type="primary", key="generate_styles"):
                        with st.spinner("Applying style transfer..."):
                            # Generate variations
                            variations = []
                            for i in range(num_variations):
                                with st.spinner(f"Generating variation {i+1}..."):
                                    result = self.manager.ai_processor.diffusion_styler.apply_style_transfer(
                                        image,
                                        selected_style,
                                        strength=strength
                                    )
                                    if result:
                                        variations.append(result)
                            
                            # Display results
                            if variations:
                                st.success(f"Generated {len(variations)} variations!")
                                
                                cols = st.columns(min(4, len(variations)))
                                for i, var in enumerate(variations):
                                    with cols[i % len(cols)]:
                                        st.image(var, caption=f"Variation {i+1}", use_column_width=True)
                                        
                                        # Offer download
                                        buf = BytesIO()
                                        var.save(buf, format='JPEG', quality=95)
                                        st.download_button(
                                            label=f"Download Var {i+1}",
                                            data=buf.getvalue(),
                                            file_name=f"styled_{selected_style}_{i+1}.jpg",
                                            mime="image/jpeg",
                                            key=f"download_var_{i}"
                                        )
        
        with tab2:
            st.subheader("Text to Image Generation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                prompt = st.text_area(
                    "Describe the image you want to create",
                    height=150,
                    placeholder="A beautiful sunset over mountains, digital art, vibrant colors, 4k",
                    key="text2img_prompt"
                )
                
                negative_prompt = st.text_input(
                    "What to avoid",
                    placeholder="blurry, distorted, ugly",
                    key="text2img_negative"
                )
                
                # Generation parameters
                num_images = st.slider("Number of images", 1, 4, 1, key="text2img_count")
                guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, 0.5, key="text2img_guidance")
                num_steps = st.slider("Steps", 10, 100, 30, key="text2img_steps")
                seed = st.number_input("Seed (optional)", value=-1, key="text2img_seed")
            
            with col2:
                if st.button("Generate Images", type="primary", key="generate_text2img"):
                    if prompt:
                        with st.spinner("Generating images..."):
                            try:
                                # Load diffusion model
                                pipe = self.manager.ai_processor.model_manager.load_model("diffusion")
                                
                                if pipe:
                                    # Generate images
                                    generator = None
                                    if seed != -1:
                                        generator = torch.Generator(device=self.manager.ai_processor.model_manager.device).manual_seed(seed)
                                    
                                    images = pipe(
                                        prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        guidance_scale=guidance_scale,
                                        num_inference_steps=num_steps,
                                        num_images_per_prompt=num_images,
                                        generator=generator
                                    ).images
                                    
                                    # Display results
                                    st.success(f"Generated {len(images)} images!")
                                    
                                    cols = st.columns(min(4, len(images)))
                                    for i, img in enumerate(images):
                                        with cols[i % len(cols)]:
                                            st.image(img, caption=f"Generated {i+1}", use_column_width=True)
                                            
                                            # Offer download
                                            buf = BytesIO()
                                            img.save(buf, format='JPEG', quality=95)
                                            st.download_button(
                                                label=f"Download {i+1}",
                                                data=buf.getvalue(),
                                                file_name=f"generated_{i+1}.jpg",
                                                mime="image/jpeg",
                                                key=f"download_gen_{i}"
                                            )
                                            
                                            # Option to add to album
                                            if st.button(f"Add to Album", key=f"add_gen_{i}"):
                                                # Save image
                                                save_path = Config.DATA_DIR / "generated" / f"gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                                                save_path.parent.mkdir(exist_ok=True)
                                                img.save(save_path)
                                                
                                                # Add to database
                                                metadata = ImageMetadata.from_image(save_path)
                                                thumbnail_path = self.manager.image_processor.create_thumbnail(save_path)
                                                thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                                                
                                                self.manager.db.add_image(metadata, thumbnail_path_str)
                                                
                                                # Create album entry
                                                album_entry = AlbumEntry(
                                                    entry_id=str(uuid.uuid4()),
                                                    image_id=metadata.image_id,
                                                    person_id='generated',  # Special person ID for generated images
                                                    caption=f"AI Generated: {prompt[:50]}...",
                                                    description=f"Generated with prompt: {prompt}",
                                                    location="AI Studio",
                                                    date_taken=datetime.datetime.now(),
                                                    tags=['ai-generated', 'text-to-image'],
                                                    privacy_level='public',
                                                    created_by=st.session_state['username'],
                                                    created_at=datetime.datetime.now(),
                                                    updated_at=datetime.datetime.now()
                                                )
                                                
                                                self.manager.db.add_album_entry(album_entry)
                                                st.success("Added to album!")
                                else:
                                    st.error("Could not load diffusion model")
                                    
                            except Exception as e:
                                st.error(f"Error generating images: {str(e)}")
                    else:
                        st.warning("Please enter a prompt")
        
        with tab3:
            st.subheader("Advanced AI Tools")
            
            tool_option = st.selectbox(
                "Select Tool",
                ["Image Inpainting", "Upscaling", "Background Removal", "Colorization"],
                key="advanced_tool"
            )
            
            if tool_option == "Image Inpainting":
                st.markdown("**🖌️ Image Inpainting**")
                st.markdown("Upload an image and draw a mask to remove/replace objects")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    inpaint_image = st.file_uploader(
                        "Upload image for inpainting",
                        type=['jpg', 'jpeg', 'png'],
                        key="inpaint_upload"
                    )
                    
                    if inpaint_image:
                        img = Image.open(inpaint_image)
                        st.image(img, caption="Original", use_column_width=True)
                        
                        # Simple mask drawing (in production, use a proper drawing tool)
                        st.markdown("**Draw Mask**")
                        st.info("For this demo, we'll use a simple box selection")
                        
                        mask_type = st.radio(
                            "Mask Type",
                            ["Remove Object", "Replace Background"],
                            horizontal=True,
                            key="mask_type"
                        )
                        
                        if mask_type == "Remove Object":
                            prompt = st.text_input(
                                "What should fill the masked area?",
                                placeholder="Leave empty for automatic fill",
                                key="inpaint_prompt"
                            )
                        
                        if st.button("Apply Inpainting", key="apply_inpaint"):
                            st.info("Inpainting requires a proper mask. In production, integrate with a drawing tool.")
                
                with col2:
                    st.markdown("**How it works:**")
                    st.markdown("""
                    1. Upload your image
                    2. Draw a mask over the area you want to modify
                    3. Describe what you want in that area (optional)
                    4. AI will fill the masked area naturally
                    
                    **Use cases:**
                    - Remove unwanted objects
                    - Replace backgrounds
                    - Fix damaged photos
                    - Add new elements
                    """)
            
            elif tool_option == "Upscaling":
                st.markdown("**🔍 Image Upscaling**")
                
                upscale_image = st.file_uploader(
                    "Upload image to upscale",
                    type=['jpg', 'jpeg', 'png'],
                    key="upscale_upload"
                )
                
                if upscale_image:
                    img = Image.open(upscale_image)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(img, caption=f"Original ({img.width}x{img.height})", use_column_width=True)
                    
                    with col2:
                        scale_factor = st.slider("Upscale Factor", 1.5, 4.0, 2.0, 0.5, key="upscale_factor")
                        
                        if st.button("Upscale Image", key="upscale_button"):
                            with st.spinner("Upscaling image..."):
                                # Use diffusion-based upscaling or simple interpolation
                                upscaled = self.manager.ai_processor.diffusion_styler.upscale_image(img, scale_factor)
                                
                                st.image(upscaled, caption=f"Upscaled ({upscaled.width}x{upscaled.height})", use_column_width=True)
                                
                                # Offer download
                                buf = BytesIO()
                                upscaled.save(buf, format='JPEG', quality=95)
                                st.download_button(
                                    label="Download Upscaled",
                                    data=buf.getvalue(),
                                    file_name=f"upscaled_{scale_factor}x.jpg",
                                    mime="image/jpeg"
                                )
    
    def render_quick_enhance_page(self):
        """Render quick enhancement page"""
        st.title("⚡ Quick Enhance")
        
        # Recent images grid
        st.markdown("### Recent Photos")
        
        recent_entries = self.manager.get_recent_entries(limit=12)
        
        if recent_entries:
            cols = st.columns(4)
            selected_images = []
            
            for idx, entry in enumerate(recent_entries):
                with cols[idx % 4]:
                    with st.container(border=True):
                        # Thumbnail
                        if entry.get('thumbnail_data_url'):
                            st.image(entry['thumbnail_data_url'], use_column_width=True)
                        
                        # Caption
                        st.markdown(f"**{entry.get('caption', 'Untitled')[:20]}...**")
                        
                        # Checkbox for selection
                        if st.checkbox("Select", key=f"select_{entry['entry_id']}"):
                            selected_images.append(entry['entry_id'])
            
            # Batch actions
            if selected_images:
                st.divider()
                st.markdown(f"**{len(selected_images)} images selected**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🤖 Analyze All", use_container_width=True):
                        st.session_state['batch_analysis_ids'] = selected_images
                        st.session_state['current_page'] = 'batch_analysis'
                        st.rerun()
                
                with col2:
                    if st.button("⚡ Auto Enhance All", use_container_width=True):
                        with st.spinner("Enhancing images..."):
                            enhanced_count = 0
                            for entry_id in selected_images:
                                entry = self.manager.get_entry_with_details(entry_id)
                                if entry and entry.get('filepath'):
                                    image_path = Config.DATA_DIR / entry['filepath']
                                    if image_path.exists():
                                        result = self.manager.apply_ai_modification(image_path, 'auto_enhance')
                                        if result:
                                            enhanced_count += 1
                            
                            st.success(f"Enhanced {enhanced_count} images!")
                
                with col3:
                    style = st.selectbox(
                        "Apply Style",
                        list(st.session_state.get('style_presets', {}).keys()),
                        key="batch_style_select"
                    )
                    
                    if st.button(f"🎨 Apply {style}", use_container_width=True):
                        with st.spinner(f"Applying {style} style..."):
                            styled_count = 0
                            for entry_id in selected_images:
                                entry = self.manager.get_entry_with_details(entry_id)
                                if entry and entry.get('filepath'):
                                    image_path = Config.DATA_DIR / entry['filepath']
                                    if image_path.exists():
                                        result = self.manager.apply_ai_modification(
                                            image_path,
                                            'style_transfer',
                                            style=style
                                        )
                                        if result:
                                            styled_count += 1
                            
                            st.success(f"Applied {style} style to {styled_count} images!")
        else:
            st.info("No recent photos found. Scan your directory first.")
    
    def render_ai_insights_page(self):
        """Render AI insights page"""
        st.title("📊 AI Insights")
        
        # Get all images for analysis
        all_people = self.manager.get_all_people_with_stats()
        
        if not all_people:
            st.info("No data available. Scan your directory first.")
            return
        
        # Overall AI insights
        st.subheader("🔍 Overall Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Most common tags across all images
            st.markdown("**🏷️ Most Common Tags**")
            
            # Sample analysis (in production, analyze all images)
            sample_tags = []
            for person in all_people[:5]:  # Limit for demo
                person_id = person['person_id']
                entries = self.manager.get_entries_by_person(person_id, page=1, limit=5)
                for entry in entries.get('entries', []):
                    if entry.get('tags'):
                        tags = entry['tags'].split(',') if isinstance(entry['tags'], str) else entry['tags']
                        sample_tags.extend(tags)
            
            if sample_tags:
                from collections import Counter
                tag_counts = Counter(sample_tags)
                for tag, count in tag_counts.most_common(10):
                    st.text(f"• {tag}: {count}")
            else:
                st.text("No tags found")
        
        with col2:
            st.markdown("**🎨 Style Recommendations**")
            
            recommendations = [
                "Portrait photos benefit from painting styles",
                "Landscape images work well with fantasy styles",
                "Group photos can be enhanced with vintage filters",
                "Low-light images need brightness adjustment"
            ]
            
            for rec in recommendations:
                st.text(f"• {rec}")
        
        with col3:
            st.markdown("**📈 AI Usage Stats**")
            
            ai_stats = {
                "Style Transfers": len([h for h in st.session_state.get('generation_history', []) 
                                      if h.get('type') == 'style_transfer']),
                "Text Edits": len([h for h in st.session_state.get('generation_history', []) 
                                 if h.get('type') == 'text_edit']),
                "Auto Enhancements": len(st.session_state.get('ai_modifications', {}))
            }
            
            for stat, value in ai_stats.items():
                st.metric(stat, value)
        
        # AI-generated suggestions
        st.divider()
        st.subheader("💡 AI Suggestions for Your Album")
        
        suggestions_tab1, suggestions_tab2, suggestions_tab3 = st.tabs(["Content", "Organization", "Enhancement"])
        
        with suggestions_tab1:
            st.markdown("**Content Suggestions**")
            
            content_suggestions = [
                "Consider adding more landscape photos for variety",
                "Group photos by events or themes",
                "Add descriptive captions to older photos",
                "Consider scanning physical photos to digital"
            ]
            
            for suggestion in content_suggestions:
                with st.container(border=True):
                    st.markdown(suggestion)
                    if st.button("Apply Suggestion", key=f"content_{hash(suggestion)}"):
                        st.info("This would trigger an automated process in production")
        
        with suggestions_tab2:
            st.markdown("**Organization Suggestions**")
            
            org_suggestions = [
                "Create albums by year for better navigation",
                "Add location tags to travel photos",
                "Use facial recognition to group people",
                "Create themed collections (holidays, vacations)"
            ]
            
            for suggestion in org_suggestions:
                with st.container(border=True):
                    st.markdown(suggestion)
        
        with suggestions_tab3:
            st.markdown("**Enhancement Suggestions**")
            
            enhance_suggestions = [
                "Batch process all photos with auto-enhance",
                "Apply consistent filters to event photos",
                "Upscale older low-resolution photos",
                "Colorize black and white photos"
            ]
            
            for suggestion in enhance_suggestions:
                with st.container(border=True):
                    st.markdown(suggestion)
                    if st.button("Run Enhancement", key=f"enhance_{hash(suggestion)}"):
                        st.info("Batch processing would start in production")
        
        # AI model management
        st.divider()
        st.subheader("🤖 AI Model Management")
        
        with st.expander("Model Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Loaded Models**")
                loaded_models = list(self.manager.ai_processor.model_manager.model_cache.keys())
                
                if loaded_models:
                    for model in loaded_models:
                        st.text(f"• {model}")
                else:
                    st.text("No models loaded")
                
                if st.button("Load All Models", key="load_all_models"):
                    with st.spinner("Loading AI models..."):
                        # Load caption model
                        self.manager.ai_processor.model_manager.load_model("caption")
                        # Load diffusion model
                        self.manager.ai_processor.model_manager.load_model("diffusion")
                        st.success("Models loaded!")
                        st.rerun()
            
            with col2:
                st.markdown("**Memory Management**")
                
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    
                    st.metric("GPU Memory Used", f"{memory_allocated:.2f} GB")
                    st.metric("GPU Memory Reserved", f"{memory_reserved:.2f} GB")
                
                if st.button("Clear Model Cache", key="clear_model_cache"):
                    self.manager.ai_processor.model_manager.clear_all_models()
                    st.success("Model cache cleared!")
                    st.rerun()
                
                if st.button("Unload Heavy Models", key="unload_heavy"):
                    self.manager.ai_processor.model_manager.unload_model("diffusion")
                    self.manager.ai_processor.model_manager.unload_model("inpaint")
                    st.success("Heavy models unloaded!")
                    st.rerun()
    
    def render_batch_analysis_page(self):
        """Render batch analysis results page"""
        st.title("🔍 Batch Analysis Results")
        
        if 'batch_analysis_ids' not in st.session_state or not st.session_state.batch_analysis_ids:
            st.warning("No images selected for batch analysis")
            if st.button("Back to Quick Enhance"):
                st.session_state['current_page'] = 'quick_enhance'
                st.rerun()
            return
        
        entry_ids = st.session_state.batch_analysis_ids
        
        # Progress tracking
        if 'batch_analysis_progress' not in st.session_state:
            st.session_state.batch_analysis_progress = 0
            st.session_state.batch_analysis_results = {}
        
        progress_bar = st.progress(st.session_state.batch_analysis_progress)
        status_text = st.empty()
        
        # Analyze each image
        for i, entry_id in enumerate(entry_ids):
            if entry_id not in st.session_state.batch_analysis_results:
                status_text.text(f"Analyzing image {i+1} of {len(entry_ids)}...")
                
                entry = self.manager.get_entry_with_details(entry_id)
                if entry and entry.get('filepath'):
                    image_path = Config.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        analysis = self.manager.analyze_image_with_ai(image_path)
                        st.session_state.batch_analysis_results[entry_id] = {
                            'analysis': analysis,
                            'caption': entry.get('caption'),
                            'image_path': image_path
                        }
                
                # Update progress
                st.session_state.batch_analysis_progress = (i + 1) / len(entry_ids)
                progress_bar.progress(st.session_state.batch_analysis_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"Analysis complete! Processed {len(st.session_state.batch_analysis_results)} images")
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        all_tags = []
        all_captions = []
        
        for result in st.session_state.batch_analysis_results.values():
            analysis = result.get('analysis', {})
            tags = analysis.get('tags', [])
            all_tags.extend(tags)
            
            caption = result.get('caption', '')
            if caption:
                all_captions.append(caption)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tag Cloud**")
            if all_tags:
                from collections import Counter
                tag_counts = Counter(all_tags)
                
                # Create tag cloud visualization
                cloud_html = '<div style="line-height: 2.5;">'
                for tag, count in tag_counts.most_common(20):
                    size = min(24, 12 + count * 2)
                    cloud_html += f'<span style="font-size: {size}px; margin: 5px; display: inline-block;">{tag}</span>'
                cloud_html += '</div>'
                
                st.markdown(cloud_html, unsafe_allow_html=True)
            else:
                st.text("No tags generated")
        
        with col2:
            st.markdown("**Common Themes**")
            
            # Simple theme detection
            themes = {}
            for caption in all_captions:
                words = caption.lower().split()
                for word in words:
                    if len(word) > 4 and word not in ['photo', 'image', 'picture']:
                        themes[word] = themes.get(word, 0) + 1
            
            if themes:
                for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:10]:
                    st.text(f"• {theme}: {count} images")
            else:
                st.text("No common themes detected")
        
        # Detailed results
        st.divider()
        st.subheader("📋 Detailed Results")
        
        for entry_id, result in st.session_state.batch_analysis_results.items():
            with st.expander(f"📸 {result.get('caption', 'Unknown')}"):
                analysis = result.get('analysis', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display thumbnail
                    image_path = result.get('image_path')
                    if image_path and image_path.exists():
                        st.image(str(image_path), width=200)
                    
                    # Basic info
                    st.markdown(f"**Caption:** {analysis.get('caption', 'N/A')}")
                    
                    # Tags
                    tags = analysis.get('tags', [])
                    if tags:
                        st.markdown("**Tags:**")
                        st.markdown(UIComponents.tag_badges(tags[:5]), unsafe_allow_html=True)
                
                with col2:
                    # Composition analysis
                    comp = analysis.get('composition', {})
                    if comp:
                        st.markdown("**Composition Analysis:**")
                        
                        metrics = ['brightness', 'contrast', 'sharpness', 'color_balance']
                        for metric in metrics:
                            value = comp.get(metric, 'N/A')
                            st.text(f"• {metric.title()}: {value}")
                        
                        suggestions = comp.get('suggestions', [])
                        if suggestions:
                            st.markdown("**Suggestions:**")
                            for suggestion in suggestions[:3]:
                                st.text(f"• {suggestion}")
        
        # Clear batch analysis
        if st.button("Clear Batch Results", key="clear_batch"):
            del st.session_state.batch_analysis_ids
            del st.session_state.batch_analysis_progress
            del st.session_state.batch_analysis_results
            st.rerun()
    
    def render_main(self):
        """Enhanced main application renderer with AI pages"""
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
        # AI pages
        elif current_page == 'ai_analysis':
            self.render_ai_analysis_page()
        elif current_page == 'ai_studio':
            self.render_ai_studio_page()
        elif current_page == 'quick_enhance':
            self.render_quick_enhance_page()
        elif current_page == 'ai_insights':
            self.render_ai_insights_page()
        elif current_page == 'batch_analysis':
            self.render_batch_analysis_page()
        else:
            self.render_dashboard()
        
        # Enhanced footer with AI status
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            # AI model status
            loaded_models = len(self.manager.ai_processor.model_manager.model_cache)
            st.caption(f"🤖 {loaded_models} AI models loaded")
        
        with col2:
            st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")
            st.caption("Enhanced with AI-powered image analysis and modification")
        
        with col3:
            # Clear AI cache button
            if st.button("🧹 Clear AI Cache", key="footer_clear_ai"):
                self.manager.ai_processor.model_manager.clear_all_models()
                st.session_state.ai_analysis_results = {}
                st.session_state.ai_modifications = {}
                st.success("AI cache cleared!")
                st.rerun()

# ============================================================================
# ENHANCED MAIN FUNCTION
# ============================================================================
def main_enhanced():
    """Enhanced main application entry point"""
    try:
        # Initialize enhanced app
        app = EnhancedPhotoAlbumApp()
        
        # Check initialization
        if not app.initialized:
            st.error("Application failed to initialize. Check directory permissions.")
            return
        
        # Add AI models initialization warning
        with st.sidebar:
            if st.button("🔄 Initialize AI Models", key="init_ai_models"):
                with st.spinner("Loading AI models (this may take a minute)..."):
                    # Try to load a light model first
                    try:
                        app.manager.ai_processor.model_manager.load_model("caption")
                        st.success("AI models initialized!")
                    except Exception as e:
                        st.warning(f"AI models may not load properly: {str(e)}")
        
        # Render main app
        app.render_main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        
        # Show detailed error for debugging
        with st.expander("Error Details"):
            st.exception(e)
        
        # Recovery options with AI models
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Try Again"):
                st.rerun()
        with col2:
            if st.button("Reset Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col3:
            if st.button("Disable AI Features"):
                # Fall back to basic app
                basic_app = PhotoAlbumApp()
                basic_app.render_main()

# ============================================================================
# REQUIREMENTS FILE GENERATOR
# ============================================================================
def generate_requirements():
    """Generate requirements.txt for the enhanced application"""
    requirements = """
# Core dependencies
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
sqlite3

# AI/ML dependencies
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.19.0
accelerate>=0.21.0

# Image processing
opencv-python>=4.8.0
scipy>=1.11.0

# Optional: For better performance
xformers>=0.0.20  # For faster attention in diffusion models
triton>=2.0.0     # For optimized operations

# Optional: Cloud AI APIs
openai>=0.28.0
replicate>=0.11.0

# Development
black>=23.0.0
flake8>=6.0.0
"""

    return requirements

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Check if AI dependencies are available
    try:
        import torch
        import transformers
        has_ai_deps = True
    except ImportError:
        has_ai_deps = False
        st.warning("""
        ⚠️ AI dependencies not found. Running in basic mode.
        
        To enable AI features, install required packages:
        ```
        pip install torch torchvision transformers diffusers
        ```
        
        Or use the generated requirements.txt
        """)
    
    # Let user choose mode
    if has_ai_deps:
        mode = st.sidebar.selectbox(
            "Application Mode",
            ["🎨 Enhanced (AI Features)", "📸 Basic (No AI)"],
            key="app_mode"
        )
        
        if "Enhanced" in mode:
            main_enhanced()
        else:
            # Run basic version
            main()
    else:
        # Run basic version
        main()
    
    # Show requirements in sidebar for easy access
    with st.sidebar.expander("📋 Requirements"):
        st.code(generate_requirements(), language="text")
        if st.button("Copy Requirements"):
            st.write("Requirements copied to clipboard (in a real app)")
