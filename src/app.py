import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, GPT2Tokenizer
from model import ImageCaptioningModel
import gdown
import os

# Page configuration
st.set_page_config(
    page_title="CaptiFy AI - Image Captioning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        font-size: 1.5rem;
        margin-top: 20px;
        color: #FFFFFF;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model from Google Drive"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Google Drive file ID
    GDRIVE_FILE_ID = "15qBwjanLs8022iAZ1_fj989IC-TZ6395"
    
    with st.spinner('🤖 Loading CaptiFy AI model... First time may take 1-2 minutes...'):
        # Download checkpoint if not cached
        cache_dir = "model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        checkpoint_path = os.path.join(cache_dir, "best_model.pth")
        
        if not os.path.exists(checkpoint_path):
            st.info("⬇️ Downloading model for the first time (~18MB)...")
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, checkpoint_path, quiet=False)
            st.success("✓ Model downloaded and cached!")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {
            'pad_token': '<|pad|>',
            'bos_token': '<|startoftext|>',
            'eos_token': '<|endoftext|>'
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # Load ViT processor
        vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # Load model
        model = ImageCaptioningModel(freeze_vit=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    
    return model, tokenizer, vit_processor, device

def generate_caption(image, model, tokenizer, vit_processor, device, num_beams, max_length):
    """Generate caption for uploaded image"""
    # Ensure RGB
    image = image.convert('RGB')
    
    # Preprocess
    image_processed = vit_processor(images=image, return_tensors="pt")
    image_tensor = image_processed['pixel_values'].to(device)
    
    # Generate
    with torch.no_grad():
        captions = model.generate_caption(
            image_tensor, 
            tokenizer, 
            max_length=max_length,
            num_beams=num_beams
        )
    
    return captions[0]

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ CaptiFy AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload any image and let AI describe it for you!</p>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer, vit_processor, device = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Information")
        st.write("**Name:** CaptiFy AI")
        st.write("**Architecture:** ViT + GPT-2")
        st.write("**Training Dataset:** Flickr8k")
        st.write("**Total Epochs:** 20")
        st.write(f"**Device:** {device.upper()}")
        
        st.markdown("---")
        
        # Generation settings
        st.subheader("🎛️ Generation Settings")
        
        num_beams = st.slider(
            "Beam Search Width",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher values = better quality but slower (1=fastest, 5=recommended, 10=best quality)"
        )
        
        max_length = st.slider(
            "Maximum Caption Length",
            min_value=10,
            max_value=50,
            value=30,
            help="Maximum number of words in the caption"
        )
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.write("""
        **CaptiFy AI** uses:
        - **Vision Transformer (ViT)** to understand images
        - **GPT-2** to generate natural language
        - **Beam Search** for high-quality captions
        
        Built with PyTorch & Transformers 🔥
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
    
    with col2:
        st.subheader("🎨 Result")
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Generate caption button
            if st.button("🚀 Generate Caption", type="primary", use_container_width=True):
                with st.spinner(f"🤖 CaptiFy AI is analyzing the image (using {num_beams} beams)..."):
                    caption = generate_caption(
                        image, 
                        model, 
                        tokenizer, 
                        vit_processor, 
                        device,
                        num_beams,
                        max_length
                    )
                
                # Display caption in styled box
                st.markdown(
                    f'<div class="caption-box">💬 "{caption}"</div>', 
                    unsafe_allow_html=True
                )
                
                # Success message
                st.success("✅ Caption generated successfully!")
                
                # Display in text area for easy copying
                st.text_area(
                    "Copy Caption:", 
                    value=caption, 
                    height=100,
                    label_visibility="collapsed"
                )
        else:
            st.info("👆 Upload an image to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">CaptiFy AI | Made with ❤️ using Streamlit | Powered by ViT + GPT-2</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()