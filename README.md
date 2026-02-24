# 🖼️ CaptiFy AI

AI-powered image captioning using Vision Transformer (ViT) and GPT-2.

## 🚀 Live Demo

Try it on Streamlit Cloud! (Link will be available after deployment)

## 🧠 Model Architecture

- **Vision Encoder:** ViT-Base (86M parameters, frozen)
- **Language Decoder:** GPT-2 (125M parameters)
- **Training Dataset:** Flickr8k (6,000 images)
- **Training Epochs:** 20
- **Best Validation Loss:** 0.6425

## 💻 Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run src/streamlit_app.py
```

## 📊 Features

- Upload any image and get AI-generated captions
- Adjustable beam search for quality control
- Configurable caption length
- Fast inference with model caching

## 🛠️ Tech Stack

- PyTorch
- Transformers (HuggingFace)
- Streamlit
- Vision Transformer (ViT)
- GPT-2

---

Built with ❤️ for image understanding
