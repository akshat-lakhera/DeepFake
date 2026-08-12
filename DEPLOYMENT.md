# 🚀 Deployment Guide — DeepGuard AI

Multiple deployment options depending on your needs.

---

## Option 1: Local Development

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

---

## Option 2: Docker

```bash
# Build
docker build -t deepguard-ai .

# Run
docker run -p 7860:7860 deepguard-ai

# Open http://localhost:7860
```

---

## Option 3: HuggingFace Spaces (Free Hosting)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Upload these files to the Space:
   - `app.py`
   - `requirements.txt`
   - `inference/` folder
   - `streamlit/my_model.keras` (your trained model)
4. The Space will auto-deploy and give you a shareable URL

### Required `requirements.txt` for Spaces:
```
tensorflow>=2.15.0
gradio>=4.31.0
opencv-python-headless>=4.9.0
Pillow>=10.0.0
numpy>=1.24.0
mtcnn>=0.1.1
```

---

## Option 4: Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set the main file to `streamlit/app.py` (the legacy Streamlit app)

> Note: The primary app is now Gradio-based (`app.py`). The Streamlit version in `streamlit/app.py` is the legacy version.

---

## Notes

- **Model file**: The trained `.keras` model is ~70-80MB. For HuggingFace Spaces, you can use Git LFS or upload directly.
- **GPU**: The app runs inference on CPU by default. GPU is only needed for training.
- **Cold start**: First request may take 10-30 seconds while the model loads into memory.
