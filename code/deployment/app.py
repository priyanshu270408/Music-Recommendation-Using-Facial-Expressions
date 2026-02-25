import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from tensorflow.keras.models import load_model # Changed to tensorflow.keras
import numpy as np
import requests
import re
import av # Required for new webrtc

# Load model and labels
# Ensure this path is actually correct on your machine
try:
    model = load_model("code/model/fer2013_mini_XCEPTION.102-0.66.hdf5")
except Exception as e:
    st.error(f"Error loading model: {e}. Please check the file path.")
    st.stop()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.set_page_config(page_title="Emotion-Based Music Player", layout="centered")
st.title("Facial Emotion Recognition App")

# App state
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "Neutral"
if "show_video" not in st.session_state:
    st.session_state.show_video = False

# UPDATED: Streamlit WebRTC Video Processor
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_emotion = "Neutral"

    # In new API, 'transform' is replaced by 'recv'
    def recv(self, frame):
        # Convert av.VideoFrame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            
            # Prediction
            preds = model.predict(roi)[0]
            self.last_emotion = emotions[np.argmax(preds)]
            
            # Draw rectangle and text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, self.last_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            break # Only detect one face to save performance

        # Return a new av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 🎥 Live Camera Detection Mode
if not st.session_state.show_video:
    st.subheader("📷 Capturing Your Live Emotions")
    col1, col2 = st.columns([1, 2])

    with col1:
        capture = st.button("🎵 Play Song on Last Captured Emotion")

    with col2:
        # UPDATED: key, video_processor_factory (instead of transformer), and rtc_configuration
        ctx = webrtc_streamer(
            key="emotion",
            video_processor_factory=EmotionProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        )

    if capture:
        # UPDATED: Access video_processor instead of video_transformer
        if ctx.video_processor:
            st.session_state.last_emotion = ctx.video_processor.last_emotion
            st.session_state.show_video = True
            st.rerun()
        else:
            st.warning("Please start the camera first.")

# 🎧 Play Song For Detected Mood
if st.session_state.show_video:
    st.markdown("## 🎧 Now Playing Music For Your Mood")
    st.markdown(f"**Last Detected Mood:** `{st.session_state.last_emotion}`")

    if st.button("🔁 Detect Emotions Again"):
        st.session_state.show_video = False
        st.rerun()

    # Search Logic
    search_query = f"https://www.youtube.com/results?search_query={st.session_state.last_emotion}+background+music"
    
    # Added headers to mimic a browser, improves scraping success rate slightly
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(search_query, headers=headers)
    html_content = response.text
    
    # Improved Regex to be slightly more robust
    match = re.search(r'\"videoId\":\"(.*?)\"', html_content)
    
    if match:
        video_id = match.group(1)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        st.video(video_url)
    else:
        st.error("Could not find a video. YouTube scraping might be blocked.")
        # Fallback link
        st.markdown(f"[Click here to search manually on YouTube]({search_query})")