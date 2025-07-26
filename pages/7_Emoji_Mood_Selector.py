import streamlit as st
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.emoji_mood_selector import create_emoji_mood_page

# Configure page
st.set_page_config(
    page_title="Emoji Mood Selector",
    page_icon="🎭",
    layout="wide"
)

# Create the emoji mood page
create_emoji_mood_page()

# Navigation is handled automatically by Streamlit's built-in page navigation