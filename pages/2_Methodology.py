# A comprehensive explanation of the data flows and implementation details.
# A flowchart illustrating the process flow for each use case in the application. Each use case should have its own flowchart.
# Refer to the sample here for examples of flowcharts and methodology (Slides 13, 14, and 15)

import streamlit as st
from PIL import Image
from pathlib import Path

st.title("Methodology")

img_path = Path("/mount/src/aibc_2026_aug/data/method.jpg")
img = Image.open(img_path)
st.image(img, use_container_width=True)