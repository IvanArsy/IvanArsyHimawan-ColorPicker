import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image

def get_dominant_colors(image, k=5):
    image = cv2.resize(image, (300, 200))
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=1e-4, random_state=42)
    kmeans.fit(image)
    
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    color_counts = Counter(labels)
    
    sorted_colors_counts = sorted(zip(colors, color_counts.values()), key=lambda x: x[1], reverse=True)
    sorted_colors = [color for color, _ in sorted_colors_counts]
    
    sorted_colors_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0] for color in sorted_colors]
    
    return sorted_colors_rgb

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def calculate_lightness(color):
    color = np.uint8([[color]])
    hls_color = cv2.cvtColor(color, cv2.COLOR_RGB2HLS)[0][0]
    return hls_color[1]

def main():
    st.title("Dominant Color Picker")
    st.write("Unggah gambar untuk mengetahui warna dominan.")
    
    uploaded_file = st.file_uploader("Pilih gambar.....", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar terunggah.', use_column_width=True)
        
        st.write("Generating palette...")
        
        k = st.slider('Jumlah warna dominan', min_value=1, max_value=10, value=5)
        
        image = np.array(image)
        
        colors = get_dominant_colors(image, k)
        
        colors = sorted(colors, key=calculate_lightness)
        
        st.write("Warna dominan:")
        cols = st.columns(k)
        
        for idx, color in enumerate(colors):
            hex_color = rgb_to_hex(color)
            with cols[idx]:
                st.write(hex_color)
                st.markdown(f"<div style='width:50px;height:50px;background-color:{hex_color}'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
