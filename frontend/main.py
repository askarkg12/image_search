import streamlit as st
import requests
from pathlib import Path
from PIL import Image
import zipfile
import io

image_dir = Path(__file__).parent / "test3.jpg"


# Define a function to fetch results from the API
def fetch_filenames(search_query):
    # Placeholder API URL (replace with your actual API endpoint)
    api_url = "http://backend:8000"
    # Send a GET request with the search query as a parameter
    response = requests.get(api_url, params={"query": search_query}, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"{response.status_code}: Failed to fetch results. Please try again.")
        return []


def display_image_grid(images, cols=3, rows=2):
    """Display a grid of images."""
    for i, img in enumerate(images):
        if (i) % cols == 0:
            col_list = st.columns(cols)
        with col_list[i % cols]:
            st.image(images[i])


if query := st.text_input("Image Search"):
    # Fetch results from the API
    image_names = fetch_filenames(query)
    pass
