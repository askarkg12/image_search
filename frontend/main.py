import streamlit as st
import requests
from pathlib import Path
from PIL import Image
import zipfile
import io
from minio import Minio
from dotenv import dotenv_values

config = dotenv_values(".env")

minio_client = Minio(
    config["MINIO_SERVER"],
    access_key=config["MINIO_ACCESS_KEY"],
    secret_key=config["MINIO_SECRET_KEY"],
    secure=False,
)

API_URL = config["API_URL"]


# Define a function to fetch results from the API
def fetch_filenames(search_query):
    # Placeholder API URL (replace with your actual API endpoint)
    api_url = API_URL
    # Send a GET request with the search query as a parameter
    response = requests.get(api_url, params={"query": search_query}, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data["image_files"]
    else:
        st.error(f"{response.status_code}: Failed to fetch results. Please try again.")
        return []


def display_image_grid(images, cols=3):
    """Display a grid of images."""
    for i, img in enumerate(images):
        if (i) % cols == 0:
            col_list = st.columns(cols)
        with col_list[i % cols]:
            st.image(images[i])


def fetch_images(image_names, num_images: int = 20):
    images = []
    for filename in image_names[:num_images]:
        try:
            response = minio_client.get_object("images", filename.strip())
            img = Image.open(io.BytesIO(response.data))
            images.append(img)

            # Close the response stream
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return images


if query := st.text_input("Image Search"):
    with st.spinner("Loading..."):
        # Fetch results from the API
        image_names = fetch_filenames(query)
        images = fetch_images(image_names)

    if images:
        display_image_grid(images)
    else:
        st.warning("No images found for your query.")
