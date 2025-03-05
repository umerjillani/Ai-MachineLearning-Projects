# cloudinary_utils.py
import cloudinary
import cloudinary.uploader
import logging
import os


# Configure Cloudinary using environment variables
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def upload_image(image_path):
    """
    Uploads an image to Cloudinary and returns its public URL.
    """
    try:
        response = cloudinary.uploader.upload(image_path)
        return response['secure_url']
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        return None