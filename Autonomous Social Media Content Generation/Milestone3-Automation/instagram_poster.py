# instagram_poster.py
import requests
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Instagram Graph API Config
INSTAGRAM_API_URL = "https://graph.instagram.com/v22.0"
INSTAGRAM_BUSINESS_ID = os.getenv("INSTAGRAM_BUSINESS_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

def publish_single_post(image_url, caption):
    """
    Publishes ONE post per script execution using Instagram Graph API.
    Returns media ID if successful, None otherwise.
    """
    try:
        # Step 1: Create Media Container
        container_url = f"{INSTAGRAM_API_URL}/{INSTAGRAM_BUSINESS_ID}/media"
        params = {
            "image_url": image_url,
            "caption": caption,
            "access_token": ACCESS_TOKEN
        }
        
        response = requests.post(container_url, params=params)
        if not response.ok:
            logging.error(f"Container error: {response.json()}")
            return None
        
        creation_id = response.json().get("id")
        
        # Step 2: Publish the Container
        publish_url = f"{INSTAGRAM_API_URL}/{INSTAGRAM_BUSINESS_ID}/media_publish"
        params = {
            "creation_id": creation_id,
            "access_token": ACCESS_TOKEN
        }
        
        response = requests.post(publish_url, params=params)
        if not response.ok:
            logging.error(f"Publish error: {response.json()}")
            return None
        
        logging.info("Success! Post published via Instagram Graph API.")
        return creation_id  # Return media ID instead of True

    except Exception as e:
        logging.error(f"API failure: {str(e)}")
        return None