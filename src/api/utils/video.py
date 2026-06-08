# src/api/utils/video.py
import yt_dlp
import logging

logger = logging.getLogger("traffic-api.utils.video")

def get_stream_url(url: str) -> str:
    """
    Get a direct stream URL from a YouTube or other video platform link.
    If the URL is already a direct link, it returns it as is.
    """
    if "youtube.com" in url or "youtu.be" in url:
        logger.info(f"Extracting YouTube stream URL for: {url}")
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return info['url']
            except Exception as e:
                logger.error(f"Failed to extract YouTube URL: {e}")
                raise ValueError(f"Could not extract video stream from YouTube URL: {e}")
    
    return url
