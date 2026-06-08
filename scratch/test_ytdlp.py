import yt_dlp
url = "https://www.youtube.com/watch?v=Mp6klx9oeZs"
ydl_opts = {
    'format': 'best[ext=mp4]/best',
    'quiet': True,
    'no_warnings': True,
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    try:
        info = ydl.extract_info(url, download=False)
        print(f"Success: {info['url']}")
    except Exception as e:
        print(f"Error: {e}")
