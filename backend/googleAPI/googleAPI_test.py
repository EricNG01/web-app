from flask import request
import requests
import json, base64
from PIL import Image
from io import BytesIO
import base64

# google seach api
GOOGLE_API_KEY = "API KEY"
SEARCH_ENGINE_ID = "SEARCH_ENGINE_ID"

enquiry = "Go to facebook.com"

# Image search engine
query = enquiry
# constructing the URL
# doc: https://developers.google.com/custom-search/v1/using_rest
num =10 # number of returned images
safety = "active" # Search safety level (prevent violence/sexual content)
imgSize = "large" # huge/icon/large/medium/small/xlarge/xxlarge
fileType = "PNG"
url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&searchType=image&q={query}&num={num}&start=1&safe={safety}&imgSize={imgSize}&fileType={fileType}"


# make the API request
data = requests.get(url).json()

# Save the links of the first {num} images from the search
searched_images = []
for item in data.get("items"):
    searched_images.append(item.get("link"))

pick=[]
for link in searched_images:
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # Or PNG, etc.
    encoded_img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    pick.append(encoded_img_str)
print(pick.__len__())
