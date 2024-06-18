from flask import Flask, request, make_response, jsonify
import flask
import json, base64
from flask_cors import CORS
from io import BytesIO
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, AutoModel
import openai, requests
# openAI api
openai.api_key = "openai.api_key"

# google seach api
GOOGLE_API_KEY = "GOOGLE_API_KEY"
SEARCH_ENGINE_ID = "SEARCH_ENGINE_ID"

port = 9527
app = Flask(__name__)
CORS(app)

# load pickScore model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

# stable diffusion
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")

# Define how many steps steps to be run
n_steps = 100

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=200,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=200,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)  
    
    return probs.cpu().tolist()


def index_of_kth_largest(nums, k):
    return np.argsort(nums)[-k]

@app.route("/")
def main():
    return "<h1>Backend of the app</h1><p>GPT model API</p><p>Stable diffusion API</p>"

@app.route('/users', methods=["GET"])
def users():
    print("users endpoint reached...")
    with open("users.json", "r") as f:
        data = json.load(f)
        data.append({
            "username": "user4",
            "pets": ["hamster"]
        })
        return flask.jsonify(data)
    
@app.route('/llm', methods=["POST"])
def llm():
    enquiry = request.json["enquiry"]   
    if enquiry == "":
        return make_response(jsonify({'error': 'Empty enquiry is not allowed'}), 201)
    else:
        # print(enquiry)
        content = request.json
        messages = [
            {
            "role": 'system',
            "content": "You are a helpful AI assistant to help people with highly intellectual disabilities to solve the difficulties they are facing in their daily life. You will provide a list of instructions in steps in order to solve their problem. Please be detailed in each step. Please only provide the steps.",
            },
            {
            "role": 'user',
            "content": content['enquiry']
            },
        ]
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:personal:13mar2024:922hHa8U",
            messages=messages,
            max_tokens=100,
            temperature=0.1,
            frequency_penalty=0.4,
            presence_penalty=0.2
        )
        result = response.choices[0].message.content
        return flask.jsonify(result)
    
@app.route('/imageSearching', methods=["GET", "POST"])
def imageSearching():
    enquiry = request.json["enquiry"] 

    # Image search engine
    query = enquiry
    # constructing the URL
    # doc: https://developers.google.com/custom-search/v1/using_rest
    num = 3 # number of returned images
    safety = "active" # Search safety level (prevent violence/sexual content)
    imgSize = "large" # huge/icon/large/medium/small/xlarge/xxlarge
    fileType = "PNG"
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&searchType=image&q={query}&num={num}&start=1&safe={safety}&imgSize={imgSize}&fileType={fileType}"

    # make the API request
    data = requests.get(url).json()

    # Save the links of the first {num} images from the search
    searched_images = []
    try:

        for item in data.get("items"):
            searched_images.append(item.get("link"))

        image_buffer = BytesIO()    
        # download the images using the link and store the images
        pick=[]
        for link in searched_images:
            pick.append(Image.open(requests.get(link, stream=True, timeout=10).raw))
        
        # index of the top-1 image
        _1st = index_of_kth_largest(calc_probs(query, pick), 1)
        Image.open(requests.get(searched_images[_1st], stream=True, timeout=10).raw).save(image_buffer, format='PNG')
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8') 
    except TypeError as e:
        print(f"Request failed: {e}")
        image_base64 = None
    except UnidentifiedImageError as e:
        print(f"Request failed: {e}")
        image_base64 = None
    except requests.Timeout:
        # If a timeout exception is caught
        print("The request timed out")
        image_base64 = None
    except requests.RequestException as e:
        # For handling other kinds of requests exceptions
        print(f"Request failed: {e}")
        image_base64 = None
    return flask.jsonify({
            'msg': 'success', 
            'img': image_base64
        })

@app.route('/imageGeneration', methods=["GET", "POST"])
def imageGeneration():
    enquiry = request.json["enquiry"] 
    instruction = request.json["instruction"]
    print(type(instruction))
    # Image generation
    # prompt
    messages = [
        {
            "role": 'system',
            "content": f'You are a helpful prompt generator. A user asks a question "{enquiry}" and receive a list of instructions. You will be given an instruction which is one of the steps to solve the question asked by the user. You want to rewrite this instruction into a prompt that could be used for stable diffusion model to generate image. You only provide the rewritten prompt.',
        },
        {
            "role": 'user',
            "content": instruction
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.1,
    )
    prompt = response.choices[0].message.content
    print(f"The enquiry: {enquiry}, Convert {instruction} to {prompt}")
    negative_prompt = "NSFW, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (extra legs:1.331), (fused fingers:1.5), (too many fingers:1.5), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit, bad hands, missing fingers, (((extra arms and legs))),(((anime))), ((illustration)), cartoon, animation"

    # run both experts
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=n_steps, 
        height=768, width=768, 
        guidance_scale=8,
        num_images_per_prompt=2
    ).images

    # index of the top-1 image
    _1st = index_of_kth_largest(calc_probs(instruction, image), 1)
    print(f"Image {_1st} is better")
    # Convert PIL Image to Base64 String
    image_buffer = BytesIO()
    image[_1st].save(image_buffer, format='PNG')
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8') 
    
    # return the Base64 encoded string 
    return flask.jsonify({
                'msg': 'success', 
                'img': image_base64
           })


if __name__ == "__main__":
    app.run("localhost", port)  