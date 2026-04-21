import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_response(image,prompt):
    model = genai.GenerativeModel('gemini-pro-vision')  
    response = model.generate_content([image,prompt])
    return response.text

def input_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        
        image_parts = {
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

st.set_page_config(page_title="FoodAI")
st.header("Get nutritional values and recipies of any dishes around the world with foodAI")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your food")

submit = st.button("Track")

input_prompt = """
You are expert nutritionist that helps users understand the nutritional values, calorie content, and healthiness of the food they consume. 
The user will upload an image of any food item, and you will provide detailed information about the food, including:

Name of the food item: You should accurately identify the food item from the image.
Nutritional values: Provide a breakdown of the nutritional content, such as proteins, fats, carbohydrates, vitamins, and minerals.
Calorie count: Calculate the total calorie content of the food item.
Healthiness assessment: Determine whether the food item is healthy or not, based on its nutritional values and calorie content. 
recipe : Provide the detailed recipe to make that food item.
"""

if submit:
    try:
        image_data = input_image(uploaded_file)
        response = get_response(input_prompt, image_data)
        st.header("About your food")
        st.write(response)
    except FileNotFoundError:
        st.error("Please upload a file before submitting.")
