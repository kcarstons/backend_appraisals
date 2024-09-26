import psutil
import time
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, AutoModelForCausalLM
import boto3
from io import BytesIO
import pypdfium2 as pdfium 
import os
import re
import timm
import flash_attn
import einops


def read_from_s3(file_name):
    """
    Read one file in at a time.
    """
    s3 = boto3.client("s3")
    bucket = 'a1156-val'
    data_key = f"backend_appraisals/sample_data/{file_name}.pdf"
    data_location = 's3://{}/{}'.format(bucket, data_key) 
    print("location: ", data_location);

    pdf_file = s3.get_object(Bucket = bucket, Key = data_key)[
        "Body"
    ].read()

    return pdf_file


def extract_images(file_name): 
    """
    Extract all images from a pdf and store in a list. 
    """
    all_images = []
    pdf = pdfium.PdfDocument(BytesIO(read_from_s3(file_name)))

    for i in range(len(pdf)):
        page = pdf[i]
        
        for obj in page.get_objects(): 
            if obj.type == 3: 
                image = obj.get_bitmap().to_pil() 
                all_images.append(image)

    return all_images

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")  # RSS in MB




# def mm_generate_caption(pil_image):

#     inputs = processor(images=pil_image, return_tensors="pt")
#     generated_ids = model.generate(**inputs)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return generated_text


# def answer_prompt(pil_image, prompt):

#     inputs = processor(images=pil_image, text = prompt, return_tensors="pt")

#     generated_ids = model.generate(**inputs, max_new_tokens = 20)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return generated_text
    

# Example usage before running a model
print_memory_usage()
time.sleep(1)  # Simulate some processing time


# load images - nice fake house
extracted_images_sf = extract_images("appraisal_sf")
subject_images_sf = extracted_images_sf[3:12]
labels_sf = [ 'subject front', 'subject rear', 'subject street', 'kitchen', 'nook', 'living/dining', 'bedroom', 'bathroom', 'bathroom']
labeled_images_sf = [{'image':img, 'label': lbl} for img, lbl in zip(subject_images_sf, labels_sf)]

# load images - real neutral condition
extracted_images_fha = extract_images("fha_appraisal")
subject_images_fha = extracted_images_fha[11:18]
labels_fha = ['Subject Front', 'Subject Rear', 'Subject Street', 'Interior', 'Interior', 'Interior', 'Interior']
labeled_images_fha = [{'image':img, 'label': lbl} for img, lbl in zip(subject_images_fha, labels_fha)]


# Configure model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

max_new_tokens = 1024

################### LLAVA #########################
# llava_pipeline = pipeline("image-to-text", model = "llava-hf/llava-1.5-7b-hf",  model_kwargs={"quantization_config": quantization_config})
# llava_prompt = "USER: <image>\nDescribe what I am looking at it in detail. Additionally, rate the general condition from a scale of very good, good, neutral, poor, very poor. \nASSISTANT:"

# for item in labeled_images_sf:
#     pil_image = item['image']
#     label = item['label']
#     caption = llava_pipeline(pil_image, prompt=llava_prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
#     print(f"Label: {label}, Generated Caption: {caption[0]['generated_text']}")


# for item in labeled_images_fha:
#     pil_image = item['image']
#     label = item['label']
#     caption = llava_pipeline(pil_image, prompt=llava_prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
#     print(f"Label: {label}, Generated Caption: {caption[0]['generated_text']}")

################### FLORENCE2 #########################

# model_id = "microsoft/Florence-2-large"
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quantization_config, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# prompt = "<MORE_DETAILED_CAPTION>"


# def florence2(pil_image, prompt):

#     inputs = processor(images=pil_image, text = prompt, return_tensors="pt")

#     generated_ids = model.generate(
#       input_ids=inputs["input_ids"],
#       pixel_values=inputs["pixel_values"],
#       max_new_tokens=1024,
#       do_sample=False,
#       num_beams=3,
#     )

#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(pil_image.width, pil_image.height))

#     return parsed_answer


# for item in labeled_images_sf:
#     pil_image = item['image']
#     label = item['label']
#     caption = florence2(pil_image, prompt)
#     print(f"Label: {label}, Generated Caption: {caption[0]['<MORE_DETAILED_CAPTION>']}")




################### BLIP2 #########################

# def generate_caption(pil_image, prompt = None):

#     inputs = processor(images=pil_image, text = prompt, return_tensors="pt")
#     generated_ids = model.generate(**inputs, max_new_tokens = 20)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return generated_text

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")



# Check memory usage after running the model
print_memory_usage()
