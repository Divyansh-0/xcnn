import json
import os
from dotenv import load_dotenv
import PIL.Image
import textwrap
import markdown






from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import google.generativeai as genai




def fm_txt(txt):
  txt = txt.replace('.' , '*')
  return markdown.markdown(textwrap.indent(txt , ">" ,predicate =lambda _: True))



load_dotenv()
API = os.getenv('GEM_API')
genai.configure(api_key = API)
os.environ["GOOGLE_API_KEY"] = API


PROMPT = """ As a healthcare assistant, you will analyze X-ray reports of patients who have tested positive 
for pneumonia. Additionally, you will receive a heatmap highlighting feature activations from a CNN model 
that classified the image as positive for pneumonia. Carefully examine the image and the heatmap patches to
 explain the reasoning behind the CNN's prediction. Explain your observations as if you are explaining to a layperson.
"""




def extract_info_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    heatmap_paths = {
        "positive": [neuron["heatmap_path"] for neuron in data["positive"]],
        "negative": [neuron["heatmap_path"] for neuron in data["negative"]],
        "neutral": [neuron["heatmap_path"] for neuron in data["neutral"]],
    }
    
    mean_weights = {
        "positive": [neuron["mean_weight"] for neuron in data["positive"]],
        "negative": [neuron["mean_weight"] for neuron in data["negative"]],
        "neutral": [neuron["mean_weight"] for neuron in data["neutral"]],
    }
    
    return heatmap_paths, mean_weights



def vision_model(prompt, img):
    model = genai.GenerativeModel('gemini-pro-vision')
    img = PIL.Image.open(img)

    res = model.generate_content([
    prompt,img] , stream = False,
        generation_config=genai.types.GenerationConfig( temperature=0.1))
    res.resolve()
    return res.text

def prefetch():
    heatmap_paths, mean_weights = extract_info_from_json("neuron_influence_results.json")
    pos = vision_model(PROMPT, heatmap_paths["positive"][0]) 
    pos = f"{pos} \n The weight for this inference is {mean_weights['positive'][0]}"
    neg = vision_model(PROMPT, heatmap_paths["negative"][0]) 
    neg = f"{neg} \n The weight for this inference is {mean_weights['negative'][0]}"
    neu = vision_model(PROMPT, heatmap_paths["neutral"][0]) 
    neu = f"{neu} \n The weight for this inference is {mean_weights['neutral'][0]}"
    return pos , neg , neu


def text_model(hprompt):
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(input_variables=[  "s1" , "s2" , "s3" ], template = hprompt)
    chain = prompt | gemini_llm
    pos , neg , neu = prefetch()
    res = chain.invoke({"s1": pos , "s2": neg , "s3" : neu})
    

    return res









# positive 







