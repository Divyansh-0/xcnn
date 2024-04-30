import streamlit as st
import tensorflow as tf 
import numpy as np
from keras.models import load_model
import json
import os

from streamlit.components.v1 import html

import time


import re




from utils import make_gradcam_heatmap, plot_heatmaps_with_titles, analyze_neuron_influences_with_images

from llm_utils import text_model, vision_model_test , vision_model , text_model_rlhf , flow


if "svg_height" not in st.session_state:
    st.session_state["svg_height"] = 200





st.set_page_config(page_title="Deconvolve", page_icon=":microscope:")

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

with st.sidebar:
    add_selectbox = st.sidebar.selectbox(
    "Select Mode",
    ("Inference", "Evaluation", "Educate")
)

# Function to load the model
def load_model1():
    return load_model("pneumonia_reduced_params.h5")

# Function to visualize explanations
def visualize_explanations(model, image, layer_name, mode="grayscale"):
    heatmaps = make_gradcam_heatmap(image, model, layer_name)
    return heatmaps

# session state
session_state = SessionState(model=None, selected_layer=None, visualization_mode="Grayscale", heatmaps=None , res_content = None)


if add_selectbox == "Inference":
    
    st.title("Deconvolve")
    st.progress(100)
    with st.container():
     st.empty()

    with st.container():
     st.empty()

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
        
         st.write("> **Helps you manage a plethora of options**")
         st.write("> **Rapidly Visualise and Obtain explanations**")


    # Column 2
    with col2:
        with st.container():
        
         st.write("> **Navigate and understand with ease**")
         st.write("> **Customise the framework to work with any CNN**" )


    with st.container():
     st.empty()

    with st.container():
     st.empty()


    st.markdown("<p style='font-size:18px;'>Click <a href='https://docs.google.com/presentation/d/1b-tBEulWrQ48l2fVd0TL4orwDk8XiSam/edit#slide=id.p6' target='_blank'> here</a> to learn more about our framework</p>", unsafe_allow_html=True)

    with st.container():
     st.empty()
    # Upload model file
    uploaded_file = st.file_uploader("Start by uploading your CNN model here (.h5)", type="h5")
    if uploaded_file is not None:
        session_state.model = load_model1()
        st.success("Model loaded successfully!")

        # Select visualization mode
        session_state.visualization_mode = st.selectbox("HeatMap Visualization Mode", ["Grayscale", "Color"])

        # Select layer
        if hasattr(session_state.model, 'layers'):
            layer_names = [layer.name for layer in session_state.model.layers]
            session_state.selected_layer = st.selectbox("Layer to Visualize", layer_names)
        else:
            st.warning("Model structure not accessible. Layer selection unavailable.")
            session_state.selected_layer = None

        chat_mod = st.selectbox("Select the LLM model" , ["Gemini-Pro" , "Claude-opus"])

        # Upload image for explanation
        image_file = st.file_uploader("Upload image for explanation", type=["jpg", "jpeg", "png"])

        if image_file is not None and session_state.model is not None and session_state.selected_layer is not None:
            image = tf.keras.preprocessing.image.load_img(image_file, target_size=(120, 120))  # Adjust size as needed
            image = tf.keras.preprocessing.image.img_to_array(image)
            img = image / 255.0  # Normalize image (adjust if needed)
            
            image = np.expand_dims(img, axis=0)  # Add batch dimension

            # Generate explanation using chosen visualization technique
            session_state.heatmaps = visualize_explanations(session_state.model, image, session_state.selected_layer, mode=session_state.visualization_mode)

            # Display original image and explanation
           
            st.image(image[0], use_column_width=False , caption="Test Image")
            st.subheader("Explanation Heatmap ({})".format(session_state.visualization_mode))
            if session_state.heatmaps:
                plot_heatmaps_with_titles(session_state.heatmaps)

            #Vision Inference
            st.subheader("Vision Model Prompt")
            Vis_prmpt = st.text_area(""" As a healthcare assistant, you will analyze X-ray reports of patients who have tested positive 
    for pneumonia. Additionally, you will receive a heatmap highlighting feature activations from a CNN model 
    that classified the image as positive for pneumonia. Carefully examine the image and the grayscale heatmap alongwith test X-ray to
     explain the reasoning behind the CNN's prediction.The white portion highlight the activation. Explain your observations as if you are
       explaining to a layperson.""")
              
            if st.button("View Vision model inference"):
                im = st.image("assets/heatmap_neuron_conv2d_0.png" , caption="Vision Model Inference Image")
                if Vis_prmpt: 
                 res = vision_model_test( "D:\\xcnn\\assets\\heatmap_neuron_conv2d_0.png" , Vis_prmpt)
                else: 
                    res = vision_model( "D:\\xcnn\\assets\\heatmap_neuron_conv2d_0.png")

                st.write(res)

            
            # Analyse Each neuron 
            st.title("Neuron Influence Analysis")

            if st.button("Display Neuron Influence Results"):
                # if analyze_neuron_influences_with_images(session_state.model, session_state.selected_layer, image , img):
                    with open('neuron_influence_results.json', 'r') as json_file:
                        data = json.load(json_file)

                    st.header("Positive Influences")
                    st.json(data["positive"])
                    st.header("Neutral Influences")
                    st.json(data["neutral"])

                    st.header("Negative Influences")
                    st.json(data["negative"])



            # Display explanation text (replace with explanation logic)
            st.header("Model Decision Explanation ")
          
          
            s1 , s2, s3 = "" , ""  ,""
            Human_Prompt = f"""
    Goal:

    Your objective is to conduct a thorough analysis of these inferences and their associated weights to provide a precise and comprehensive explanation of the CNN's prediction. Utilize the provided inferences to derive reasoning without making assumptions.

    Presentation:

    Present your analysis in a clear and understandable manner, as if explaining to a layperson. Pay close attention to the given inferences and ensure clarity in your explanations.

    Output Format:

    Influence Category:
    Mean Weight:
    Region of Lung Captured By this Category:
    Mention the features in this region indicative of pneumonia.
    Inference:
    Overall Analysis:

    Explain in simple terms why do you feel model classfied it as pneumonia. Provide strong reasoning. Avoid assuming additional information.

    Given:
    - The inference of neurons having a positive influence on prediction, along with its mean weight: {s1}
    - The inference of neurons having a negative influence on prediction, along with its mean weight: {s2}
    - The inference of neurons having a neutral influence on prediction, along with its mean weight: {s3}

            """
            if session_state.heatmaps:
                bu = st.toggle("French")
                if bu:
                    Human_Prompt += "Please answer in French Language"
                if session_state.res_content : 
                    st.markdown(st.session_state["res_content"],  unsafe_allow_html=True)
                else: 
             
                  res = text_model(Human_Prompt)
                  st.session_state["res_content"] = res.content
                  st.markdown(st.session_state["res_content"],  unsafe_allow_html=True)
            

            fl = st.button("View Flowchart")
            if fl: 
                def mermaid(code: str) -> None:
                        html(
                            f"""
                            <pre class="mermaid">
                                {code}
                            </pre>

                            <script type="module">
                                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                                mermaid.initialize({{ startOnLoad: true }});
                            </script>
                            """,
                            height=st.session_state["svg_height"] + 500,
                        )
                markdown_file_path = "flow.txt"
                if os.path.exists(markdown_file_path):

                 with open(markdown_file_path, "r") as file:
                   res = file.read()
                   mermaid(res)
                else:
      
                 r1_prmpt = f"""Given this response {res.content}, Convert it into a flowchart. Just Produce a Mermaid code for a concise flowchart. And dont output any text which is not mermaid code """
                 code = flow(r1_prmpt)
                 cc = f"{code.content}"
              
                 pattern = r'mermaid[\s\S]*?'

                 m = re.sub(pattern, '', cc)    
                 p = r'```'
                 m_new = re.sub(p, '', m)
                 p_l = r'[\s\S]*?graph TD'
                 m_1 = re.sub(p_l, 'graph TD', m_new, count=1)

                 print(m_1)


                 mermaid(m_1)


elif add_selectbox == "Evaluation":
    st.image("D:\\xcnn\\st.jpg")
    st.warning("ENTERING EVAL MODE")

    # Define the layout for responses using columns
    s1 , s2, s3 = "" , ""  ,""
    Human_Prompt = f"""
    Goal:

    Your objective is to conduct a thorough analysis of these inferences and their associated weights to provide a precise and comprehensive explanation of the CNN's prediction. Utilize the provided inferences to derive reasoning without making assumptions.

    Presentation:

    Present your analysis in a clear and understandable manner, as if explaining to a layperson. Pay close attention to the given inferences and ensure clarity in your explanations.

    Output Format:

    Influence Category:
    Mean Weight:
    Region of Lung Captured By this Category:
    Mention the features in this region indicative of pneumonia.
    Inference:
    Overall Analysis:

    Explain in simple terms why do you feel model classfied it as pneumonia. Provide strong reasoning. Avoid assuming additional information.

    Given:
    - The inference of neurons having a positive influence on prediction, along with its mean weight: {s1}
    - The inference of neurons having a negative influence on prediction, along with its mean weight: {s2}
    - The inference of neurons having a neutral influence on prediction, along with its mean weight: {s3}

            """
    col1, col2 = st.columns(2)
    markdown_file_path = "markdown_content.md"
    if os.path.exists(markdown_file_path):

      with open(markdown_file_path, "r") as file:
        res = file.read()
    
    else:
      
      res = text_model(Human_Prompt)
      res = res.content



    with col1:
        st.markdown('#### Response 1')
        with st.container(border=True):
          time.sleep(20)
          st.markdown(res, unsafe_allow_html=True) 
        
     

        with st.popover("Liked the response"):
            st.markdown("Your Feedback on Response 1")
            user_input = st.text_area("Enter" , key=2)
            if user_input: 
             st.write("Feedback Submitted!")

                

    with col2:
        st.markdown('#### Response 2')
        r1_prmpt = f"Given this response {res} . Reframe it too be more technical and more crisp and concise"
        # r1_prmpt = f"Given this response {res} . Reframe it to a flowchart that can be displayed on a code editor"
        res1 = text_model_rlhf(r1_prmpt)
        with st.container(border=True):
          st.markdown(res1.content , unsafe_allow_html=True) 
        # 
        # st.markdown(res1.content)
        with st.popover("Liked the response"):
            st.markdown("Your Feedback on Response 1")
            user_input = st.text_area("Enter" , key=1)
            if user_input: 
             st.write("Feedback Submitted!")

elif add_selectbox == "Educate":
    st.title("Deconvolve")
    st.progress(100)
    with st.container():
     st.empty()
    st.write("Does your model operate in a domain that's outside the LLM's ares of expertise ?")
    with st.container():
     st.empty()
    st.write("No worries, just upload documents and images elucidating your domain knowledge and the LLM will learn it all up  ")
    st.markdown("<p style='font-size:18px;'>( This works using <a href='https://paperswithcode.com/method/rag' target='_blank'> R.A.G )</a> </p>", unsafe_allow_html=True)

    up_file = st.file_uploader("Start by uploading your documents")
    if up_file is not None:
        
        st.success("Document uploaded successfully!")


    

    
     

