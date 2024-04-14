import streamlit as st
import tensorflow as tf 
import numpy as np
from keras.models import load_model
import json

from utils import make_gradcam_heatmap, plot_heatmaps_with_titles, analyze_neuron_influences_with_images

from llm_utils import text_model, vision_model_test


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Function to load the model
def load_model1():
    return load_model("pneumonia_reduced_params.h5")

# Function to visualize explanations
def visualize_explanations(model, image, layer_name, mode="grayscale"):
    heatmaps = make_gradcam_heatmap(image, model, layer_name)
    return heatmaps

# Create or get the session state
session_state = SessionState(model=None, selected_layer=None, visualization_mode="Grayscale", heatmaps=None)

st.title("Model Explainability App")

# Upload model file
uploaded_file = st.file_uploader("Upload your CNN model here (.h5)", type="h5")
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
                res = vision_model_test( "D:\\xcnn\\assets\\heatmap_neuron_conv2d_0.png")

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
         res = text_model(Human_Prompt)
         st.markdown(res.content,  unsafe_allow_html=True)
