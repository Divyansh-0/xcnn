import streamlit as st
import tensorflow as tf 
import numpy as np
from keras.models import load_model
import json

from utils import make_gradcam_heatmap, plot_heatmaps_with_titles, analyze_neuron_influences_with_images

from llm_utils import text_model, fm_txt







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
uploaded_file = st.file_uploader("Upload your model (.h5)", type="h5")
if uploaded_file is not None:
    session_state.model = load_model1()
    st.success("Model loaded successfully!")

    # Select visualization mode
    session_state.visualization_mode = st.selectbox("Visualization Mode", ["Grayscale", "Color"])

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
        st.subheader("Original Image")
        st.image(image[0], use_column_width=True)
        st.subheader("Explanation Heatmap ({})".format(session_state.visualization_mode))
        if session_state.heatmaps:
            plot_heatmaps_with_titles(session_state.heatmaps)
        
        # Analyse Each neuron 
        st.title("Neuron Influence Analysis")

        if st.button("Display Neuron Influence Results"):
            # if analyze_neuron_influences_with_images(session_state.model, session_state.selected_layer, image , img):
                with open('neuron_influence_results.json', 'r') as json_file:
                    data = json.load(json_file)

                st.header("Positive Influences")
                st.json(data["positive"])

                st.header("Negative Influences")
                st.json(data["negative"])

                st.header("Neutral Influences")
                st.json(data["neutral"])

        # Display explanation text (replace with explanation logic)
        st.subheader("Model Decision Explanation (Placeholder)")
        s1 , s2, s3 = "" , ""  ,""
        Human_Prompt = f"""
        You are a healthcare analyst tasked with interpreting the inferences and neuron weights provided by a 
        vision model for each neuron in the heatmap of a CNN model analyzing X-ray images for pneumonia. 
        The vision model has generated multiple inferences for each neuron, along with their respective weights. 
        Your goal is to analyze these inferences and weights to provide a clear and comprehensive analysis explaining the reasoning
        behind the CNN's prediction. Present your analysis in a clear and understandable manner, as if you are explaining to a layperson.

        The inference of neurons having positive influence on prediction along with its mean weight {s1} 
        The inference of neurons having negative influence on prediction along with its mean weight {s2} 
        The inference of neurons having neutral influence on prediction along with its mean weight {s3} 
        """
        if session_state.heatmaps:
         res = text_model(Human_Prompt)
         st.write(st.markdown(res.content,  unsafe_allow_html=True))
