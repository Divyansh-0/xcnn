# from IPython.display import Image, display
# import matplotlib as mpl
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
import keras

import textwrap
import cv2
import json
from keras.models import Model
# import PIL.Image





def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer('global_average_pooling2d').output]
    )
    # grad_model.summary()

    # print("Error HEREv1")

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        num_neurons = preds.shape[1]
        

    heatmaps = []
    for class_index in range(num_neurons):

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, class_index]
            

        grads = tape.gradient(class_channel, last_conv_layer_output)

        if grads is not None:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            heatmap = tf.reduce_sum(last_conv_layer_output[0] * pooled_grads, axis=-1)
            heatmaps.append(heatmap)
        else:
            print(f"No gradients for class {class_index}")
            heatmaps.append(tf.zeros_like(last_conv_layer_output[0][:, :, 0]))

    return heatmaps


def plot_heatmaps_with_titles(heatmaps, size=(128,128), threshold_factor=0.5, boundary_color=(0, 0, 255)):
    num_heatmaps = len(heatmaps)
    num_cols = min(num_heatmaps, 4)
    num_rows = (num_heatmaps + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))

    for i, heatmap in enumerate(heatmaps):
        if not isinstance(heatmap, np.ndarray):
            heatmap = np.array(heatmap)  # Convert to NumPy array if not already one
        resized_heatmap = cv2.resize(heatmap, dsize=size, interpolation=cv2.INTER_CUBIC)

        # # Normalize the heatmap to [0, 1]
        # resized_heatmap = (resized_heatmap - np.min(resized_heatmap)) / (np.max(resized_heatmap) - np.min(resized_heatmap))
        
        threshold = 0.2 * np.max(resized_heatmap)
        mask = np.where(resized_heatmap > threshold, 1, 0)

        # Find contours of the mask
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(resized_heatmap, [contour], -1, boundary_color, 1)

        ax = axs[i // num_cols, i % num_cols]
        ax.imshow(resized_heatmap, cmap='gray')
        ax.set_title(f'Neuron {i + 1}')
        ax.axis('off')

    for j in range(i + 1, num_rows * num_cols):
        ax = axs[j // num_cols, j % num_cols]
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(fig)


def save_heatmap_and_image(heatmap, original_image, save_path):
    plt.figure(figsize=(10, 5))
    heatmap = np.array(heatmap)
    original_image = np.array(original_image)

    # Plot original image
    print(original_image)
   
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.axis('off')

    # Plot heatmap
    heatmap_resized = cv2.resize(heatmap, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_resized, cmap='gray')
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()


def analyze_neuron_influences_with_images(model, layer_name, img_array,image_org, threshold_pos=0.001, threshold_neg=-0.001, num_buckets=10):
    # Get the layer and weights
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()[0]  # Assuming weights are at index 0
    print(f"Weight shape: {weights.shape}")

    # Initialize influence dictionaries with image data and CAM placeholders
    positive_influences, negative_influences, neutral_influences = [], [], []

    # Analyze weights for each neuron
    for neuron_index in range(weights.shape[3]):
        neuron_weights = weights[:, :, :, neuron_index]

        # Calculate mean and standard deviation
        mean_weight = np.mean(neuron_weights)
        std_dev = np.std(neuron_weights)

        # Classify influence based on thresholds
        influence_category = None
        if mean_weight > threshold_pos:
            influence_category = "positive"
        elif mean_weight < threshold_neg:
            influence_category = "negative"
        else:
            influence_category = "neutral"

        # Create influence entry with layer name, neuron number, image data, and CAM placeholder
        influence_entry = {
            "layer_name": layer_name,
            "neuron_number": neuron_index,
            "image_path": None,
            "cam_path": None,
            "mean_weight": mean_weight,
        }

        # Append influence entry to the appropriate list
        if influence_category:
            # Pre-process image for the model (replace with your pre-processing steps)
            original_image = image_org

            influence_entry["image_path"] = f"image_neuron_{layer_name}_{neuron_index}.png"  # Assign image path

            heatmaps = make_gradcam_heatmap(img_array, model, layer_name)
            heatmap = heatmaps[neuron_index]
            heatmap_path = f"assets/heatmap_neuron_{layer_name}_{neuron_index}.png"
            save_heatmap_and_image(heatmap, original_image, heatmap_path)  # Save heatmap using matplotlib
            influence_entry["heatmap_path"] = heatmap_path  # Assign heatmap path

            if influence_category == "positive":
                positive_influences.append(influence_entry)
            elif influence_category == "negative":
                negative_influences.append(influence_entry)
            else:
                neutral_influences.append(influence_entry)

    # Save results to JSON file
    result = {
        "positive": positive_influences,
        "negative": negative_influences,
        "neutral": neutral_influences,
    }

    with open('neuron_influence_results.json', 'w') as json_file:
        json.dump(result, json_file, indent=5, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, np.float32)) else x)
    return True



