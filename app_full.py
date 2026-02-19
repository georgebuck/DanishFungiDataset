import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append('/home/dell/Desktop/DanishFungiDataset')
from inference.inference import load_trained_model, build_species_lookup, get_transforms, classify_frame, build_safety_from_csv

HF_MODEL = "/home/dell/.cache/huggingface/hub/models--BVRA--vit_base_patch16_224.ft_df20_224/snapshots/f8c92e3d3d7ce2f421bad1f5411ed67902c8a3f6/pytorch_model.bin"
HF_CONFIG = "/home/dell/.cache/huggingface/hub/models--BVRA--vit_base_patch16_224.ft_df20_224/snapshots/f8c92e3d3d7ce2f421bad1f5411ed67902c8a3f6/config.yaml"
TRAIN_CSV = "/home/dell/Desktop/DanishFungiDataset/DanishFungi2024/DanishFungi2024-train.csv"

st.set_page_config(page_title="🍄 Mushroom Identifier", layout="centered")
st.title("🍄 Mushroom Identifier")
st.caption("1,604 species — AI-powered identification with safety guidance. Not a substitute for expert advice.")

@st.cache_resource
def load_model_cached():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, config = load_trained_model(HF_MODEL, HF_CONFIG, device)
    transform = get_transforms(mean, std)
    id2species = build_species_lookup(TRAIN_CSV)
    safety_data = build_safety_from_csv(TRAIN_CSV)
    return model, transform, id2species, safety_data, device

model, transform, id2species, safety_data, device = load_model_cached()

safety_colors = {
    "edible":  "#28a745",
    "caution": "#fd7e14",
    "toxic":   "#dc3545",
    "deadly":  "#7b0000",
    "unknown": "#6c757d"
}

safety_emojis = {
    "edible":  "✅",
    "caution": "⚠️",
    "toxic":   "☠️",
    "deadly":  "💀",
    "unknown": "❓"
}

def show_results(results, dataset_confidence):
    if dataset_confidence > 0.7:
        st.success(f"📊 Dataset confidence: {dataset_confidence:.0%} — likely a known species")
    elif dataset_confidence > 0.4:
        st.warning(f"📊 Dataset confidence: {dataset_confidence:.0%} — uncertain match")
    else:
        st.error(f"📊 Dataset confidence: {dataset_confidence:.0%} — probably not in our dataset")

    top = results[0]
    safety = top["safety"]
    if safety in ["deadly", "toxic"]:
        st.error(f"{safety_emojis[safety]} WARNING: Top match is {safety.upper()} — do not consume")
    elif safety == "caution":
        st.warning(f"{safety_emojis[safety]} CAUTION: Exercise care with this species")
    elif safety == "edible":
        st.success(f"{safety_emojis[safety]} Top match is generally considered edible")
    else:
        st.info("❓ Safety profile unknown — do not consume without expert verification")

    st.subheader("Top Predictions")
    for result in results:
        prob = result["probability"]
        species = result["species"]
        s = result["safety"]
        color = safety_colors.get(s, "#6c757d")
        emoji = safety_emojis.get(s, "❓")
        st.markdown(f"""
        <div style='padding:10px; margin:5px 0; border-left: 5px solid {color}; background:#1e1e1e; border-radius:4px'>
            <b style='color:white'>{emoji} {species}</b>
            <span style='float:right; color:{color}'><b>{prob:.1%}</b></span><br>
            <small style='color:#aaa'>Safety: {s.capitalize()}</small>
        </div>
        """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])

with tab1:
    camera_image = st.camera_input("Point camera at mushroom")
    if camera_image is not None:
        image = Image.open(camera_image)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Identifying..."):
            results, dataset_confidence = classify_frame(model, frame, transform, id2species, safety_data, device)
        show_results(results, dataset_confidence)

with tab2:
    uploaded_image = st.file_uploader("Upload a mushroom photo", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Identifying..."):
            results, dataset_confidence = classify_frame(model, frame, transform, id2species, safety_data, device)
        show_results(results, dataset_confidence)

st.divider()
st.caption("⚠️ Always consult an expert mycologist before consuming any wild mushroom.")
