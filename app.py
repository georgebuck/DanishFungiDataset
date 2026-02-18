import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/dell/Desktop/DanishFungiDataset')
from inference.inference import load_trained_model, build_species_lookup, get_transforms, classify_frame, load_safety_data

HF_MODEL = "/home/dell/.cache/huggingface/hub/models--BVRA--vit_base_patch16_224.ft_df20m_224/snapshots/660ae494751fdb38276dbdeac1e159fb16f8c155/pytorch_model.bin"
HF_CONFIG = "/home/dell/.cache/huggingface/hub/models--BVRA--vit_base_patch16_224.ft_df20m_224/snapshots/660ae494751fdb38276dbdeac1e159fb16f8c155/config.yaml"
TRAIN_CSV = "/home/dell/Desktop/DanishFungiDataset/DanishFungi2024-Mini/DanishFungi2024-Mini-train.csv"
SAFETY_JSON = "/home/dell/Desktop/DanishFungiDataset/inference/species_safety.json"

st.set_page_config(page_title="🍄 Mushroom Identifier", layout="centered")
st.title("🍄 Mushroom Identifier")
st.caption("AI-powered species identification with safety guidance. Not a substitute for expert advice.")

@st.cache_resource
def load_model_cached():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, config = load_trained_model(HF_MODEL, HF_CONFIG, device)
    transform = get_transforms(mean, std)
    id2species = build_species_lookup(TRAIN_CSV)
    safety_data = load_safety_data(SAFETY_JSON)
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

camera_image = st.camera_input("Point your camera at a mushroom")

if camera_image is not None:
    image = Image.open(camera_image)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    with st.spinner("Identifying..."):
        results = classify_frame(model, frame, transform, id2species, safety_data, device)
    
    top = results[0]
    safety = top["safety"]
    
    # Big safety banner for dangerous species
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

    st.divider()
    st.caption("⚠️ Always consult an expert mycologist before consuming any wild mushroom. This tool is for educational purposes only.")
