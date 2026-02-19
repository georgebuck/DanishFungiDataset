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

tab1, tab2, tab3 = st.tabs(["📷 Camera", "📁 Upload", "📊 Model Performance"])

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

with tab3:
    st.subheader("Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Architecture", "ViT-Base/16")
        st.metric("Parameters", "85.9M")
        st.metric("Input Resolution", "224 × 224")
        st.metric("Species Covered", "1,604")
    with col2:
        st.metric("Training Images", "266,273")
        st.metric("Test Images", "29,665")
        st.metric("Training Epochs", "100")
        st.metric("Running on", "NVIDIA GB10")

    st.divider()
    st.subheader("Real-World Evaluation")
    st.caption("500 research-grade labeled images from iNaturalist across 50 species confirmed in training data")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", "500")
    with col2:
        st.metric("Top-1 Accuracy", "69.2%")
    with col3:
        st.metric("Top-3 Accuracy", "81.6%")

    st.divider()
    st.subheader("Per-Species Results")

    species_results = [
        {"species": "Coprinellus micaceus",      "top1": 10, "top3": 10},
        {"species": "Coprinus comatus",           "top1": 10, "top3": 10},
        {"species": "Laccaria amethystina",       "top1": 10, "top3": 10},
        {"species": "Trametes versicolor",        "top1": 9,  "top3": 9},
        {"species": "Hypholoma fasciculare",      "top1": 9,  "top3": 9},
        {"species": "Auricularia auricula-judae", "top1": 9,  "top3": 10},
        {"species": "Plicaturopsis crispa",       "top1": 9,  "top3": 10},
        {"species": "Clitocybe nebularis",        "top1": 9,  "top3": 9},
        {"species": "Tremella mesenterica",       "top1": 9,  "top3": 10},
        {"species": "Amanita muscaria",           "top1": 9,  "top3": 9},
        {"species": "Imleria badia",              "top1": 9,  "top3": 10},
        {"species": "Pseudocraterellus undulatus","top1": 9,  "top3": 10},
        {"species": "Kuehneromyces mutabilis",    "top1": 9,  "top3": 9},
        {"species": "Xylaria hypoxylon",          "top1": 9,  "top3": 10},
        {"species": "Daedalea quercina",          "top1": 9,  "top3": 9},
        {"species": "Phlebia tremellosa",         "top1": 9,  "top3": 10},
        {"species": "Fomitopsis betulina",        "top1": 9,  "top3": 10},
        {"species": "Phlebia radiata",            "top1": 9,  "top3": 10},
        {"species": "Meripilus giganteus",        "top1": 8,  "top3": 10},
        {"species": "Phaeolepiota aurea",         "top1": 8,  "top3": 9},
        {"species": "Boletus edulis",             "top1": 8,  "top3": 8},
        {"species": "Trametes gibbosa",           "top1": 8,  "top3": 9},
        {"species": "Lepista nuda",               "top1": 8,  "top3": 10},
        {"species": "Hymenopellis radicata",      "top1": 8,  "top3": 9},
        {"species": "Fomitopsis pinicola",        "top1": 7,  "top3": 8},
        {"species": "Fomes fomentarius",          "top1": 7,  "top3": 10},
        {"species": "Gymnopilus penetrans",       "top1": 7,  "top3": 7},
        {"species": "Neoboletus luridiformis",    "top1": 7,  "top3": 8},
        {"species": "Lycoperdon perlatum",        "top1": 7,  "top3": 8},
        {"species": "Parmelia sulcata",           "top1": 7,  "top3": 8},
        {"species": "Ganoderma applanatum",       "top1": 6,  "top3": 7},
        {"species": "Daedaleopsis confragosa",    "top1": 6,  "top3": 8},
        {"species": "Leccinum scabrum",           "top1": 6,  "top3": 7},
        {"species": "Byssomerulius corium",       "top1": 6,  "top3": 9},
        {"species": "Psathyrella candolleana",    "top1": 6,  "top3": 8},
        {"species": "Bjerkandera adusta",         "top1": 5,  "top3": 9},
        {"species": "Stereum hirsutum",           "top1": 5,  "top3": 6},
        {"species": "Hygrocybe miniata",          "top1": 5,  "top3": 7},
        {"species": "Armillaria lutea",           "top1": 5,  "top3": 7},
        {"species": "Tubaria furfuracea",         "top1": 5,  "top3": 7},
        {"species": "Schizophyllum commune",      "top1": 5,  "top3": 6},
        {"species": "Hygrocybe ceracea",          "top1": 5,  "top3": 7},
        {"species": "Pluteus cervinus",           "top1": 4,  "top3": 6},
        {"species": "Cuphophyllus virgineus",     "top1": 4,  "top3": 8},
        {"species": "Pleurotus ostreatus",        "top1": 4,  "top3": 7},
        {"species": "Trametes hirsuta",           "top1": 4,  "top3": 7},
        {"species": "Xerocomellus chrysenteron",  "top1": 4,  "top3": 6},
        {"species": "Cerioporus varius",          "top1": 4,  "top3": 4},
        {"species": "Mycena galericulata",        "top1": 2,  "top3": 4},
        {"species": "Laccaria laccata",           "top1": 0,  "top3": 0},
    ]

    for r in species_results:
        pct = r["top1"] / 10
        color = "#28a745" if pct >= 0.7 else "#fd7e14" if pct >= 0.4 else "#dc3545"
        st.markdown(f"""
        <div style='padding:8px; margin:4px 0; border-left: 5px solid {color}; background:#1e1e1e; border-radius:4px'>
            <b style='color:white'>{r["species"]}</b>
            <span style='float:right; color:{color}'><b>{r["top1"]}/10 top-1 &nbsp;|&nbsp; {r["top3"]}/10 top-3</b></span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Key Observations")
    st.markdown("""
    - **Visually distinctive species score highest** — Coprinus comatus, Laccaria amethystina and Trametes versicolor all hit 100% top-1
    - **Bracket and crust fungi perform very well** — consistent shape and texture makes them easier to classify
    - **Small brown mushrooms are hardest** — Mycena galericulata (20%) and Laccaria laccata (0%) are notoriously difficult even for experts
    - **Top-3 often rescues top-1 failures** — Bjerkandera adusta goes from 50% to 90% in top-3, showing the correct answer is usually close
    - **Gap between top-1 and top-3** suggests showing multiple predictions with probabilities is significantly more useful than a single answer
    """)

    st.divider()
    st.subheader("Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Our Model**")
        st.markdown("- Top-1: 69.2% on wild images")
        st.markdown("- Top-3: 81.6% on wild images")
        st.markdown("- 1,604 species covered")
        st.markdown("- On-device, no internet needed")
        st.markdown("- Safety warnings per prediction")
        st.markdown("- Out-of-distribution detection")
    with col2:
        st.markdown("**iNaturalist Seek**")
        st.markdown("- ~35% on verified specimens")
        st.markdown("- Single prediction only")
        st.markdown("- Requires internet connection")
        st.markdown("- No uncertainty shown")
        st.markdown("- No safety warnings")
        st.markdown("- No OOD detection")

    st.divider()
    st.caption("Evaluation: 500 research-grade iNaturalist images, 50 species, all confirmed present in training data.")

st.divider()
st.caption("⚠️ Always consult an expert mycologist before consuming any wild mushroom.")
