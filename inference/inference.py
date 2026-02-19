import torch
import cv2
import numpy as np
import pandas as pd
import json
from scipy.special import softmax
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from fgvc.utils.experiment import load_config, load_model

HF_MODEL = "/home/dell/.cache/huggingface/hub/models--BVRA--vit_base_patch16_224.ft_df20m_224/snapshots/660ae494751fdb38276dbdeac1e159fb16f8c155/pytorch_model.bin"
HF_CONFIG = "/home/dell/.cache/huggingface/hub/models--BVRA--vit_base_patch16_224.ft_df20m_224/snapshots/660ae494751fdb38276dbdeac1e159fb16f8c155/config.yaml"
TRAIN_CSV = "/home/dell/Desktop/DanishFungiDataset/DanishFungi2024-Mini/DanishFungi2024-Mini-train.csv"
SAFETY_JSON = "/home/dell/Desktop/DanishFungiDataset/inference/species_safety.json"

def load_trained_model(checkpoint_path, config_path, device):
    config = load_config(config_path)
    model, model_mean, model_std = load_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    return model, model_mean, model_std, config

def get_transforms(model_mean, model_std, image_size=(224, 224)):
    return Compose([
        Resize(*image_size),
        Normalize(mean=model_mean, std=model_std),
        ToTensorV2()
    ])

def build_species_lookup(train_csv_path):
    df = pd.read_csv(train_csv_path)
    return dict(zip(df["class_id"], df["species"]))

def load_safety_data(safety_json_path):
    try:
        with open(safety_json_path) as f:
            return json.load(f)
    except:
        return {}
def build_safety_from_csv(train_csv_path):
    df = pd.read_csv(train_csv_path)
    safety = {}
    for _, row in df[['species', 'poisonous']].drop_duplicates().iterrows():
        if pd.isna(row['species']):
            continue
        safety[row['species']] = {
            "safety": "toxic" if row['poisonous'] == 1 else "edible"
        }
    return safety
def get_safety_tier(species_name, safety_data):
    return safety_data.get(species_name, {}).get("safety", "unknown")

def classify_frame(model, frame, transform, id2species, safety_data, device, top_k=5):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    probs = softmax(logits.cpu().numpy(), axis=1)[0]
    top_indices = np.argsort(-probs)[:top_k]

    # Calculate entropy-based OOD confidence
    num_classes = len(probs)
    max_entropy = np.log(num_classes)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    dataset_confidence = float(1 - (entropy / max_entropy))

    results = []
    for idx in top_indices:
        species = id2species.get(idx, "Unknown")
        results.append({
            "species": species,
            "probability": float(probs[idx]),
            "safety": get_safety_tier(species, safety_data)
        })

    return results, dataset_confidence
def run_live(checkpoint_path=HF_MODEL, config_path=HF_CONFIG,
             train_csv=TRAIN_CSV, safety_json=SAFETY_JSON):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, mean, std, config = load_trained_model(checkpoint_path, config_path, device)
    transform = get_transforms(mean, std)
    id2species = build_species_lookup(train_csv)
    safety_data = load_safety_data(safety_json)

    safety_colors = {
        "edible":  (0, 200, 0),
        "caution": (0, 165, 255),
        "toxic":   (0, 0, 255),
        "deadly":  (0, 0, 180),
        "unknown": (180, 180, 180)
    }

    cap = cv2.VideoCapture(0)
    print("Camera open — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = classify_frame(model, frame, transform, id2species, safety_data, device)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 40
        for result in results[:3]:
            color = safety_colors.get(result["safety"], (180, 180, 180))
            text = f"{result['species'][:35]}: {result['probability']:.1%}"
            cv2.putText(frame, text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 55

        if results[0]["safety"] in ["deadly", "toxic"]:
            cv2.putText(frame, "WARNING: TOXIC SPECIES DETECTED",
                       (20, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Mushroom Identifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()
