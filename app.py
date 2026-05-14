import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["NORMAL", "PNEUMONIA"]

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(
    torch.load("models/resnet18_pneumonia_model.pth", map_location=device)
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("🩻 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the trained ResNet18 model will classify it as **NORMAL** or **PNEUMONIA**.")

st.info("This application is a coursework prototype and should not be used as a real medical diagnosis tool.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-ray Image")
    st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    normal_probability = probabilities[0][0].item() * 100
    pneumonia_probability = probabilities[0][1].item() * 100

    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_index]
    confidence_score = probabilities[0][predicted_index].item() * 100

    st.subheader("Prediction Result")

    if predicted_class == "PNEUMONIA":
        st.error(f"Predicted Class: {predicted_class}")
    else:
        st.success(f"Predicted Class: {predicted_class}")

    st.metric("Confidence Score", f"{confidence_score:.2f}%")

    st.subheader("Class Probability")
    st.write(f"Normal: {normal_probability:.2f}%")
    st.progress(normal_probability / 100)

    st.write(f"Pneumonia: {pneumonia_probability:.2f}%")
    st.progress(pneumonia_probability / 100)

    st.caption("The prediction is based on the trained ResNet18 model selected as the best-performing model in the notebook.")