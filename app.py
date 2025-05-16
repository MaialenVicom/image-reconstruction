import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import albumentations as A
import time
import cv2
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim_sk, peak_signal_noise_ratio as psnr_sk
import torch.nn.functional as F
import lpips
from innitius_enhance_shadow_light_specular import (
    add_uneven_illumination,
    add_random_shadows,
    add_streak_reflections,
    add_elliptical_reflections
)
import pandas as pd
from skimage.measure import shannon_entropy

# ---------------------------
# TRANSFORMACIONES
# ---------------------------
alb_transforms = [
    ("RandomBrightnessContrast", A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=1.0)),
    ("GaussianBlur", A.Blur(blur_limit=(3, 5), p=1.0)),
    ("GaussNoise", A.GaussNoise(std_range=(0.05, 0.1), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=(3, 5), p=1.0))
]

custom_transforms = [
    ("UnevenIllumination", add_uneven_illumination),
    ("RandomShadows", add_random_shadows),
    ("StreakReflections", add_streak_reflections),
    ("EllipticalReflections", add_elliptical_reflections),
]

todas = alb_transforms + custom_transforms
nombre_transformaciones = [n for n, _ in todas]

# ---------------------------
# MODELO AUTOENCODER
# ---------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True))

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 64, 3, padding=1), nn.Sigmoid())
        self.residual = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)

        y = self.dec1(x5)
        y = self.up1(y)
        y = torch.cat([y, x3], dim=1)
        y = self.dec2(y)
        y = self.up2(y)
        y = torch.cat([y, x1], dim=1)
        y = self.dec3(y)
        residual = self.tanh(self.residual(y))
        out = torch.clamp(x + residual, 0.0, 1.0)
        return out

# ---------------------------
# FUNCIONES AUXILIARES
# ---------------------------
def cargar_imagen(img_file):
    imagen = Image.open(img_file).convert('RGB')
    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor()
    ])
    return transform(imagen).unsqueeze(0)

def tensor_a_pil(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    return T.ToPILImage()(tensor)

def aplicar_transformacion_especifica(tensor_img, nombre_transformacion):
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    for nombre, transformacion in todas:
        if nombre == nombre_transformacion:
            if isinstance(transformacion, A.BasicTransform):
                img_np = A.Compose([transformacion])(image=img_np)['image']
            else:
                img_np = transformacion(img_np)
            break

    tensor_resultado = T.ToTensor()(Image.fromarray(img_np)).unsqueeze(0)
    return tensor_resultado

def calcular_metricas(img1_pil, img2_pil, img1_tensor, img2_tensor):
    img1_pil = img1_pil.convert('RGB')
    img2_pil = img2_pil.convert('RGB')

    img1_np = np.array(img1_pil.resize((320, 320))).astype(np.float32) / 255.0
    img2_np = np.array(img2_pil.resize((320, 320))).astype(np.float32) / 255.0

    # Forzar que tenga 3 canales
    if img1_np.ndim == 2:
        img1_np = np.stack([img1_np] * 3, axis=-1)
    if img2_np.ndim == 2:
        img2_np = np.stack([img2_np] * 3, axis=-1)

    ssim_val = ssim_sk(img1_np, img2_np, data_range=1.0, channel_axis=2)
    psnr_val = psnr_sk(img1_np, img2_np, data_range=1.0)
    mse_val = F.mse_loss(img1_tensor.to(dispositivo), img2_tensor.to(dispositivo)).item()

    def normalizar_lpips(t):
        return (t * 2) - 1

    lpips_val = lpips_fn(normalizar_lpips(img1_tensor).to(dispositivo),
                         normalizar_lpips(img2_tensor).to(dispositivo)).item()

    return ssim_val, psnr_val, mse_val, lpips_val

def calculate_entropy(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(img)

def calculate_sharpness(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()
    mean_intensity = np.mean(img)
    normalized_sharpness = variance / (mean_intensity + 1e-5)
    return normalized_sharpness

def calculate_contrast(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.std()

def calculate_colorfulness(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_std = np.std(rg)
    yb_std = np.std(yb)
    colorfulness = np.sqrt(rg_mean ** 2 + yb_mean ** 2) + 0.3 * np.sqrt(rg_std ** 2 + yb_std ** 2)
    return colorfulness

def calculate_1image_metrics(image):
    entropy = calculate_entropy(image)
    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    colorfulness = calculate_colorfulness(image)
    return round(entropy, 2), round(sharpness, 2), round(contrast, 2), round(colorfulness, 2)

# ---------------------------
# STREAMLIT UI SETUP
# ---------------------------

st.set_page_config(page_title="Restaurador de Imágenes - Vicomtech", layout="wide")

# --- Estilos CSS para tablas grisáceas y tamaños de imagen ---
st.markdown("""
    <style>
    /* Fondo gris claro para tablas */
    table.dataframe tbody tr {
        background-color: #f5f5f5;
    }
    /* Alternar filas para mejor legibilidad */
    table.dataframe tbody tr:nth-child(even) {
        background-color: #e0e0e0;
    }
    /* Encabezados tabla */
    table.dataframe thead th {
        background-color: #b0b0b0;
        color: black;
        font-weight: bold;
    }
    /* Ajustar tamaño imágenes */
    .image-container img {
        max-width: 250px;
        height: auto;
    }
    /* Cabecera */
    header {
        display: flex;
        align-items: center;
        padding: 10px 20px;
        background-color: #0a3d62;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    header img {
        height: 50px;
        margin-right: 15px;
    }
    header h1 {
        font-size: 28px;
        margin: 0;
    }
    /* Pie de página */
    footer {
        margin-top: 50px;
        padding: 15px 20px;
        text-align: center;
        font-size: 12px;
        color: #888888;
        border-top: 1px solid #cccccc;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Cabecera con logo y título
st.markdown("""
<header>
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_3yi4I09vqmWhm478F6IWs6RJKW96k9froA&s" />
    <h1>🧽 Restaurador de Imágenes - Centro Tecnológico Vicomtech</h1>
</header>
""", unsafe_allow_html=True)

st.markdown("""
Este sistema permite **eliminar artefactos** en imágenes mediante una red neuronal convolucional autoencoder con skip connections.

1. Sube una imagen.
2. Elige qué tipo de degradación aplicar.
3. El modelo intentará restaurarla y mostrar métricas cuantitativas.
""")

archivo_subido = st.file_uploader("📤 Sube tu imagen aquí:", type=["png", "jpg", "jpeg"])

nombre_seleccionado = st.selectbox("🛠️ Elige una degradación para aplicar:", nombre_transformaciones)

if archivo_subido and nombre_seleccionado:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net='alex').to(dispositivo)
    modelo = ConvAutoencoder().to(dispositivo)
    checkpoint = torch.load("autoencoder/Buenos/checkpoint_1.pt", map_location=dispositivo)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()

    img_tensor = cargar_imagen(archivo_subido)
    img_ruidosa = aplicar_transformacion_especifica(img_tensor, nombre_seleccionado)

    with torch.no_grad():
        img_restaurada = modelo(img_ruidosa.to(dispositivo))

    original = tensor_a_pil(img_tensor)
    ruidosa = tensor_a_pil(img_ruidosa)
    restaurada = tensor_a_pil(img_restaurada)

    # Mostrar imágenes en fila con columnas, centradas y tamaño 350px
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='text-align: center;'>"
                    "<h3>📷 Imagen Original</h3>", unsafe_allow_html=True)
        st.image(original, use_container_width=False, width=450)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div style='text-align: center;'>"
                    f"<h3>🌫️ Con {nombre_seleccionado}</h3>", unsafe_allow_html=True)
        st.image(ruidosa, use_container_width=False, width=450)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='text-align: center;'>"
                    "<h3>✨ Restauración</h3>"
                    "</div>", unsafe_allow_html=True)
        placeholder = st.empty()
        restaurada_np = np.array(restaurada).astype(np.float32)
        negra = np.zeros_like(restaurada_np)
        steps = 20
        for alpha in np.linspace(0, 1, steps):
            frame = (negra * (1 - alpha) + restaurada_np * alpha).astype(np.uint8)
            placeholder.image(frame, use_container_width=False, width=450)
            time.sleep(0.07)


        # --- Cálculo de métricas con referencia ---
    ssim_rd, psnr_rd, mse_rd, lpips_rd = calcular_metricas(original, ruidosa, img_tensor, img_ruidosa)
    ssim_rr, psnr_rr, mse_rr, lpips_rr = calcular_metricas(original, restaurada, img_tensor, img_restaurada)

    # ---------------------------
    # 🔔 AVISO SI HAY PEORÍA EN LA RESTAURACIÓN
    # ---------------------------
    if ssim_rr < ssim_rd or psnr_rr < psnr_rd or mse_rr > mse_rd:
        st.error(
            "⚠️ La imagen restaurada presenta una peor calidad que la imagen degradada")

    # --- Cálculo de métricas sin referencia ---
    ruidosa_np = np.array(ruidosa)
    restaurada_np = np.array(restaurada)

    ent_ruidosa, sharp_ruidosa, cont_ruidosa, color_ruidosa = calculate_1image_metrics(ruidosa_np)
    ent_restaurada, sharp_restaurada, cont_restaurada, color_restaurada = calculate_1image_metrics(restaurada_np)

    # Construcción del DataFrame con las métricas
    df_metricas = pd.DataFrame({
        "Métrica": ["SSIM", "PSNR", "MSE", "LPIPS"],
        "Imagen degradada": [
            round(ssim_rd, 4), round(psnr_rd, 2), round(mse_rd, 6), round(lpips_rd, 4)        ],
        "Imagen restaurada": [
            round(ssim_rr, 4), round(psnr_rr, 2), round(mse_rr, 6), round(lpips_rr, 4)
        ]
    })

    st.markdown("### 📊 Métricas Con Referencia")
    st.dataframe(df_metricas.style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#b0b0b0')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#e0e0e0')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f5f5f5')]},
    ]))


    # ---------------------------
    # MÉTRICAS SIN REFERENCIA
    # ---------------------------
    def calculate_entropy(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return shannon_entropy(img)


    def calculate_sharpness(img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        mean_intensity = np.mean(img)
        normalized_sharpness = variance / (mean_intensity + 1e-5)
        return normalized_sharpness


    def calculate_contrast(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.std()


    def calculate_colorfulness(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        R, G, B = cv2.split(image_rgb)
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        rg_mean = np.mean(rg)
        yb_mean = np.mean(yb)
        rg_std = np.std(rg)
        yb_std = np.std(yb)
        colorfulness = np.sqrt(rg_mean ** 2 + yb_mean ** 2) + 0.3 * np.sqrt(rg_std ** 2 + yb_std ** 2)
        return colorfulness


    def calculate_1image_metrics(image):
        entropy = calculate_entropy(image)
        sharpness = calculate_sharpness(image)
        contrast = calculate_contrast(image)
        colorfulness = calculate_colorfulness(image)
        return round(entropy, 2), round(sharpness, 2), round(contrast, 2), round(colorfulness, 2)


    # Preparar imágenes para métricas sin referencia (redimensionar para evitar inconsistencias)
    original_np = np.array(original.resize((320, 320)))
    ruidosa_np = np.array(ruidosa.resize((320, 320)))
    restaurada_np = np.array(restaurada.resize((320, 320)))

    metrics_original = calculate_1image_metrics(original_np)
    metrics_ruidosa = calculate_1image_metrics(ruidosa_np)
    metrics_restaurada = calculate_1image_metrics(restaurada_np)

    st.markdown("### 📊 Métricas Sin Referencia")

    df_metricas_sin_ref = pd.DataFrame({
        "Métrica": ["Entropía", "Nitidez", "Contraste", "Colorfulness"],
        "Original": metrics_original,
        f"Degradada ({nombre_seleccionado})": metrics_ruidosa,
        "Restaurada": metrics_restaurada
    })

    st.dataframe(df_metricas_sin_ref.style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#b0b0b0')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#e0e0e0')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#f5f5f5')]},
    ]))

    # Pie de página simple
    st.markdown(
        """
        <hr>
        <p style='font-size:0.8em; text-align:center; color:gray;'>
            © 2025 Proyecto de mejora de imágenes. Todos los derechos reservados.
        </p>
        """,
        unsafe_allow_html=True
    )