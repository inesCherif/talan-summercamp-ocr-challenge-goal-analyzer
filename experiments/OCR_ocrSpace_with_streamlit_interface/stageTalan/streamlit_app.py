import io
from pathlib import Path

import streamlit as st
from ocr import ocr_space

st.set_page_config(page_title="OCR.space Dashboard", layout="wide")

st.title("OCR.space Dashboard")
st.markdown(
    "Chargez une image ou utilisez le fichier `image.jpg` existant, puis entrez votre clé API OCR.space pour tester l'extraction de texte."
)

api_key = st.sidebar.text_input(
    "OCR.space API key",
    placeholder="Entrez votre clé API OCR.space",
    type="password",
)

use_sample = st.sidebar.checkbox("Utiliser image.jpg du workspace", value=False)

uploaded_file = st.file_uploader(
    "Téléversez une image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "gif"],
)

image_source = None
image_preview = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_source = io.BytesIO(image_bytes)
    image_preview = image_bytes
elif use_sample:
    sample_path = Path("image.jpg")
    if sample_path.exists():
        image_source = sample_path
        image_preview = sample_path.read_bytes()
    else:
        st.sidebar.error("Le fichier image.jpg est introuvable dans le dossier courant.")

if image_source is not None:
    st.subheader("Aperçu de l'image")
    st.image(image_preview, use_column_width=True)

if st.button("Exécuter OCR"):
    if not api_key:
        st.error("Merci d'entrer une clé API OCR.space dans la barre latérale.")
    elif image_source is None:
        st.error("Merci de téléverser une image ou de sélectionner l'option `image.jpg`.")
    else:
        try:
            with st.spinner("Extraction du texte en cours..."):
                texte = ocr_space(image_source, api_key)

            st.success("OCR terminé")
            st.subheader("Texte extrait")
            st.text_area("Résultat OCR", value=texte, height=320)
        except Exception as exc:
            st.error(f"Erreur lors de l'appel OCR.space : {exc}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Exécuter le tableau de bord**\n\n`streamlit run streamlit_app.py`"
)
