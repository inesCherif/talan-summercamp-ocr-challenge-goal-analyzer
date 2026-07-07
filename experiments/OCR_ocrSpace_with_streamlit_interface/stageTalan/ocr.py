import io
import requests
from PIL import Image


def compresser_image(image_source, taille_max_mo=1.4):
    """Redimensionne/compresse l'image si elle dépasse la taille max."""
    if hasattr(image_source, "seek"):
        image_source.seek(0)
        img = Image.open(image_source)
    else:
        img = Image.open(image_source)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 95
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)

    while buffer.tell() > taille_max_mo * 1024 * 1024 and quality > 10:
        quality -= 10
        buffer.seek(0)
        buffer.truncate()
        img.save(buffer, format="JPEG", quality=quality)

    buffer.seek(0)
    return buffer


def ocr_space(image_source, api_key):
    """Extrait le texte d'une image locale ou d'un flux de bytes via l'API OCR.space."""
    image_buffer = compresser_image(image_source)
    response = requests.post(
        "https://api.ocr.space/parse/image",
        files={"file": ("image.jpg", image_buffer, "image/jpeg")},
        data={"apikey": api_key, "OCREngine": 2},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    if "ParsedResults" in result and result["ParsedResults"]:
        return result["ParsedResults"][0].get("ParsedText", "")

    raise ValueError(f"Erreur OCR.space: {result}")


if __name__ == "__main__":
    api_key = "K84132262688957"  # Remplacez par votre clé OCR.space.
    texte = ocr_space("image.jpg", api_key)
    print(texte)
