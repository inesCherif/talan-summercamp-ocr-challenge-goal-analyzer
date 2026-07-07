#!/usr/bin/env python3
"""
ocr.py — OCR manuscrit en local avec TrOCR (microsoft/trocr-large-handwritten).

CONTEXTE :
L'API gratuite d'inférence de Hugging Face ("hf-inference") ne propose plus
aucun modèle d'OCR ou de vision (VLM) sur son offre gratuite. On exécute donc
le modèle TrOCR directement sur la machine, via la librairie `transformers`.
Le modèle est téléchargé une seule fois (environ 1,3 Go) puis mis en cache
localement par Hugging Face ; les exécutions suivantes sont hors-ligne.

Usage :
    python ocr.py chemin/vers/image.jpg

Prérequis (à installer une seule fois) :
    pip install transformers torch pillow

Aucune clé API ni variable d'environnement n'est nécessaire.
"""

import os
import sys

# --- Dépendances lourdes : importées après la vérification des arguments,
# pour éviter d'attendre le chargement de torch/transformers si l'utilisateur
# a juste tapé une commande incorrecte.


def verifier_arguments():
    """Vérifie que l'utilisateur a bien fourni un chemin d'image en argument."""
    if len(sys.argv) != 2:
        print("Usage : python ocr.py chemin/vers/image.jpg", file=sys.stderr)
        sys.exit(1)

    chemin_image = sys.argv[1]
    if not os.path.isfile(chemin_image):
        print(f"Erreur : le fichier '{chemin_image}' est introuvable.", file=sys.stderr)
        sys.exit(1)

    return chemin_image


def charger_modele():
    """
    Charge le processeur (tokenizer + préparation d'image) et le modèle
    TrOCR depuis Hugging Face. Le premier lancement télécharge le modèle
    (~1,3 Go) et le met en cache ; les lancements suivants sont instantanés.
    """
    print("Chargement du modèle TrOCR (peut prendre un moment au premier lancement)...")

    # Import ici pour ne pas ralentir le démarrage du script en cas d'erreur d'usage.
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    nom_modele = "microsoft/trocr-large-handwritten"
    processeur = TrOCRProcessor.from_pretrained(nom_modele)
    modele = VisionEncoderDecoderModel.from_pretrained(nom_modele)

    return processeur, modele


def extraire_texte(chemin_image: str, processeur, modele) -> str:
    """
    Effectue l'OCR sur l'image donnée et renvoie le texte reconnu.
    """
    from PIL import Image

    # TrOCR attend une image en RGB (donc on convertit, même si l'image
    # d'origine est en niveaux de gris ou en RGBA).
    image = Image.open(chemin_image).convert("RGB")

    # Prépare l'image sous la forme attendue par le modèle (tenseur de pixels).
    valeurs_pixels = processeur(images=image, return_tensors="pt").pixel_values

    # Génère la séquence de tokens correspondant au texte reconnu.
    ids_generes = modele.generate(valeurs_pixels)

    # Décode les tokens en texte lisible.
    texte = processeur.batch_decode(ids_generes, skip_special_tokens=True)[0]

    return texte


def main():
    chemin_image = verifier_arguments()

    try:
        processeur, modele = charger_modele()
        texte = extraire_texte(chemin_image, processeur, modele)
    except Exception as e:
        print(f"Erreur pendant l'OCR : {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Texte extrait ===")
    print(texte.strip())
    print("======================\n")


if __name__ == "__main__":
    main()