"""Einzige Quelle für Detektionsklassen und Farbpalette.

Das Modell erkennt nur noch, WAS ein Griff ist (hold/volume), nicht welche
Farbe er hat. Die Farbe wird nachgelagert aus den Bildpixeln bestimmt
(color_extraction.py) und auf die Palette hier gemappt. Neue Grifffarbe im
Gym = eine Zeile hier, kein Retraining.

Die Reihenfolge der Palette ist abwärtskompatibel zur alten App: Die ersten
zehn Indizes entsprechen der früheren Klassenliste, damit alte App-Versionen
über det["class"] weiterhin sinnvolle Farben anzeigen.
"""

# Klassen, die das YOLO-Modell lernt (data.yaml muss dieselbe Reihenfolge haben)
DETECTION_CLASSES = ["hold", "volume"]

# Benannte Grifffarben. hex dient der App-Anzeige UND als Referenzfarbe für
# die Klassifikation (Lab-Anker werden daraus berechnet, siehe
# color_extraction.anchors()).
COLOR_PALETTE = [
    {"name": "black",     "hex": "#212121"},
    {"name": "blue",      "hex": "#1E63D0"},
    {"name": "grey",      "hex": "#9E9E9E"},
    {"name": "orange",    "hex": "#F57C00"},
    {"name": "purple",    "hex": "#7B2FBE"},
    {"name": "red",       "hex": "#D32F2F"},
    {"name": "turquoise", "hex": "#00BCD4"},
    {"name": "white",     "hex": "#F0F0F0"},
    {"name": "wood",      "hex": "#A0764F"},
    {"name": "yellow",    "hex": "#FBC02D"},
    {"name": "green",     "hex": "#388E3C"},
    {"name": "pink",      "hex": "#EC407A"},
]

COLOR_INDEX = {c["name"]: i for i, c in enumerate(COLOR_PALETTE)}

# Farben ohne Farbstich: werden über Helligkeit statt Buntheit unterschieden
ACHROMATIC = {"black", "grey", "white"}
