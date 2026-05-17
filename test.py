
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw  # per gestire l'immagine del disegno
from tensorflow.keras.models import load_model

# ----------------------------
# impostazioni della finestra
# ----------------------------

CANVAS_SIZE = 280       # dimensione della tela in pixel (10x quella di MNIST che è 28x28)
PENNELLO = 10           # spessore del pennello quando disegni
SFONDO = "black"        # colore di sfondo (come in MNIST, sfondo nero)
COLORE_DISEGNO = "white"  # colore del tratto (come in MNIST, numero bianco)

# percorso del modello salvato - cambialo se il tuo è in un'altra cartella
MODEL_PATH = "export/mnist_ai_1_0.h5"


class AppDisegno:
    """classe principale dell'app, gestisce tutto: finestra, canvas, predizione"""

    def __init__(self, root):
        self.root = root
        self.root.title("disegna un numero - test")

        # carica il modello all'avvio
        print("carico il modello...")
        self.model = load_model(MODEL_PATH)
        print("modello caricato!")

        # immagine PIL parallela al canvas: ci servirà per mandare i pixel al modello
        # lavoriamo in scala di grigi (mode 'L')
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._costruisci_interfaccia()

    def _costruisci_interfaccia(self):
        """crea tutti i widget della finestra"""

        # titolo in cima
        label_titolo = tk.Label( # label è un widget per mostrare testo, immagini o entrambi
            self.root, # il primo argomento è sempre il "padre" del widget, in questo caso la finestra principale
            text="disegna un numero (0-9)",
            font=("Arial", 16, "bold")
        )
        label_titolo.pack(pady=8) # spazio sopra e sotto il titolo

        # la tela dove disegni
        self.canvas = tk.Canvas( # widget per disegnare, mostrare grafica, ecc.
            self.root, # padre del canvas è la finestra principale
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=SFONDO,
            cursor="crosshair"  # cursore a croce, più comodo per disegnare
        )
        self.canvas.pack(padx=10) # spazio intorno al canvas

        # collega gli eventi del mouse alle funzioni di disegno
        self.canvas.bind("<B1-Motion>", self._disegna)       # tasto sinistro tenuto premuto
        self.canvas.bind("<ButtonRelease-1>", self._fine_tratto)  # quando rilasci

        # riga dei pulsanti
        frame_bottoni = tk.Frame(self.root) # contenitore per i pulsanti, così possiamo gestire meglio il layout
        frame_bottoni.pack(pady=10) # spazio sopra e sotto i pulsanti

        btn_predici = tk.Button(
            frame_bottoni,
            text="che numero è? 🔍",
            font=("Arial", 13),
            command=self._predici,
            bg="#4CAF50",
            fg="white",
            padx=10
        )
        btn_predici.pack(side=tk.LEFT, padx=8) # spazio tra i pulsanti

        btn_cancella = tk.Button(
            frame_bottoni,
            text="cancella ✖",
            font=("Arial", 13),
            command=self._cancella,
            bg="#f44336",
            fg="white",
            padx=10
        )
        btn_cancella.pack(side=tk.LEFT, padx=8) # spazio tra i pulsanti

        # label che mostra il risultato
        self.label_risultato = tk.Label(
            self.root,
            text="disegna qualcosa e poi clicca 'che numero è?'",
            font=("Arial", 14),
            fg="#333333"
        )
        self.label_risultato.pack(pady=8)

        # variabile per ricordare l'ultima posizione del mouse (serve per il tratto continuo)
        self.ultima_x = None
        self.ultima_y = None

    def _disegna(self, event):
        """chiamata ogni volta che muovi il mouse con il tasto premuto"""

        x = event.x
        y = event.y

        if self.ultima_x is not None:
            # disegna sul canvas visivo
            self.canvas.create_oval(
                x - PENNELLO, y - PENNELLO,
                x + PENNELLO, y + PENNELLO,
                fill=COLORE_DISEGNO,
                outline=COLORE_DISEGNO
            )
            # disegna anche sull'immagine PIL (quella che mandiamo al modello)
            self.pil_draw.ellipse(
                [x - PENNELLO, y - PENNELLO, x + PENNELLO, y + PENNELLO],
                fill=255  # bianco in scala di grigi
            )

        self.ultima_x = x
        self.ultima_y = y

    def _fine_tratto(self, event):
        """quando rilasci il mouse, azzera la posizione precedente"""
        self.ultima_x = None
        self.ultima_y = None

    def _cancella(self):
        """pulisce il canvas e l'immagine PIL"""
        self.canvas.delete("all")
        # ricrea l'immagine PIL tutta nera (vuota)
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.label_risultato.config(text="disegna qualcosa e poi clicca 'che numero è?'")

    def _predici(self):
        """prende il disegno, lo ridimensiona a 28x28 e chiede al modello cosa vede"""

        # ridimensiona l'immagine da 280x280 a 28x28 (come le immagini MNIST)
        img_piccola = self.pil_image.resize((28, 28), Image.LANCZOS)

        # converti in array numpy e normalizza i valori tra 0 e 1
        img_array = np.array(img_piccola).astype("float32") / 255.0

        # controlla che ci sia qualcosa disegnato (se è tutto nero, i pixel sono tutti 0)
        if img_array.max() == 0:
            messagebox.showwarning("ops!", "non hai disegnato niente!\ndisegna un numero prima.")
            return

        # aggiunge le dimensioni che il modello si aspetta: (1, 28, 28, 1)
        # 1 = una sola immagine, 28x28 = dimensioni, 1 = canale colore (grigio)
        img_array = img_array.reshape(1, 28, 28, 1)

        # chiede al modello le probabilità per ogni cifra (da 0 a 9)
        probabilita = self.model.predict(img_array, verbose=0)[0]

        # prende la cifra con la probabilità più alta
        cifra_prevista = np.argmax(probabilita)
        confidenza = probabilita[cifra_prevista] * 100  # in percentuale

        # aggiorna il testo del risultato
        self.label_risultato.config(
            text=f"probabilmente è un  {cifra_prevista}  ({confidenza:.1f}% sicuro)",
            font=("Arial", 16, "bold"),
            fg="#1565C0"
        )

        # stampa anche le probabilità complete nel terminale, utile per debug
        print("\nprobabilità per ogni cifra:")
        for i, p in enumerate(probabilita):
            barra = "█" * int(p * 30)  # piccola barra visiva nel terminale
            print(f"  {i}: {barra} {p*100:.1f}%")


# ----------------------------
# punto di partenza del programma
# ----------------------------

if __name__ == "__main__":
    root = tk.Tk() # crea la finestra principale
    app = AppDisegno(root) # crea l'app passando la finestra, questo costruisce tutta l'interfaccia
    root.mainloop()  # avvia il loop principale della finestra