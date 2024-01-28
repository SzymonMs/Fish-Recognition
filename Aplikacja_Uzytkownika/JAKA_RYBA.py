import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtCore import  QSize

import datetime
import onnxruntime
import numpy as np
from PIL import Image

x = ""
state = False #Blokowanie przycisku JAKA RYBA przed wyborem zdjęcia
session = onnxruntime.InferenceSession(r"files\best.onnx")

'''
Funkcja do odczytywania pliku txt w formacie:
a,b,c,...
'''
def readFile(nazwa_pliku):
    tabela = []
    try:
        with open(nazwa_pliku, 'r') as plik:
            for linia in plik:
                slowa = [slowo.strip() for slowo in linia.split(',')]
                tabela.append(slowa)
    except FileNotFoundError:
        print(f"Plik o nazwie {nazwa_pliku} nie istnieje.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

    return tabela

'''
Funkcja do wczytania obrazu
'''
def preprocess_image(image_path):
    # Wczytaj obraz i dostosuj go do oczekiwanego formatu przez model
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))  # Dostosuj rozmiar obrazu do wymagań modelu
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalizuj wartości pikseli
    image_array = np.transpose(image_array, (2, 0, 1))  # Zamień kanały kolorów (HWC na CHW)
    image_array = np.expand_dims(image_array, axis=0)  # Dodaj wymiar wsadowy (batch dimension)
    return image_array

class DialogWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('JAKA RYBA')
        self.setFixedSize(QSize(700, 500))
        self.setWindowIcon(QIcon('files\logo.png'))

        self.layout = QGridLayout()

        #OBRAZEK
        self.label_image = QLabel(self)
        self.layout.addWidget(self.label_image,0,0)
        #WYBÓR ZDJĘCIA
        self.button = QPushButton('WYBIERZ ZDJĘCIE', self)
        self.button.clicked.connect(self.show_file_dialog)
        self.layout.addWidget(self.button,1,0)
        #PREDYKCJA
        self.button2 = QPushButton('JAKA RYBA', self)
        self.button2.clicked.connect(self.show_prediction_result)
        self.layout.addWidget(self.button2,2,0)
        #NAZWA RYBY, DATY
        self.text_nazwaRyba = QLabel(self)
        self.layout.addWidget(self.text_nazwaRyba,0,1) 
        self.text_nazwaRyba.setText("Predykcja dla pierwszego zdjęcia\nmoże potrwać kilka sekund.\nNie wyłączaj aplikacji!")

        self.setLayout(self.layout)

    '''
    Funkcja do załadowania zdjęcia
    '''
    def show_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz obrazek", "", "Obrazy (*.png *.jpg *.bmp *.gif);;Wszystkie pliki (*)", options=options)

        if file_name:
            global x
            global state
            x = file_name
            state = True
            self.show_image(file_name)
            self.text_nazwaRyba.setText("")
        
    '''
    Funkcja do predykcji ryby na podstawie zdjęcia
    '''
    def show_prediction_result(self):
            global x
            if state:
                date = datetime.datetime.now()
                tab = readFile('files\ochrona.txt')
                labels = []
                for n in tab:
                    labels.append(n[0])
                
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                prediction = session.run([output_name], {input_name: preprocess_image(x)})
                # Wypisz wynik
                index = np.argmax(prediction)
                self.text_nazwaRyba.setText("Predykcja: "+tab[index][0]+"\n\n"+"Dzisiejsza data: "+str(date.day)+"/"+str(date.month)+"\n\nOkres ochronny: "+tab[index][2]+"\n\nWymiar ochronny: "+tab[index][1])
            else:
                pass
    '''
    Funkcja do pokazania wybranego zdjęcia
    '''
    def show_image(self, file_path):
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(400,400)
        self.label_image.setPixmap(pixmap)
        self.label_image.setScaledContents(False)

def main():
    app = QApplication(sys.argv)
    window = DialogWindow()
    window.show()
    sys.exit(app.exec_())

main()
