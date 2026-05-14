# -*- coding: utf-8 -*-

from tensorflow.python.client import device_lib # per vedere dispositivi disponibili

from tensorflow.keras.utils import to_categorical # per convertire le etichette in one-hot encoding 5 -> [0,0,0,0,0,1,0,0,0,0]

from tensorflow.keras.models import Sequential # importa il tipo di modello lineare a strati
from tensorflow.keras.layers import Dense, Dropout, Flatten # importa layer neurali 
# Dense = layer completamente connesso, 
# Dropout = spegne neuroni casuali (non lo usiamo), 
# Flatten = appiattisce i dati (da matrice a vettore)

from tensorflow.keras.layers import Conv2D, MaxPooling2D # layer per CNN:
# Conv2D = convoluzione
# MaxPooling2D = riduzione dimensioni immagine

from tensorflow.keras.optimizers import SGD # Stochastic Gradient Descent, algoritmo di ottimizzazione per aggiornare i pesi del modello durante l'addestramento
from tensorflow.keras.models import load_model # per caricare un modello salvato in precedenza, in questo caso il nostro modello addestrato su MNIST

import cv2 # per visualizzare le immagini, non è necessario se si usa matplotlib, ma lo usiamo per convertire le immagini da BGR a RGB
import numpy as np # per manipolare array e matrici
import matplotlib.pyplot as plt # per visualizzare le immagini, grafici e risultati

import idx2numpy # per convertire i file .idx3-ubyte e .idx1-ubyte in array numpy, questi file contengono le immagini e le etichette del dataset MNIST

def imshow(title, image = None, size = 6): # funzione per visualizzare un'immagine con matplotlib, prende in input il titolo, l'immagine e la dimensione della figura
    w, h = image.shape[0], image.shape[1] # preleva la larghezza e l'altezza dell'immagine
    aspect_ratio = w/h # calcola la proporzione dell'immagine per mantenere le proporzioni corrette
    plt.figure(figsize=(size*aspect_ratio, size)) # imposta la dimensione della figura in base alla proporzione dell'immagine
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # converte l'immagine da BGR a RGB e la visualizza con matplotlib
    plt.title(title) # imposta il titolo della figura
    plt.show() # mostra la figura

def main():
    
    # percorso immagini del trainig
    train_img_path = 'dati/SAllenamento/train-images.idx3-ubyte'
    # percorso etichette del training
    train_lbl_path = 'dati/SAllenamento/train-labels.idx1-ubyte'

    # percorso immagini del test
    test_img_path = 'dati/STest/t10k-images.idx3-ubyte'
    # percorso etichette del test
    test_lbl_path = 'dati/STest/t10k-labels.idx1-ubyte'

    # Carica i file locali
    # immagini trainig
    x_train = idx2numpy.convert_from_file(train_img_path)
    # etichette training
    y_train = idx2numpy.convert_from_file(train_lbl_path)

    # immagini test
    x_test = idx2numpy.convert_from_file(test_img_path)
    # etichette test
    y_test = idx2numpy.convert_from_file(test_lbl_path)

    #posizione del modello salvatoF
    model_path = "export/mnist_ai_1_0.h5"


    print(device_lib.list_local_devices()) # stampa i dispositivi disponibili, come CPU e GPU, per verificare se TensorFlow può utilizzare la GPU per l'addestramento del modello

    
    # mostra la forma o le dimensioni degli array x_train, y_train, x_test e y_test, che contengono rispettivamente le immagini di addestramento, le etichette di addestramento, le immagini di test e le etichette di test del dataset MNIST
    print("Initial shape or dimensions of x_train : " + str(x_train.shape)) # es: (60000, 28, 28) significa che ci sono 60000 immagini di addestramento, ognuna con dimensioni 28x28 pixel


    # mostra 4 immagini casuali dal dataset di addestramento per visualizzare i dati
    for i in range(0,4):
        random_num = np.random.randint(0, len(x_train)) # genera un numero casuale tra 0 e la lunghezza di x_train (60000)
        img = x_train[random_num] # prende l'immagine
        imshow(f"sample{i}", img, size = 2) # mostra l'immagine

    #----------------------
    # PRE-PROCESSING
    #----------------------
    
    # prende le dimensioni delle immagini dal primo elemento di x_train, che è un array 2D di dimensioni 28x28 pixel, e le assegna a img_rows e img_cols
    img_rows = x_train[0].shape[0]
    img_cols = x_train[0].shape[1]

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) # aggiunge canale colore -> (60000, 28, 28) -> (60000, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) # aggiunge canale colore -> (10000, 28, 28) -> (10000, 28, 28, 1)

    input_shape = (img_rows, img_cols, 1) # definisce la forma di input per il modello, che è (28, 28, 1) per indicare che le immagini hanno dimensioni 28x28 pixel e 1 canale di colore

    x_train = x_train.astype('float32') # converte i dati di x_train in tipo float32
    x_test = x_test.astype('float32') # converte i dati di x_test in tipo float32

    #----------------------
    # NORMALIZZAZIONE
    #----------------------

    # fa diventare i pixel da 0-255 a 0-1, dividendo ogni pixel per 255.0, in modo che i valori dei pixel siano compresi tra 0 e 1, il che aiuta a migliorare le prestazioni del modello durante l'addestramento
    x_train /= 255.0
    x_test /= 255.0

    # mostra nuova forma o dimensioni degli array x_train e x_test dopo la ristrutturazione e la normalizzazione, che ora dovrebbero essere (60000, 28, 28, 1) per x_train e (10000, 28, 28, 1) per x_test
    print("x_train shape : ", x_train.shape)
    print("x_test shape : ", x_test.shape)

    #----------------------
    # ONE HOT ENCODING
    #----------------------

    # trasforma le etichette di y_train e y_test in formato one-hot encoding,
    # che è una rappresentazione binaria in cui ogni classe è rappresentata da un vettore con un solo elemento a 1 (indica la classe corretta) e tutti gli altri elementi a 0 (indicano le classi errate). Ad esempio, se la classe è 5, il vettore sarà [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # mostra nuova forma o dimensioni degli array y_train e y_test dopo la conversione in one-hot encoding, che ora dovrebbero essere (60000, 10) per y_train e (10000, 10) per y_test, dove 10 è il numero di classi (cifre da 0 a 9)
    print('y_train : ', y_train.shape)
    print('y_test : ', y_test.shape)

    # preleva il numero di classi dal secondo elemento di y_test, che è 10, e lo assegna a num_classes, che rappresenta il numero di classi nel dataset MNIST (cifre da 0 a 9)
    num_classes = y_test.shape[1]

    y_train[0] # mostra la prima etichetta di y_train dopo la conversione in one-hot encoding, che dovrebbe essere un vettore di 10 elementi con un solo elemento a 1 (indica la classe corretta) e tutti gli altri elementi a 0 (indicano le classi errate)

  
    #----------------------
    # CREAZIONE DEL MODELLO
    #----------------------

    model = Sequential() # crea modello sequenziale 'vuoto', che è un modello lineare a strati in cui ogni strato ha esattamente un input e un output, e i dati fluiscono attraverso il modello in ordine sequenziale
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape)) # Primo layer convoluzionale 
    # con 32 filtri, kernel di dimensione 3x3, funzione di attivazione ReLU e forma di input definita da input_shape (28, 28, 1)

    model.add(Conv2D(64, (3, 3), activation='relu')) # secondo layer convoluzionale con 64 filtri, kernel di dimensione 3x3 e funzione di attivazione ReLU

    model.add(MaxPooling2D(pool_size=(2, 2))) # riduce dmensioni dell'immagine di input di 2x2, prendendo il valore massimo in ogni finestra di 2x2 pixel, il che aiuta a ridurre la complessità del modello e a prevenire l'overfitting
    
    model.add(Flatten()) # appiattisce i dati in un vettore 1D, trasformando l'output del layer precedente (che è una matrice 2D) in un vettore 1D che può essere utilizzato come input per i layer densi successivi

    model.add(Dense(128, activation = 'relu')) # layer denso completamente connesso con 128 neuroni e funzione di attivazione ReLU, che aiuta a catturare le relazioni non lineari nei dati

    model.add(Dense(num_classes, activation = 'softmax')) # produce probabilità per ciascuna delle 10 classi (cifre da 0 a 9) utilizzando la funzione di attivazione softmax, che normalizza le uscite in modo che la somma delle probabilità sia uguale a 1

    # compila il modello
    model.compile(loss = 'categorical_crossentropy', # funzione di perdita utilizzata per problemi di classificazione multi-classe, che misura la differenza tra le probabilità previste dal modello e le etichette reali in formato one-hot encoding
                optimizer = SGD(0.001), # algoritmo di ottimizzazione Stochastic Gradient Descent con un tasso di apprendimento di 0.001, che viene utilizzato per aggiornare i pesi del modello durante l'addestramento
                metrics = ['accuracy']) # metrica utilizzata per valutare le prestazioni del modello durante l'addestramento e il test, in questo caso l'accuratezza, che misura la percentuale di previsioni corrette rispetto al totale delle previsioni
    
    print(model.summary()) # stampa un riepilogo del modello, mostra struttura rete neurale, numero di parametri e forma di output di ogni layer

    
    #----------------------
    # ADDESTRAMENTO
    #----------------------

    batch_size = 128 # 128 immagini vengono elaborate insieme prima di aggiornare i pesi del modello, il che aiuta a migliorare l'efficienza dell'addestramento e a stabilizzare l'aggiornamento dei pesi
    epochs = 25 # numero di volte in cui l'intero dataset di addestramento viene passato attraverso il modello durante l'addestramento, più epoche possono portare a un modello più accurato, ma anche a un rischio maggiore di overfitting se il numero di epoche è troppo alto

    # avvia l'addestramento
    history = model.fit(x_train, # dati di addestramento (immagini)
                        y_train, # etichette di addestramento (in formato one-hot encoding)
                        batch_size=batch_size, # dimensione del batch
                        epochs=epochs, # numero di epoche
                        verbose=1, # impostazione per visualizzare l'output dell'addestramento, 1 mostra una barra di avanzamento per ogni epoca
                        validation_data = (x_test, y_test)) # dati di validazione (immagini e etichette di test) utilizzati per valutare le prestazioni del modello dopo ogni epoca di addestramento, il che aiuta a monitorare l'overfitting e a scegliere il numero ottimale di epoche


    #----------------------
    # VALUTAZIONE
    #----------------------
    
    score = model.evaluate(x_test, y_test, verbose=0) # calcola accuratezza finale
    print('Test loss : ', score[0]) # stampa la perdita del modello sui dati di test, che misura la differenza tra le probabilità previste dal modello e le etichette reali in formato one-hot encoding, un valore più basso indica un modello migliore
    print('Test accuracy : ', score[1]) # stampa l'accuratezza del modello sui dati di test, che misura la percentuale di previsioni corrette rispetto al totale delle previsioni, un valore più alto indica un modello migliore

    #  mostra lo storico dell'addestramento
    history_dict = history.history
    history_dict

    x_test.shape

    # salvataggio modello
    model.save(model_path) # salva rete neurale
    print("model saved")

    classifier = load_model(model_path) # carica modello salvato, in questo caso il nostro modello addestrato su MNIST, che può essere utilizzato per fare previsioni su nuove immagini di test

    # ricarica modello e cerca di predirre le cifre
    print("predicting classes for all 10000 test images")
    pred = np.argmax(classifier.predict(x_test), axis=-1) # argmax prende la probabilità più alta, es: se il modello prevede [0.1, 0.2, 0.05, 0.05, 0.1, 0.3, 0.05, 0.05, 0.05, 0.1] per un'immagine di test, argmax restituirà 5, che è la classe con la probabilità più alta (cifra 5)
    print("completed")

    # mostra le predizioni
    print(pred) 
    print(type(pred)) 
    print(len(pred))

    # prende un'immagine di test specifica (in questo caso, la 30esima immagine, che ha indice 29) e la visualizza,
    # mostrando anche la forma dell'immagine prima e dopo la ristrutturazione per il modello
    input_img = x_test[29]
    imshow('img', input_img)
    print(input_img.shape) # prima della ristrutturazione, dovrebbe essere (28, 28, 1)

    input_img = input_img.reshape(1, 28, 28, 1)
    print(input_img.shape) # dopo la ristrutturazione, dovrebbe essere (1, 28, 28, 1), che è la forma di input richiesta dal modello per fare previsioni su una singola immagine

    pred = np.argmax(classifier.predict(input_img), axis=-1) # predice cifra

    # mostra la predizione per l'immagine di test specifica, che dovrebbe essere un numero intero tra 0 e 9, rappresentante la cifra prevista dal modello per quell'immagine
    print(pred)
    print(type(pred))
    print(len(pred))

if __name__ == "__main__":
    main()