import cv2
import numpy as np
import os
import argparse
from datetime import datetime


# funkcja używana do detekcji obiektów na obrazku/klatce z wideo
def object_detection(image_in):
    # pobranie rozmiaru obrazka
    (image_height, image_width) = image_in.shape[:2]
    # tworzymy blob
    blob = cv2.dnn.blobFromImage(image_in, 1 / 255., (416, 416), swapRB=True, crop=False)
    # i przepusczamy go przez model
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        # pętla dla kazdej detekcji
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # odrzucenie detekcji poniżej poziomu ufności
            if confidence > args["confidence"]:
                box = detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                (x_center, y_center, width, height) = box.astype('int')

                x = int(x_center - (width / 2))
                y = int(y_center - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # kolory i rysowanie prostokątów z tekstem - poziom ufności
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f'{LABELS[class_ids[i]]}: {confidences[i]:.4f}'
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="ścieżka do pliku")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimalny poziom ufności ")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold ")
ap.add_argument("-s", "--save", type=bool, default=False, help="czy zapisać wynik wykrywania")
args = vars(ap.parse_args())

# jeżeli ma być zapisany wynik to sprawdzam czy istnieje folder output - jak nie to jest tworzony
# oraz tworze ściezke do zapisu pliku. Do nazwy pliku doklejam date_godzine
output_file = ""
writer = ""

# jeżeli wynik ma być zapisany to sprawdzam czy jest folder output - jak nie to go tworzę
if args["save"]:
    if not os.path.exists('output'):
        os.mkdir('output')

    # i ustawiam nazwe pliku docelowego z datą_godzina
    dt = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = os.path.join('output', os.path.basename(args["input"]).split('.')[0] + '_' + dt)

# Wczytanie etykiet klasy COCO, na których trenowano model YOLOv3
print('[INFO] Wczytanie etykiet...')
labels_path = os.path.join('model', 'coco.names')
LABELS = open(labels_path).read().strip().split('\n')

# Zainicjowanie listy kolorów reprezentującą każdą możliwą etykietę klasy, użyte jako kolor ramek/opisu
np.random.seed(5)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

# Ścieżki do wag oraz konfiguracji modelu
weights_path = os.path.join('model', 'yolov3.weights')
config_path = os.path.join('model', 'yolov3.cfg')

# załadowanie wykrywanie obiektów YOLO wytrenowane na zestawie danych COCO (80 klas)
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# rozszerzenie pliku
ext = os.path.splitext(args["input"])[-1].lower()
if ext in ('.jpg', '.jpeg', '.png'):
    image = cv2.imread(args['input'])
    # wyświetlenie wczytanego obrazka
    # cv2.imshow('oryginalny', image)

    # detekcja obiektów na wczytanym obrzku
    object_detection(image)

    # wyświetlenie przetworzonego obrazka
    cv2.imshow('przetworzony obrazek', image)

    # zapis gdy trzeba
    if args["save"]:
        cv2.imwrite(output_file + ".jpg", image)

    # klawisz escape zamyka okno z obrazkiem
    while True:
        if cv2.waitKey(0) == 27:
            break


else:
    # czyli mamy wideo
    cnt = 0

    # wczytanie pliku oraz podstawowych parametrów
    vs = cv2.VideoCapture(args['input'])
    writer_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vs.get(cv2.CAP_PROP_FPS)

    # jeżeli plik ma być zapisany:, zainicjowanie zapisywania wideo
    if args["save"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_file + ".avi", fourcc, fps, (writer_width, writer_height), True)

    while True:
        cnt += 1
        print("Klatka: ", cnt)
        # wczytaj następną klatkę z pliku wideo
        (grabbed, image) = vs.read()
        # jeżeli nie wczytana to zakończ
        if not grabbed:
            break
        # detekcja obiektów dla klatki
        object_detection(image)

        # wyświetlanie przetworzonego wideo
        cv2.imshow("output", cv2.resize(image, (writer_width, writer_height)))

        # sprawdzenie czy zapisujemy wynikowy plik
        if args["save"]:
            writer.write(cv2.resize(image, (writer_width, writer_height)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # usunięcie okien
    cv2.destroyAllWindows()
    vs.release()

    # sprawdzenie gdy ma być zapisany czy robimy kompresję
    if args["save"]:
        writer.release()
        # kompresja
        while True:
            compress = input("Czy skompresować wynikowy plik wideo (T/N) ")
            if compress.lower() == "t" or compress.lower() == "n":
                break

        if compress.lower() == 't':
            os.system(f"ffmpeg -i {output_file + '.avi'} -vcodec libx264 {output_file + '.mp4'}")

    print("[INFO] Gotowe")
