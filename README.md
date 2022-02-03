# Wykrywanie obiektów na zdjeciach oraz wideo z YOLO v3
<p>Wykrywanie obiektów na zdjęciu i w filmie przy użyciu algorytmu YOLO V3. W obu przypadkach podejście jest identyczne. Traktujemy każdą klatkę filmu jako obraz i wykrywamy obiekty na tej klatce używająć YOLO. Następnie rysujemy ramki, etykiety i iterujemy przez wszystkie klatki w danym filmie. Zjęcie oraz obraz wideo wraz z wykrytymi obiektami jest prezentowany odbiorcy. Za pomoca parametrów podanych przy uruchomieniu programu możemy zmieniać poziom ufności i próg nms, aby zobaczyć, jak zmieniają się wyniki wykrywania przez algorytm. W zależności od wybranego parametru może być on zapisany w folderze output. W projekcie wykorzystywane są wytrenowany jest wytrenowany model z repozytorium Darknet.</p>

## Pobranie niezbędnych bibliotek oraz wytrenowanego modelu
Instalacja niezbędnych bibliotek:</br>
  pip install cv2</br>
  pip install numpy as np</br>
  pip install os</br>
  pip install argparse</br>
  pip install datetime</br>

Pobranie wytrenowanego modelu z repozytorium darknet:</br>
  cd model</br>
  wget https://pjreddie.com/media/files/yolov3.weights</br>
  wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg</br>
  wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names</br>
  
## Uruchomienie programu
<p>Program uruchamiamy przez wywołanie programu wraz z przekazaniem ścieżki do pliku na którym chcemy rozpoznać obiekty (obowiązkowo) oraz parametry które chcemy wywołać - nieobowiązkowo</br>
</p>
<p>Program obsługuję następujące parametry: </br>
-i ścieżka do pliku</br>
-c minimalny poziom ufności żeby obiekt był wykryty, domyślnie 0.5</br>
-t threshold dla Non-Max Suppression, domyślnie 0.3</br>
-s czy zapisać plik z wynikami detekcji, domyślnie False</br>
</p>
### Detekcja na obrazie
<p> Obsługiwane pliki z rozszerzeniamie jpg,png,jpeg. Przykładowe wywołanie:</br>
<code>python yolo.py -i test_files/test_1.jpg</code></br>
W celu zamknięcia programu należy nacisnąć klawisz escape</p>

### Detekcja na pliku wideo
<p>Przykładowe wywołanie:
<code>python yolo.py -i test_files/traffic_1.mp4 -s TRUE</code></br>
W celu przerwania detekcji należy nacisnąć klawisz q lub poczekać do końca detekcji. Jeżeli wybrany został parametr -s TRUE po zakończeniu detekcji program zada pytanie czy chcemy skompresować wygenerowany plik. Wybranie T powoduje zapisanie pliku z rozszerzeniem MP4.</p>


