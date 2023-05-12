import cv2

webcam = cv2.VideoCapture(0)

while True:

    conectado, frame = webcam.read()

    classificadoresRosto = cv2.CascadeClassifier("venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostosEncontrados = classificadoresRosto.detectMultiScale(imagemCinza,
                                                              scaleFactor=1.01,
                                                              minNeighbors=20,
                                                              minSize=(100, 100),
                                                              maxSize=(500, 500)
                                                              )

    for (x, y, largura, altura) in rostosEncontrados:
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), (0, 0, 255), 4)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()