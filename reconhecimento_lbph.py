import cv2

webcam = cv2.VideoCapture(0)
classificadorRosto = cv2.CascadeClassifier('venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPH.yml")
largura, altura = 220, 220

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

nomes = ['Sergio', 'Gabriel']

while (True):
    conectado, imagem = webcam.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorRosto.detectMultiScale(imagemCinza, minSize=(200, 200))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        # cv2.imshow("Face", imagemFace)
        # escreve em baixo do retangulo da imagem
        try:
            cv2.putText(imagem, nomes[id - 1] + " - " + str(confianca), (x, y + a + 30), font, 2, (0, 0, 255))
        except:
            print("Error")
    cv2.imshow("Face", imagem)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
