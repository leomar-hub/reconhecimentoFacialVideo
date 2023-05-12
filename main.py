import cv2
from funcTreinamentoIA import funcTreinamentoIA

webcam = cv2.VideoCapture(0)
classificadoresRosto = cv2.CascadeClassifier("venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")

numeroAmostra = 25
amostra = 1
larguraImg, alturaImg = 220, 220
id = input("Digite o id da pessoa: ")

while True:
    conectado, imagem = webcam.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    rostosEncontrados = classificadoresRosto.detectMultiScale(imagemCinza)

    for (x, y, largura, altura) in rostosEncontrados:
        cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (0, 0, 255), 4)
        if (cv2.waitKey(1) == ord('f')):
            imageFace = cv2.resize(
                imagemCinza[y:y + altura, x:x + largura], (larguraImg, alturaImg)
            )
            cv2.imwrite("fotos/pessoa." + str(id)
                        + "."+str(amostra)+".jpg", imageFace)
            print("Foto: "+str(amostra)+",capturada com sucesso!")
            amostra += 1

    cv2.imshow('webcam', imagem)

    if(amostra >= numeroAmostra +1):
        break

    if cv2.waitKey(1) == ord('q'):
        break

print("Captura finalizada")
webcam.release()
cv2.destroyAllWindows()

funcTreinamentoIA()
