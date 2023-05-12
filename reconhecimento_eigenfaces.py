#importa o OpenCv
import cv2
#atribui a camera na variavel webcam
webcam = cv2.VideoCapture(0)
#carrega o haarcascade de rostos na variavel
classificadorRosto = cv2.CascadeClassifier('venv/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#atribui o engenFace na variavel
reconhecedor  = cv2.face.EigenFaceRecognizer_create()
#carrega o arquivo treinado no reconhecedor
reconhecedor.read("classificadorEigen.yml")
#define a largura e altura padrão das imagens treinadas e capturadas
largura, altura = 220,220

#cria um arquivo de fonte para escrever na imgem
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

nomes = ['Sergio', 'Gabriel']

while (True):
    #realiza a leitura da webcam
    conectado, imagem = webcam.read()
    #converte em escala de cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #encontra os rostos na imagem
    facesDetectadas = classificadorRosto.detectMultiScale(imagemCinza,minSize=(200,200))

    #passa por todos os rostos encontrados
    for(x, y, l, a) in facesDetectadas:
        #recorta apenas a imagem da face encontrada
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x +l], (largura, altura))
        #desenha o retangulo na imagem
        cv2.rectangle(imagem, (x,y), (x+l, y+a), (0, 0, 255), 2)
        #pega o id da pessa reconhecidas e a porcentagem de acerto
        id, confianca = reconhecedor.predict(imagemFace)
        #cv2.imshow("Face", imagemFace)

        #escreve em baixo do retangulo da imagem
        cv2.putText(imagem, nomes[id-1] + " - " +str(confianca), (x, y+a+30), font, 2, (0,0,255))
    cv2.imshow("Face", imagem)

    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()