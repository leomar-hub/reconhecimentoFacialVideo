import cv2
import os
import numpy as np

def train_eigenface():
    eigenface = cv2.face.EigenFaceRecognizer_create()

    def getImagensId():
        caminhos = [os.path.join('fotos',f) for f in os.listdir('fotos')]
        faces = []
        ids = []

        for caminhoImagem in caminhos:
            imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
            ids.append(id)
            faces.append(imagemFace)
            cv2.imshow("Face", imagemFace)
            cv2.waitKey(20)
        return np.array(ids), faces

    ids, faces = getImagensId()

    print("treinando...Eigen...")
    eigenface.train(faces, ids)
    eigenface.write('classificadorEigen.yml')

    print("Treinamento realizado!!")
