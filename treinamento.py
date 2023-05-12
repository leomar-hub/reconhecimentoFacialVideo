#importa o OpenCv para o projeto
import cv2
#importa a biblioteca do sistema
import os
#importa o Numpy para utilizarmos na crição de vetores e matrizes
import numpy as np

#carrega o algoritmo do eigenface e coloca na memória
#eigenface = cv2.face.EigenFaceRecognizer_create()
#carrega o algoritmo do FisherFace e coloca na memória
fisherface = cv2.face.FisherFaceRecognizer_create()
#carrega o algoritmo do LBPH e coloca na memória
lbph = cv2.face.LBPHFaceRecognizer_create()

#cria um método que carrega as imagens e os Ids de cada pessoa
def getImagensId():
    #pega o caminho atual da imagem
    caminhos = [os.path.join('fotos',f) for f in os.listdir('fotos')]
    #print(caminhos)
    #cria um vetor para salvar as imagens dos rostos
    faces = []
    #cria um vetor para adicionar os ids de cada pessoa referente as imagens
    ids = []

    #passa por todos os caminhos de imagens
    for caminhoImagem in caminhos:
        #carrega a imagem da face convertida em escala de cinza
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        #pega o id da pessoa baseado no nome do arquivo
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #adiciona o id atual ao vetor
        ids.append(id)
        #adiciona a imagem atual ao vetor
        faces.append(imagemFace)
        #mostra a imagem na tela
        cv2.imshow("Face", imagemFace)
        #da uma pausa de 20ms na execução
        cv2.waitKey(20)
    #retorna um vetor com todos os ids e imagens
    return np.array(ids), faces

#chama o método salvando nas variáveis
ids, faces = getImagensId()
print(ids)

#print("treinando...Eigen...")
#inicia o treinamento
#eigenface.train(faces, ids)
#salva o arquivo treinado na pasta do projeto
#eigenface.write('classificadorEigen.yml')

print("treinando...Fisher...")

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

print("treinando...LBPH...")

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado!!")