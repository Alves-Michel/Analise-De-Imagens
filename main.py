import cv2
from deepface import DeepFace
import os
import logging



# Listar arquivos no diretório atual
arquivos = os.listdir()

# Iterar sobre cada arquivo no diretório
for arquivo in arquivos:
    # Verificar se o arquivo é uma imagem JPEG
    if arquivo.lower().endswith(".jpeg"):
        # Ler a imagem
        imagem = cv2.imread(arquivo)
        
        # Verificar se a leitura da imagem foi bem-sucedida
        if imagem is not None:
            # Passar a imagem para DeepFace para análise
            resultado = DeepFace.analyze(imagem, actions=["age", "emotion"], enforce_detection=False)
            
            # Verificar se o resultado é uma lista e pegar o primeiro item
            if isinstance(resultado, list) and len(resultado) > 0:
                # Extrair idade e emoção do primeiro item da lista
                idade = resultado[0]['age']
                emocao = resultado[0]['dominant_emotion']
                
                # Imprimir os valores da idade e emoção
                print(f"Age: {idade}")
                print(f"Dominant Emotion: {emocao}")
            else:
                print(f"Nenhuma face detectada em: {arquivo}")
            
        else:
            print(f"Erro ao ler o arquivo: {arquivo}")
