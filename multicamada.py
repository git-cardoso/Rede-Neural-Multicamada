import numpy as np


class Aprendizagem:

    def sigmoid(self, soma):
        return 1 / (1 + np.exp(-soma))

    def sigmoidDerivada(self, sig):
        return sig * (1 - sig)

    def __init__(self,epocas):
        self.epocas = epocas
        entradas = np.array([[0, 0],
                             [0, 1],
                             [1, 0],
                             [1, 1]])

        saidas = np.array([[0], [1], [1], [0]])
        contagem = 0
        pesos0 = 2 * np.random.random((2, 3)) - 1
        pesos1 = 2 * np.random.random((3, 1)) - 1
        taxaAprendizagem = 0.5
        momento = 1

        for j in range(0,self.epocas):
            camada_entrada = entradas
            soma_sinapse0 = np.dot(camada_entrada, pesos0)
            camada_oculta = self.sigmoid(soma_sinapse0)
            soma_sinapse1 = np.dot(camada_oculta, pesos1)
            camada_saida = self.sigmoid(soma_sinapse1)
            erro_camada_saida = saidas - camada_saida
            media_absoluta = np.mean(np.abs(erro_camada_saida))
            print("Erro: %.2f " % media_absoluta)

            derivada_saida = self.sigmoidDerivada(camada_saida)
            delta_saida = erro_camada_saida * derivada_saida
            pesos1Transposta = pesos1.T
            delta_saida_XPeso = delta_saida.dot(pesos1Transposta)
            delta_camada_oculta = delta_saida_XPeso * self.sigmoidDerivada(camada_oculta)
            camadaOcultaTransposta = camada_oculta.T
            pesosNovo1 = camadaOcultaTransposta.dot(delta_saida)
            pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
            camadaEntradaTransposta = camada_entrada.T
            pesosNovo0 = camadaEntradaTransposta.dot(delta_camada_oculta)
            pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
            contagem = contagem + 1


if __name__ == '__main__':
    Aprendizagem(20000)
