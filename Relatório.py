import sys
import argparse
from datetime import datetime
import time
import numpy as np
import math
from Treinos import solve_multi

def read_all_digits(ndig_treino, p): # leitura dos treinos
    all_dig = []
    for digito in range(10):
        filename = 'dig_treino/W_{}_{}_{}.txt'.format(digito, ndig_treino, p)
        with open(filename) as f:
            time = f.readline()
            Wd = np.loadtxt(f)
            all_dig.append(Wd)
    return all_dig

class PPTO: # Palpitômetro
    def __init__(self):
        # abrindo os arquivos
        with open('dados_mnist/test_images.txt') as f:
            self.A = np.loadtxt(f)
        with open('dados_mnist/test_index.txt') as f:
            self.asw = np.loadtxt(f, dtype=np.int8)
        
        # setando variáveis
        self.n, self.m = self.A.shape
        self.p = 0
        
        # vetores auxiliares
        self.D = np.zeros(self.m, dtype=np.int8)
        self.E = np.zeros(self.m)
        
        # todos os erros
        self.all_E = np.zeros((10, self.m))
        self.onde_errou = []

        # setando o contador
        self.count_dig = {}
        for digito in self.asw:
            sdigito = str(digito)
            if sdigito not in self.count_dig:
                self.count_dig[sdigito] = 1
            else:
                self.count_dig[sdigito] += 1

    def MMQS(self, Wds):
        self.n, self.p = Wds[0].shape # todos os 9 Wd's têm a mesma forma (shape)
        num_dig = len(Wds)
        for d in range(num_dig):
            # setando as variáveis
            Wd = Wds[d].copy()
            cpA = self.A.copy()
            
            # resolvendo o sistema simultâneo
            solve_multi(Wd, cpA, self.n, self.m, self.p)
            WH = np.dot(Wds[d], cpA[:self.p,:])
            C = np.subtract(self.A, WH)
            
            # calculando o erro de cada coluna
            for col in range(self.m):
                err = 0.0
                for row in range(self.n):
                    err += (C[row, col])**2
                err = math.sqrt(err)
                
                # coletando todos os erros
                self.all_E[d, col] = err
                
                # gerando os palpites
                if d == 0:
                    self.E[col] = err
                elif err < self.E[col]:
                    self.E[col] = err
                    self.D[col] = d

    def reset_stats(self):
        # Resetando os valores
        self.D = np.zeros(self.m, dtype=np.int8)
        self.E = np.zeros(self.m)

        self.all_E = np.zeros((10, self.m))
        self.onde_errou = []

    def generate_report(self, ndig_treino, save=True):
        # inicializando o contador de acertos
        correct = 0
        correct_dig = {}
        
        for num in range(10):
            snum = str(num)
            correct_dig[snum] = 0
        
        # contando os acertos
        for row in range(self.m):
            if self.D[row] == self.asw[row]:
                correct += 1
                correct_dig[str(self.D[row])] += 1
            else:
                self.onde_errou.append(row)
            
        # gerando o relatório
        if save:
            with open('reports/report_{}_{}.txt'.format(ndig_treino, self.p), 'w+') as f:
                f.write('Relatório do caso p = {} e ndig_treino = {}\n\n'.format(self.p, ndig_treino))
                f.write('Percentual total de acertos: {:.2f}%\n\n'.format((correct/self.m)*100))
                
                for dig, right in correct_dig.items():
                    f.write('Digito ' + dig + '\n')
                    total = self.count_dig[dig]
                    f.write('Acertos: {}/{}\n'.format(right, total))
                    f.write('Percentual de acertos: {:.2f}%\n\n'.format((right/total)*100))
            
            with open('erros/erros_{}_{}.txt'.format(ndig_treino, self.p), 'w+') as f:
                np.savetxt(f, self.all_E)
            with open('erros/onde_errou_{}_{}.txt'.format(ndig_treino, self.p), 'w+') as f:
                for item in self.onde_errou:
                    f.write(str(item) + '\n')
        else:
            print('Relatório do caso p = {} e ndig_treino = {}\n'.format(self.p, ndig_treino))
            time.sleep(2)
            print('Percentual total de acertos: {:.2f}%\n'.format((correct/self.m)*100))

            for dig, right in correct_dig.items():
                    print('Digito {}'.format(dig))
                    total = self.count_dig[dig]
                    print('Acertos: {}/{}'.format(right, total))
                    print('Percentual de acertos: {:.2f}%\n'.format((right/total)*100))
            time.sleep(2)
            print('OBS: os arquivos relacionados ao erro (usados nas discussões) não são gerados nessa execução.')

def run_classifier(case=[]):
    print('Carregando as informações...')
    pp = PPTO()
    if len(case) == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Começo das classificações às {}\n".format(current_time))

        ndig_treinos = [100, 1000, 4000]
        ps = [5, 10, 15]

        for ndig_treino in ndig_treinos:
            for p in ps:
                print('Caso ndig_treino = {} com p = {}'.format(ndig_treino, p))
                start_time = time.time() # start counting time

                Wds = read_all_digits(ndig_treino, p)
                pp.MMQS(Wds)

                elapsed_time = time.time() - start_time # stop counting time
                fmt_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print('Tempo gasto para o MMQ: {}\n'.format(fmt_time))
                
                # gerando as informações sobre a classificação
                pp.generate_report(ndig_treino)
                # resetando os valores para uma nova classificação
                pp.reset_stats()
                

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Classificações concluídas às {}'.format(current_time))
    else:
        ndig_treino = case[0]
        p = case[1]
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Começo da classificação às {}".format(current_time))
        start_time = time.time() # start counting time

        Wds = read_all_digits(ndig_treino, p)
        pp.MMQS(Wds)

        elapsed_time = time.time() - start_time # stop counting time
        fmt_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('Tempo gasto para o MMQ: {}'.format(fmt_time))
        
        # gerando as informações sobre a classificação
        pp.generate_report(ndig_treino, save=False)
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Classificação concluída às {}'.format(current_time))

def main():
    parser = argparse.ArgumentParser(
        description=
        '''
        Executa a classificação dos dígitos para parâmetros p e ndig_treino 
        especificados ou executa com todos os parâmetros possíveis.
        ''')

    parser.add_argument('--all',
                       action='store_true',
                       help='Executa todas as classificações possíveis.')
    
    parser.add_argument('-s','--specify', nargs='*',
                       help='Recebe os argumentos ndig_treino e p (e.g. -s 1000 5).',
                       type=int)
    
    args = parser.parse_args()

    ex_all = args.all
    case = args.specify

    if ex_all:
        run_classifier()
    else:
        run_classifier(case=case)

if __name__ == "__main__":
    main()