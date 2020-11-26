import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import time

def cos_sin(W, i, j, k):
    # ajuste de contagem
    i = i - 1
    j = j - 1
    k = k - 1
    
    if abs(W[i,k]) > abs(W[j,k]):
        t = -W[j,k]/W[i,k]
        c = 1/math.sqrt(1 + t**2)
        s = c*t
    else:
        t = -W[i, k]/W[j, k]
        s = 1/math.sqrt(1 + t**2)
        c = s*t    
    return c, s

def rot_Givens(W, n, m, i, j, c, s):
    # ajuste de contagem
    i = i - 1
    j = j - 1

    W[i,0:m], W[j,0:m] = c*W[i,0:m] - s*W[j,0:m], s*W[i,0:m] + c*W[j,0:m]

def solve(W, b, n, m):
    eps = 10e-10       # zero
    for k in range(1, m + 1):
        for j in range(n, k, -1):
            i = j - 1
            if abs(W[j - 1, k - 1]) > eps:
                cos, sin = cos_sin(W, i, j, k)
                rot_Givens(W, n, m, i, j, cos, sin)
                rot_Givens(b, n, 1, i, j, cos, sin)
    
    for k in range(m, 0, -1):
        soma = 0
        for j in range(k + 1, m + 1):
            soma += W[k - 1, j - 1]*b[j - 1, 0]
        b[k - 1, 0] = (b[k - 1, 0] - soma)/W[k - 1, k - 1]

def solve_multi(W, A, n, m, p):
    eps = 10e-10       # zero
    for k in range(1, p + 1):
        for j in range(n, k, -1):
            i = j - 1
            if abs(W[j - 1, k - 1]) > eps:
                cos, sin = cos_sin(W, i, j, k)
                rot_Givens(W, n, p, i, j, cos, sin)
                rot_Givens(A, n, m, i, j, cos, sin)
    
    for k in range(p, 0, -1):
        for j in range(1, m + 1):
            soma = 0
            for i in range(k+1,p+1):
                soma += W[k-1,i-1]*A[i-1,j-1]
            A[k-1,j-1] = (A[k-1,j-1] - soma)/W[k-1,k-1]
def erro(A,W,H,n,m):
    E = 0
    prod = W.dot(H)
    for i in range(1,n+1):
        for j in range(1,m+1):
            E += (A[i-1,j-1] - prod[i-1,j-1])**2
    return E

def norm(W,n,p):
    for j in range(1,p+1):
        s = 0
        for i in range(1,n+1):
            s += (W[i-1,j-1])**2
        s = math.sqrt(s)
        W[0:n,j-1] = W[0:n,j-1]/s
    return W

def set_matrix(A,p,m):
    B = np.zeros((p,m))
    for i in range(1,p+1):
        for j in range(1,m+1):
            if A[i-1,j-1] > 0:
                B[i-1,j-1] = A[i-1,j-1]
    return B

def NMF(A,n,m,p):
    # parâmetros
    eps = 10e-5
    it_max = 100
    # inicialização de variáveis de apoio
    e1 = 0
    e2 = 0
    t = 0
    # inicialização/setagem das matrizes de interesse
    W = np.random.rand(n,p)
    A_copia = np.copy(A)
    At_copia = np.copy(np.transpose(A))
    H = np.zeros((p,m))
    
    while(t < it_max):
        if (t > 1 and abs(e1 - e2) < eps):
            break
    
        A = np.copy(A_copia)              # recuperar A original
        W = norm(W,n,p)                   # normalizar W
        solve_multi(W, A, n, m, p)        # WH = A
        H = set_matrix(A,p,m)             # H a partir de A

        At = np.copy(At_copia)            # computar A^t
        Ht = np.copy(np.transpose(H))     # computar H^t
        solve_multi(Ht, At, m, n, p)      # (WH)^t = H^tW^t = A^t
        Wt = set_matrix(At, p, n)         # W^t a partir de A^t
        W = np.copy(np.transpose(Wt))     # computar W (a partir de W^t)
        
        if t % 2 != 0:                    # erro em iteração ímpar
            e1 = erro(A_copia, W, H, n, m)
        else:                             # erro em iteração par
            e2 = erro(A_copia, W, H, n, m)
        
        t += 1
    return W, H    

def train_dig(digito, ndig_treino, p):
    n = 784
    A = np.loadtxt("dados_mnist/train_dig" + str(digito) + ".txt")
    A = A[:,:ndig_treino]
    m = ndig_treino
    Wd, H = NMF(A,n,m,p)
    return Wd

def train_all(train=[], ndig_treino = [100,1000,4000], ps = [5,10,15], specific=False):
    print('Iniciando os treinos...\nDados serão salvos na pasta ./dig_treino/')
    time.sleep(2) # tempo para ler
    if specific:
        # get the data
        ndig = train[0]
        p = train[1]
        digito = train[2]
        # train
        start = time.time()
        print('--- Treino do dígito {} para o caso {}_{}\nTreinando...'.format(digito, ndig, p))
        Wd = train_dig(digito, ndig, p)
        elapsed_time_lc = (time.time() - start)
        filename = "dig_treino/W_" + str(digito) + "_" + str(ndig) + "_" + str(p) + ".txt"
        print('Treino bem sucedido! O tempo gasto foi de {} segundos.'.format(round(elapsed_time_lc)))
        with open(filename, "w+") as f:
            f.write(str(elapsed_time_lc) + "\n")
            np.savetxt(f, Wd)
    else:
        for ndig in ndig_treino:
            for p in ps:
                print('--- Treino dos dígitos para o caso {}_{}'.format(ndig, p))
                for digito in range(10):
                    start = time.time()
                    print('Treino do dígito {}'.format(digito))
                    Wd = train_dig(digito, ndig, p)
                    elapsed_time_lc = (time.time() - start)
                    filename = "dig_treino/W_" + str(digito) + "_" + str(ndig) + "_" + str(p) + ".txt" 
                    print('Treino bem sucedido! Preparando o próximo.\n')
                    with open(filename, "w+") as f:
                        f.write(str(elapsed_time_lc) + "\n")
                        np.savetxt(f, Wd)

def teste_a():
    n = 64
    m = 64
    W = np.zeros((n,m))
    b = np.zeros((n,1))
    for i in range(1, n+1):
        for j in range(1, m+1):
            if i == j:
                W[i - 1, j - 1] = 2
            elif abs(i - j) == 1:
                W[i - 1, j - 1] = 1
            elif abs(i - j) > 1:
                W[i - 1, j - 1] = 0
            b[i - 1, 0] = 1
    solve(W,b,n,m)
    return b

def teste_b():
    n = 20
    m = 17
    W = np.zeros((n,m))
    b = np.zeros((n,1))
    
    for i in range(1,n+1):
        for j in range(1,m+1):
            if abs(i - j) <= 4:
                W[i - 1, j - 1] = 1.0/(i + j - 1)
            elif abs(i - j) > 4:
                W[i - 1, j - 1] = 0
        b[i - 1, 0] = i
    
    solve(W, b, n, m)
    return b[0:m,:]

def teste_c():
    n = 64
    m = 3
    p = 64
    W = np.zeros((n,p))
    A = np.zeros((n,m))
    
    for i in range(1,n+1):
        for j in range(1, p+1):
            if abs(i - j) == 1:
                W[i - 1, j - 1] = 1
            elif abs(i - j) > 1:
                W[i - 1, j - 1] = 0
        A[i - 1, 0] = 1
        A[i - 1, 1] = i
        A[i - 1, 2] = 2*i
    solve_multi(W,A,n,m,p)
    return A

def teste_d():
    n = 20
    m = 3
    p = 17
    W = np.zeros((n,p))
    A = np.zeros((n,m))
    
    for i in range(1,n+1):
        for j in range(1,p+1):
            if abs(i - j) <= 4:
                W[i - 1, j - 1] = 1.0/(i + j - 1)
            elif abs(i - j) > 4:
                W[i - 1, j - 1] = 0
        A[i - 1, 0] = 1
        A[i - 1, 1] = i
        A[i - 1, 2] = 2*i
    solve_multi(W,A,n,m,p)
    return A[0:p,:]

def test_all():
    print('Iniciando os testes... \nDados serão salvos no arquivo ./testes.txt')
    time.sleep(2)
    with open('testes.txt', 'w+') as f:
        f.write('Teste a:\n')
        solution = teste_a()
        f.write(str(solution) + '\n\n')
        print('Teste (a) concluído')

        f.write('Teste (b):\n')
        solution = teste_b()
        f.write(str(solution) + '\n\n')
        print('Teste (b) concluído')

        
        f.write('Teste (c):\n')
        solution = teste_c()
        f.write(str(solution) + '\n\n')
        print('Teste (c) concluído')

        
        f.write('Teste (d):\n')
        solution = teste_b()
        f.write(str(solution) + '\n\n')
        print('Teste (d) concluído\n')

def main():
    parser = argparse.ArgumentParser(
        description="Executa os treinos das matrizes Wd's e também roda os testes sugeridos.")

    parser.add_argument('--train',
                       action='store_true',
                       help='Executa todos os treinos possíveis.')

    parser.add_argument('--test',
                       action='store_true',
                       help='Executa todos os testes pedidos.')
    
    parser.add_argument('-l','--list', nargs='*',
                       help='Recebe os argumentos para um teste específico (e.g. 100 5 0).',
                       type=int)
    # parser.add_argument('test_all',
    #                    metavar='test',
    #                    type=bool,
    #                    help='Executa todos os testes pedidos.')

    args = parser.parse_args()

    train = args.train
    test = args.test
    specify = args.list

    if train and test:
        test_all()
        time.sleep(2)
        train_all()
    elif train:
        train_all()
    elif test:
        test_all()
    else:
        train_all(train=specify, specific=True)


if __name__ == '__main__':
    main()