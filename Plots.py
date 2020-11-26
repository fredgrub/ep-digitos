import sys
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import time
from datetime import datetime
import re

def get_data():
    data = [[],[],[]]
    ndig = [100, 1000, 4000]
    ps = [5, 10, 15]

    for x in range(len(ps)):
        for y in range(len(ndig)):
            report = "reports/report_{}_{}.txt".format(ndig[y], ps[x])
            with open(report, "r") as f:
                    lines = f.readlines()
                    data[y].append(float(lines[2][-7:-2]))

    return data[0], data[1], data[2]

def get_data1(ndig, p):
    report = "reports/report_{}_{}.txt".format(ndig, p)
    
    with open(report, "r", encoding="utf-8") as f:
        lines = f.read().replace('\n', ' ')
        matched = re.findall('[0-9]+\.[0-9]+', lines)
        matched = matched[1:]
        for index in range(len(matched)):
            matched[index] = float(matched[index])
        return matched

def data_mean(data):
    y = []

    for p in data:
        y.append((p[0] + p[1] + p[2])/3)
    return y

def general_plot():

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    labels = ['p = 5','p = 10','p = 15']
    data1, data2, data3 = get_data()

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    y = data_mean([data1, data2, data3])

    mean = ax.plot([0,1,2], y,'.-', color='#274c77', label="média dos ndig")

    rects1 = ax.bar(x - width, data1, width, label='ndig = 100', fc='#e0aaff')
    rects2 = ax.bar(x, data2, width, label='ndig = 1000', fc='#c77dff')
    rects3 = ax.bar(x + width, data3, width, label='ndig = 4000', fc='#9d4edd')

    ax.set_ylabel('Porcentagem de acertos')
    ax.set_title('Acertos por ndig e p')
    ax.set_ylim(ymin=80)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=2)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    #fig.tight_layout()
    filepath = 'figuras/acerto_ndig_p.png'
    print('Imagem salva em ./{}'.format(filepath))
    plt.savefig(filepath, dpi=150)

def compare_cases(ndig1, p1, ndig2, p2):
    fig1, ax1 = plt.subplots()

    caso1 = get_data1(ndig1,p1)
    caso2 = get_data1(ndig2,p2)

    labels = ['0','1','2','3','4','5','6','7','8','9']
    x = np.arange(len(labels))  # the label locations
    width = 0.25

    rects1 = ax1.bar(x - width/2, caso1, width, label='{}_{}'.format(ndig1,p1), fc='#e0aaff')
    rects2 = ax1.bar(x + width/2, caso2, width, label='{}_{}'.format(ndig2,p2), fc='#c77dff')

    ax1.set_ylabel('Porcentagem de acertos')
    ax1.set_title('Variação de acertos (caso {}_{} e {}_{})'.format(ndig1, p1, ndig2, p2))
    ax1.set_ylim(ymin=76)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax1.legend(loc=1)
    filepath = 'figuras/acertos-{}_{}-{}_{}.png'.format(ndig1, p1, ndig2, p2)
    print('Imagem salva em ./{}'.format(filepath))
    plt.savefig(filepath, dpi=150)

# depende dos erros e do onde errou
def get_k_rank(k, ndig, p, digito):
    # inicializa o dicionário
    topk_digitos = {}
    for i in range(1, k+1):
        topk_digitos[i] = {}
        
    with open('dados_mnist/test_index.txt') as f:
        asw = np.loadtxt(f, dtype=np.int8)
    
    erros = np.loadtxt('erros/erros_{}_{}.txt'.format(ndig, p))
    
    with open('erros/onde_errou_{}_{}.txt'.format(ndig, p), "r") as f:
        # iterando pelos erros
        for pos in f.readlines():
            pos = int(pos)
            d = asw[pos]
            #print(pos)
            if d == digito: # apenas olho para o digito escolhido
                erro_d = {}
                #print(pos)
                for val in range(10):
                    erro_d[erros[val, pos]] = val    
                erro_d = sorted(erro_d.items()) # ordena os erros
                #print(str(d) + '\n' + str(erro_d))
                for i in range(0, k):
                    dl = erro_d[i][1]
                    #print(dl)
                    if dl not in topk_digitos[i + 1]: # inicializa os valores
                        topk_digitos[i + 1][dl] = 1
                    else:
                        topk_digitos[i + 1][dl] += 1 # faz o update
                #print('----')
        return topk_digitos

def autolabel_deluxe(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_top_k(topks, k, ndig, p, digito, preset):
    labels = ['0','1','2','3','4','5','6','7','8','9']
    
    fig, ax = plt.subplots(preset[0], preset[1], figsize=(13,6))
    
    plt.subplots_adjust(top=0.9)
    fig.tight_layout(pad=2.0) # distância entre os plots
    plt.subplots_adjust(hspace=1)
    fig.suptitle('Histogramas do top {} digitos chutados no lugar do {}'.format(k, digito), size=15, y=1.05)
    
    for i in range(k):
        rect = ax[i].bar(list(topks[i + 1].keys()), topks[i + 1].values(), width=0.5, color='#c77dff')
        autolabel_deluxe(rect, ax[i])
        ax[i].set_title('{}º posição'.format(i + 1), size=14)
        
        ax[i].set_ylabel('Frequência', size=14)
        x = np.arange(len(labels))  # the label locations
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        
    filepath = 'figuras/top_{}-digito_{}-{}_{}.png'.format(k, digito, ndig, p)
    print('Imagem salva em ./{}'.format(filepath))
    plt.savefig(filepath, dpi=150)

def main():
    parser = argparse.ArgumentParser(
        description=
        '''
        Gera plots informativos sobre as classificações.
        ''')

    parser.add_argument('-g','--general',
                       action='store_true',
                       help='Gera o plot que compara o acerto geral de todos os casos.')

    parser.add_argument('-cc','--compare-cases', nargs='*',
                       help='Recebe dois casos distinto e gera um plot, comparando o acerto de cada dígito (e.g. -cc 1000 5 100 5).',
                       type=int)
    
    parser.add_argument('-e', '--erros', nargs='*',
                       help='Analisa as classificações que não obteram acerto - top 3. (e.g. -e 1000 5 4)',
                       type=int)
    
    
    args = parser.parse_args()
    
    general = args.general
    compare = args.compare_cases
    erros = args.erros

    if general:
        general_plot()
    
    if compare != None:
        ndig1 = compare[0]
        p1 = compare[1]
        ndig2 = compare[2]
        p2 = compare[3]

        print('Gerando plot das comparações...')
        time.sleep(2)

        compare_cases(ndig1, p1, ndig2, p2)
    
    if erros != None:
        k = 3
        preset = (1,3)
        ndig = erros[0]
        p = erros[1]
        digito = erros[2]
        
        print('Gerando plot do top 3 dígitos que apareceram no lugar do {}...'.format(digito))
        time.sleep(2)

        topks = get_k_rank(k,ndig,p,digito)
        plot_top_k(topks,k,ndig,p,digito,preset)

if __name__ == '__main__':
    main()