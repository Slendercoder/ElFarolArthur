print("Importando paquetes...")
from random import choice, sample, randint
import numpy as np
import pandas as pd
import redes
print("Listo!")

def distancia(x, y):
    return abs(x - y)

class Predictor:
    def __init__(self, long_memoria, num_agentes):
        self.ventana = randint(1, long_memoria)
        self.ciclico = choice([True, False])
        self.espejo = choice([True, False])
        # self.codigo = str(self.ventana) + "-" + str(self.ciclico) + "-" + str(self.espejo)
        self.precision = 0
        self.prediccion = []

    def predecir(self, memoria, num_agentes):
        long_memoria = len(memoria)
        ciclico = self.ciclico
        ventana = self.ventana
        espejo = self.espejo
        if ciclico:
            # indices = list(range(long_memoria - retardo, -1, -retardo))
            indices = list(range(long_memoria - 1, -1, -ventana))
            valores = [memoria[x] for x in indices]
        else:
            # valores = historia[max(long_memoria-retardo-ventana+1, 0):max(long_memoria-retardo+1, 0)]
            valores = memoria[-ventana:]
        try:
            prediccion = int(np.mean(valores))
        except:
            prediccion = memoria[-1]
        if espejo:
            prediccion = num_agentes - prediccion
        self.prediccion.append(prediccion)
        # print(self)
        # print("Memoria para tomar decision:", memoria)
        # print("Valores:", valores)
        # print("Prediccion:", prediccion)

    def __str__(self):
        ventana = str(self.ventana)
        ciclico = "ciclico" if self.ciclico else "ventana"
        espejo = "-espejo" if self.espejo else ""
        return ventana + "-" + ciclico + espejo

class Agente:
    def __init__(self, estados, scores, vecinos, predictores, predictor_activo):
        self.estado = estados # lista
        self.score = scores # lista
        self.vecinos = vecinos # lista
        self.predictores = predictores # lista
        self.predictor_activo = predictor_activo

    def __str__(self):
        return "E:{0}, S:{1}, V:{2}, P:{3}".format(self.estado, self.score, self.vecinos, str(self.predictor_activo[-1]))

class Bar:
    def __init__(self, num_agentes, umbral, long_memoria, num_predictores, conectividad, identificador):
        self.num_agentes = num_agentes
        self.umbral = umbral
        self.long_memoria = long_memoria
        self.num_predictores = num_predictores
        self.conectividad = conectividad
        self.identificador = identificador
        self.historia = []
        self.predictores = []
        for i in range(100):
            p = Predictor(self.long_memoria, self.num_agentes)
            if str(p) not in [str(pr) for pr in self.predictores]:
                self.predictores.append(p)
        self.agentes = []
        for i in range(self.num_agentes):
            predictores_agente = sample(self.predictores, self.num_predictores)
            # print(f"Predictores agente {i}:", str([str(p) for p in predictores_agente]))
            self.agentes.append(Agente([randint(0,1)], [], [], predictores_agente, [choice(predictores_agente)]))
        self.calcular_asistencia() # Encuentra la asistencia al bar
        self.calcular_puntajes() # Encuentra los puntajes de los agentes
        self.actualizar_predicciones() # Predice de acuerdo a la primera asistencia aleatoria
        self.leer_red() # Lee red desde archivo para incluir vecinos

    def calcular_estados(self):
        for a in self.agentes:
            prediccion = a.predictor_activo[-1].prediccion[-1] / self.num_agentes
            if prediccion <= self.umbral:
                a.estado.append(1)
            else:
                a.estado.append(0)

    def calcular_asistencia(self):
        asistencia = np.sum([a.estado[-1] for a in self.agentes])
        self.historia.append(asistencia)

    def calcular_puntajes(self):
        asistencia = self.historia[-1]/self.num_agentes
        for a in self.agentes:
            if a.estado[-1] == 1:
                if asistencia > self.umbral:
                    a.score.append(-1)
                else:
                    a.score.append(1)
            else:
                a.score.append(0)

    def leer_red(self):
        net = {}
        aux = '-' if self.identificador != '' else ''
        In = open("data/redes/connlist" + aux + str(self.identificador) + ".dat", "r")
        for line in In:
            v = list(map(int, line.split()))
            if v[0] not in net.keys():
                net[v[0]] = [v[1]]
            else:
                net[v[0]].append(v[1])
            if v[1] not in net.keys():
                net[v[1]] = [v[0]]
            else:
                net[v[1]].append(v[0])
        In.close()
        # print('Red', net)
        for i in range(len(self.agentes)):
            try:
                self.agentes[i].vecinos = net[i]
            except:
                self.agentes[i].vecinos = []

    def actualizar_predicciones(self):
        historia = self.historia[-self.long_memoria:]
        # print("Historia para predecir:", historia)
        for p in self.predictores:
            p.predecir(historia, self.num_agentes)

    def actualizar_precision(self):
        historia = self.historia[-self.long_memoria - 1:]
        for p in self.predictores:
            predicciones = p.prediccion[-self.long_memoria:]
            # print("Historia vs prediccion", historia, predicciones)
            precision = [distancia(historia[i + 1], predicciones[i]) for i in range(len(historia) - 1)]
            p.precision = np.mean(precision)

    def escoger_predictor(self):
        for a in self.agentes:
            precisiones = [p.precision for p in a.predictores]
            index_min = np.argmin(precisiones)
            a.predictor_activo.append(a.predictores[index_min])

    def juega_ronda(self):
        self.calcular_estados()
        self.calcular_asistencia()
        self.calcular_puntajes()
        self.actualizar_precision()
        self.escoger_predictor()
        self.actualizar_predicciones()

    def crea_dataframe_agentes(self):
        ronda = []
        agente = []
        estado = []
        puntaje = []
        politica = []
        num_iteraciones = len(self.historia) - 1
        for i in range(self.num_agentes):
            for r in range(num_iteraciones):
                agente.append(i)
                ronda.append(r)
                estado.append(self.agentes[i].estado[r])
                puntaje.append(self.agentes[i].score[r])
                politica.append(str(self.agentes[i].predictor_activo[r]))
        data = pd.DataFrame.from_dict(\
                                    {\
                                    'Ronda': ronda,\
                                    'Agente': agente,\
                                    'Estado': estado,\
                                    'Puntaje': puntaje,\
                                    'Politica': politica\
                                    })

        id = self.identificador if self.identificador != '' else 'A'
        data['Identificador'] =  id
        data['Memoria'] = self.long_memoria
        data['Num_predic'] = self.num_predictores
        data['Conectividad'] = self.conectividad
        data = data[['Memoria', 'Num_predic', 'Identificador','Ronda','Agente','Estado','Puntaje','Politica']]
        return data

def guardar(dataFrame, archivo, inicial):
    archivo = "data/" + archivo
    if inicial:
        #os.remove(archivo)
        dataFrame.to_csv(archivo, index = False)
    else:
        with open(archivo, 'a') as f:
            dataFrame.to_csv(f, header=False, index=False)

def simulacion(num_agentes, umbral, long_memoria, num_predictores, num_rondas, conectividad, inicial=True, identificador='', DEB=False):
    bar = Bar(num_agentes, umbral, long_memoria, num_predictores, conectividad, identificador)
    if DEB:
        print("****************************")
        print("Agentes iniciales:")
        for a in bar.agentes:
            print(a)
        print("****************************")
        print("")
    for i in range(num_rondas):
        if DEB:
            print("Ronda", i)
            print("Historia:", bar.historia)
            for p in bar.predictores:
                print(f"Predictor: {str(p)} - Prediccion: {p.prediccion} - Precision: {p.precision}")
            print("****************************")
        bar.juega_ronda()
        if DEB:
            for a in bar.agentes:
                print(a)
    data = bar.crea_dataframe_agentes()
    guardar(data, 'simulacion-' + str(long_memoria) + '-' + str(num_predictores) + '-' + str(conectividad) +'.csv', inicial)
    # guardar(data, 'agentes.csv', inicial)

def correr_sweep(num_experimentos, num_agentes, umbral, num_rondas):
    print('********************************')
    print('Corriendo simulaciones...')
    print('********************************')
    print("")
    memoria = [1, 3, 6, 9, 12, 15, 18]
    predictores = [1, 3, 6, 9, 12]
    conectividades = [0.01 * x for x in range(10)] + [0.1 * x for x in range(1, 11)]
    conectividades = [round(x, 2) for x in conectividades]
    inicial = True
    identificador = 0
    for d in memoria:
        for k in predictores:
            if k <= d:
                for p in conectividades:
                    print('Corriendo experimentos con parametros:')
                    print('Memoria:', d, 'Predictores:', k, 'Conectividad:', p)
                    for i in range(num_experimentos):
                        redes.random_graph(num_agentes, p, imagen=False, identificador=identificador)
                        simulacion(num_agentes, umbral, d, k, num_rondas, p, inicial=inicial, identificador=identificador)
                        identificador += 1
                        inicial = False
####################################

num_agentes = 100
umbral = .6
# long_memoria = 5 # Longitud de la historia
# num_predictores = 1 # Cantidad de predictores en la bolsa de cada agente
num_rondas = 100
num_experimentos = 100
correr_sweep(num_experimentos, num_agentes, umbral, num_rondas)
