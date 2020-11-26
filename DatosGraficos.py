import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def leer_datos(memorias, predictores, conectividades):
    names = ['Memoria', 'Num_predic', 'Identificador', 'Ronda', 'Agente', 'Estado', 'Puntaje', 'Politica', 'Prediccion', 'Precision']
    df_list = []
    for d in memorias:
        for k in predictores:
            for p in conectividades:
                print(f"Leyendo datos sweep memoria {d} predictores {k} y conectividad {p}")
                archivo = './data/simulacion-' + str(d) + "-" + str(k) + '-' + str(p) + ".csv"
                print(f"Cargando datos de archivo {archivo}...")
                try:
                    aux = pd.read_csv(archivo, names=names, header=None)
                    if 'Memoria' in aux['Memoria'].unique():
                        aux= aux.iloc[1:]
                    aux['Conectividad'] = p
                    # print(aux.head())
                    df_list.append(aux)
                    print("Listo")
                except:
                    print("Archivo no existe! Saltando a siguiente opción")
    print("Preparando dataframe...")
    data = pd.concat(df_list)
    print(data.head())
    try:
        # data = data.dropna()
        data['Conectividad'] = data['Conectividad'].astype(float)
        data['Memoria'] = data['Memoria'].astype(int)
        data['Num_predic'] = data['Num_predic'].astype(int)
        data['Identificador'] = data['Identificador'].astype(int)
        data['Ronda'] = data['Ronda'].astype(int)
        data['Agente'] = data['Agente'].astype(int)
        data['Estado'] = data['Estado'].astype(int)
        data['Puntaje'] = data['Puntaje'].astype(int)
        data['Politica'] = data['Politica'].astype(str)
        data['Prediccion'] = data['Prediccion'].astype(int)
        data['Precision'] = data['Precision'].astype(float)
    except:
        data = data.iloc[1:]
        print(data.head())
        # data = data.dropna()
        data['Conectividad'] = data['Conectividad'].astype(float)
        data['Memoria'] = data['Memoria'].astype(int)
        data['Num_predic'] = data['Num_predic'].astype(int)
        data['Identificador'] = data['Identificador'].astype(int)
        data['Ronda'] = data['Ronda'].astype(int)
        data['Agente'] = data['Agente'].astype(int)
        data['Estado'] = data['Estado'].astype(int)
        data['Puntaje'] = data['Puntaje'].astype(int)
        data['Politica'] = data['Politica'].astype(str)
        data['Prediccion'] = data['Prediccion'].astype(int)
        data['Precision'] = data['Precision'].astype(float)
    data = data[['Conectividad','Memoria','Num_predic','Identificador','Ronda','Agente','Estado','Puntaje','Politica', 'Prediccion', 'Precision']]
    print("Shape:", data.shape)
    print("Memoria value counts:", data['Memoria'].value_counts())
    print("Predictores value counts:", data['Num_predic'].value_counts())
    print("Conectividades value counts:", data['Conectividad'].value_counts())
    print("Agente value counts:", data['Agente'].value_counts())
    return data

def dibuja_asistencia_vs(data, variable='Memoria'):
    Numero_agentes = max(data['Agente']) + 1
    aux = data.groupby([variable, 'Identificador', 'Ronda'])['Estado']\
        .sum().reset_index()
    aux.columns = [variable,
                   'Identificador',
                   'Ronda',
                   'Asistencia_total']

    aux['Asistencia_total'] = (aux['Asistencia_total']/Numero_agentes)*100
    rondas = aux['Ronda'].unique()
    aux1 = aux[aux['Ronda'] > rondas[-5]]
    aux1 = aux1.groupby([variable, 'Identificador'])['Asistencia_total']\
        .mean().reset_index()
    aux1.columns = [variable,
                   'Identificador',
                   'Asistencia_total']
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    for v, grp in aux.groupby(variable):
        sns.lineplot(x=grp['Ronda'], y=grp['Asistencia_total'], label=v, ax=ax[0])
    ax[0].legend().set_title(variable)
    sns.boxplot(x=aux1[variable], y=aux1['Asistencia_total'], ax=ax[1])
    ax[0].set_xlabel('Ronda')
    ax[0].set_ylabel('Asistencia promedio')
    ax[0].set_title('Asistencia promedio por ronda')
    ax[1].set_xlabel(variable)
    ax[1].set_ylabel('Asistencia últimas 5 rondas')
    ax[1].set_title('Distribución asistencia en las últimas 5 rondas')

def dibujar_puntaje_vs(data, variable):
    fig, ax = plt.subplots(2,1,figsize=(8,8))
    data_aux = data.groupby([variable, 'Identificador'])['Puntaje'].mean().reset_index()
    sns.boxplot(
        x=data_aux[variable],
        y=data_aux['Puntaje'],
        ax=ax[0]
    )
    df = data.groupby([variable, 'Identificador', 'Agente'])['Puntaje'].mean().reset_index()
    for key, grp in df.groupby(variable):
        sns.distplot(grp['Puntaje'], ax=ax[1], label=key)
    ax[1].legend().set_title(variable)
    ax[0].set_ylabel('Puntaje')
    ax[0].set_title('Distribución puntaje vs ' + variable)
    ax[1].set_xlabel('Puntaje promedio')
    ax[1].set_ylabel('')
    ax[1].set_title('Distribución de la recompensa\n por cada ' + variable)
    fig.tight_layout()

def dibuja_usopredictores_vs(data, variable):
    df = data.groupby(variable)['Politica'].value_counts().rename_axis([variable, 'Politica']).reset_index(name='Usos')
    g = sns.FacetGrid(df, col=variable, aspect=1.5, height=4, sharex=False)
    g.map(sns.barplot, "Politica", "Usos")
    g.set_xticklabels(rotation=90)

def dibuja_puntajepredictor_vs(data, variable):
    data['Politica_lag'] = data.groupby([variable, 'Identificador', 'Agente'])['Politica'].transform('shift', -1)
    df = data.groupby([variable, 'Politica_lag', 'Identificador'])['Puntaje'].mean().reset_index()
    for p, Grp in df.groupby(variable):
        grp = Grp.sort_values(by='Puntaje')
        fig, ax = plt.subplots(1,1,figsize = (8,5))
        sns.swarmplot(x=grp['Politica_lag'], y=grp['Puntaje'])
        fig.suptitle(variable + ': ' + str(p), fontsize=14)
        plt.xticks(rotation=90)
