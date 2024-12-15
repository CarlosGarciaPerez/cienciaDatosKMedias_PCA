# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:23:03 2022

@author: PC-GARCIA
"""
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash import Input, Output

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from sklearn.cluster import KMeans
#from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn import preprocessing
from sklearn.decomposition import PCA
#from skimage import data
#from yellowbrick.cluster import SilhouetteVisualizer

from dash.dependencies import State

import conecDataBaseMySqlv4

""" app = dash.Dash(__name__) """
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

app.title = 'Ciencia de datos en el abandono escolar'

conexDB = conecDataBaseMySqlv4.classConecionDB()
dataframe2023 = pd.DataFrame( conexDB.obtenerDataSet2023())


# conversion y normalizacion de datos 2023 (DATAFRAME  2023)  para algoritmo KMEANS
data2023 = dataframe2023.copy()
data2023= data2023.drop(['id','Estado','Número de Habitantes[1]'], axis =1)  
dataEscalda = preprocessing.Normalizer().fit_transform(data2023)
dataEscalda2023= dataEscalda.copy()
#X= dataEscalda.copy()
#print(X)

#c copia valores conjunto de datos  
dataKmedias2023 = data2023.copy()

estados = ['Aguascalientes', 'Baja California','Baja California Sur', 'Campeche', 'Coahuila de Zaragoza', 
           'Colima', 'Chiapas', 'Chihuahua', 'Ciudad de México','Durango', 
           'Guanajuato', 'Guerrero', 'Hidalgo', 'Jalisco','Estado de Mexico', 'Michoacán de Ocampo',
           'Morelos', 'Nayarit', 'Nuevo León', 'Oaxaca','Puebla',
           'Querétaro', 'Quintana Roo', 'San Luis Potosí', 'Sinaloa','Sonora',
           'Tabasco', 'Tamaulipas', 'Tlaxcala', 'Veracruz de Ignacio de la Llave','Yucatán',
           'Zacatecas'
           ]

estados = pd.DataFrame(
    data=estados, 
    columns=['Estado'])

########################################


#Analisys de Silueta datos 2020
"""
K = range(2,13)    
for k in K: 
  #Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(dataEscalda2020) 
    silhouette_avg = silhouette_score(dataEscalda2020, kmeanModel.labels_, metric='euclidean')
    print("For n_clusters =", k,
    "The average silhouette_score is :", silhouette_avg)
   
""" 

def generate_puntajeSilueta(dataframe):

    i=0   
    puntajeSilueta= []  
    dfPuntajeSilueta = pd.DataFrame(columns=['Número de grupos', 'Puntaje de coeficiente de silueta'],index=range(5))
     
    for k in [2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit_predict(dataframe)
        #km.fit_predict(dataEscalda2020)
        
        score = silhouette_score(dataframe, km.labels_, metric='euclidean')
        #score = silhouette_score(dataEscalda2020, km.labels_, metric='euclidean')
        #print("For n_clusters =", k ,'Silhouetter Score: %.4f' % score)
        puntajeSilueta.append(score)
    
    for k in [2, 3, 4, 5, 6]:
        #print("Para el numero de grupos =", k ,'Puntaje de Silueta:',  puntajeSilueta[i])
        dfPuntajeSilueta.iloc[i] = [k, puntajeSilueta[i]]
        i+=1
       
    return dfPuntajeSilueta


def generate_GraficasSilueta(dataframe, periodo):
    
    range_n_clusters = [2, 3, 4, 5, 6]
    estePeriodo = periodo
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        #fig, (ax1) = plt.subplots(1, 1)
        fig, (ax1) = plt.subplots()
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataframe) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(dataframe)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataframe, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataframe, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i ]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        #ax1.set_title("Grafica de la silueta para varios grupos")
        #ax1.set_xlabel("Valor del coficiente de la silueta")
        #ax1.set_ylabel("Grupos")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        numGrupo=n_clusters
        nombreFigura= str(estePeriodo)+'_siluetaClusters_'+ str(numGrupo)+'.jpg'
        plt.savefig(nombreFigura)
        finaliza = True
     
        return  finaliza

    

#Algoritmop KMEANS
"""
algoritmo = KMeans (n_clusters = 3 ,  init ='k-means++',  max_iter =300, n_init=10 )
algoritmo.fit(dataEscalda2020)
#Se obtiene los datos de los centroides y las etiquetas
centroides, etiquetas = algoritmo.cluster_centers_, algoritmo.labels_

muestra_prediccion= algoritmo.predict(dataEscalda2020)

for i, pred in enumerate(muestra_prediccion):
    print("Muestra", i, "se encuentra en el clúster:", pred)
    
""" 


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "0rem",
    "padding": "2rem 1rem",
    
}

TABLE_STYLE = {
        "margin-left": "5rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "overflow": "auto",
}

TABLE_STYLE_SILUETA = {
        'textAlign': 'center',
        "margin-left": "5rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "overflow": "auto",
}

TABLE_AGRUPAMIENTO_STYLE = {
        'textAlign': 'center',
        "margin-left": "1rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "overflow": "auto",
}
    
GRAFICA_STYLE = {
        "margin-left": "8rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "overflow": "scroll",
}

SILUETA_STYLE = {
        "margin-left": "6rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "overflow": "scroll",
}

AGRUPAMIENTO_STYLE = {
        "margin-left": "5rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "overflow": "scroll",
}    

DROP_STYLE = {
    
        "overflow": "scroll",
}  

def generate_table(dataframe, max_rows=32):
     dataSet2012 =  html.Div( children=[    
        html.H3('Conjunto de datos del periodo 2011 - 2015 ', style={'textAlign': 'center'}), 
        html.Hr(),
        html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in dataframe.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ])
        ])
        ], style  = TABLE_STYLE 
        )
     return dataSet2012 

def generate_table2023(dataframe):
     dataSet2021 =  html.Div( children=[    
        html.H3('Factores que influyen con la deserción escolar en México', style={'textAlign': 'center'}), 
        html.H4('Porcentaje de la población 2023', style={'textAlign': 'center'}), 
        html.Hr(),
        dbc.Table.from_dataframe(dataframe, striped=True, bordered=True, hover=True, style={'textAlign': 'center'}),
        html.Hr(),
        html.H6('[1] Fuente STATISTA, (2023). Obtenido de : https://es.statista.com/estadisticas/575948/numero-de-personas-en-mexico-por-entidad-federativa/', style={'textAlign': 'left'}), 
        html.H6('[2] Fuente INEGI, (2023). Obtenido de : https://www.inegi.org.mx/temas', style={'textAlign': 'left'}), 
        html.H6('[3] Fuente CONEVAL, (2023). Obtenido de :https://www.coneval.org.mx/Medicion', style={'textAlign': 'left'}), 
        html.H6('[4] Fuente CONAPO, (2023). Obtenido de :https://www.gob.mx/conapo', style={'textAlign': 'left'}), 
         
        
            ], style  = TABLE_STYLE 
        )
     return dataSet2021 

def generate_grafica2023(nombreEstado):
    Estado = nombreEstado
    dataframeEstado =  pd.DataFrame(conexDB.selectEstado2023(Estado))
    factoresEstado=('Desempleo','Pobreza','Alcoholismo', 'Drogadicción','Alguna Relación con Pandillas',
                    'Depresion','Ansiedad','Divorcios','Embarazos Tempranos')
                    
    listaValoresEstado= list(dataframeEstado.values.tolist())
    datosEstado = []
    #u = listaValoresEstado.index.values.tolist()
    for lista1 in listaValoresEstado:
        for numero in lista1:
            datosEstado.append(numero)
            
    fig = px.bar(x=factoresEstado, y= datosEstado, 
                 title="Porcentaje de la población del período 2023", labels=dict(x="Factores", y="Porcentaje %"))         

    graficaEstado2023 =  html.Div( children=[ 
        html.H3(f'''Factores que influyen con la deserción escolar en {Estado} ''', style={'textAlign': 'center'}), 
        dcc.Graph( figure=fig) 
         ], style  = GRAFICA_STYLE 
        )
    return graficaEstado2023     


def metodoCodo2021 ():
    inercia = []
    dfPuntajeInercia = pd.DataFrame(columns=['Número de grupos', 'Puntaje'],index=range(20))
    
    for i in range (1,20):
        algoritmoKM = KMeans (n_clusters = i , init ='k-means++', max_iter =300, n_init=10 )
        algoritmoKM.fit(dataEscalda2023)
        inercia.append(algoritmoKM.inertia_)
   
    i=0
    for k in range(1,20) :
        #print("Para el numero de grupos =", k ,'Puntaje de Silueta:',  puntajeSilueta[i])
        dfPuntajeInercia.iloc[i] = [k, inercia[i]]
        i+=1     
   
    fig = px.line(x=list(range(1,20)), y= inercia , 
                 title="Método del codo período 2019 - 2021")
    
    graficaCodo2021 =  html.Div( children=[ 
        html.H3('Método del codo período 2019 - 2021', style={'textAlign': 'center'}), 
        html.Br(),
        dcc.Graph(figure=fig),
        #dcc.Loading(children=[dcc.Graph( figure=fig )], color="#119DFF", type="dot", fullscreen=True,),

        html.Br(),
        html.H5('Tabla de puntaje del método del codo', style={'textAlign': 'center'}), 
        dbc.Table.from_dataframe(dfPuntajeInercia, striped=True, bordered=True, hover=True, style= TABLE_STYLE_SILUETA),
        ], style  = GRAFICA_STYLE 
        )
    return  graficaCodo2021 


def metodoSilueta2021 (dataEscalda2020):
    
    dfPuntajeSilueta2020 = generate_puntajeSilueta(dataEscalda2020)
    generate_GraficasSilueta(dataEscalda2020, "2020")
    
    myImage1 = Image.open("2020_siluetaClusters_2.jpg")
    myImage2 = Image.open("2020_siluetaClusters_3.jpg")
    myImage3 = Image.open("2020_siluetaClusters_4.jpg")
    myImage4 = Image.open("2020_siluetaClusters_5.jpg")
    myImage5 = Image.open("2020_siluetaClusters_6.jpg")
    #img = data.chelsea()
    fig1 = px.imshow(myImage1, title="Número de grupos: 2",  labels=dict(x="Valor del coficiente de la silueta", y="Grupos")) 
    fig2 = px.imshow(myImage2, title="Número de grupos: 3",  labels=dict(x="Valor del coficiente de la silueta", y="Grupos")) 
    fig3 = px.imshow(myImage3, title="Número de grupos: 4",  labels=dict(x="Valor del coficiente de la silueta", y="Grupos")) 
    fig4 = px.imshow(myImage4, title="Número de grupos: 5",  labels=dict(x="Valor del coficiente de la silueta", y="Grupos")) 
    fig5 = px.imshow(myImage5, title="Número de grupos: 6",  labels=dict(x="Valor del coficiente de la silueta", y="Grupos")) 
   # fig1.update_layout(dragmode="drawrect", yaxis_visible=False, yaxis_showticklabels=False, xaxis_visible=False, xaxis_showticklabels=False )
    fig1.update_layout(dragmode="drawrect", yaxis_showticklabels=False, xaxis_showticklabels=False,  hovermode=False)
    fig2.update_layout(dragmode="drawrect", yaxis_showticklabels=False, xaxis_showticklabels=False,  hovermode=False)
    fig3.update_layout(dragmode="drawrect", yaxis_showticklabels=False, xaxis_showticklabels=False,  hovermode=False)
    fig4.update_layout(dragmode="drawrect", yaxis_showticklabels=False, xaxis_showticklabels=False,  hovermode=False)
    fig5.update_layout(dragmode="drawrect", yaxis_showticklabels=False, xaxis_showticklabels=False,  hovermode=False)
    config = {
    "modeBarButtonsToAdd": [    
        "drawopenpath",
                           ], 
            }
    graficaSilueta2021 =  html.Div( children=[ 
        html.H3('Análisis de la silueta periodo 2019 - 2021', style={'textAlign': 'center'}), 
        html.Hr(),
        dbc.Table.from_dataframe(dfPuntajeSilueta2020, striped=True, bordered=True, hover=True, style= TABLE_STYLE_SILUETA),
        html.Hr(),
        html.Br(),
        html.H4('Gráficas del análisis de la silueta', style={'textAlign': 'center'}), 
        html.Br(),
        #html.H5('Numero de grupos: 2', style={'textAlign': 'left'}), 
        #dcc.Graph(figure=fig1,config=config), 
        dcc.Graph( id="my-graph", figure=fig1 ),
        #html.H5('Numero de grupos: 3', style={'textAlign': 'left'}), 
        dcc.Graph(figure=fig2), 
        #html.H5('Numero de grupos: 4', style={'textAlign': 'left'}), 
        dcc.Graph(figure=fig3), 
        #html.H5('Numero de grupos: 5', style={'textAlign': 'left'}), 
        dcc.Graph(figure=fig4), 
        #html.H5('Numero de grupos: 6', style={'textAlign': 'left'}), 
        dcc.Graph(figure=fig5), 
         ], style  = SILUETA_STYLE 
        )
    return  graficaSilueta2021 



def agrupamientoKMEANS2023 ():

   NumGrupo2023=[] 

   ## Se aplica el algoritmo de clustering ##
   #Se define el algoritmo junto con el valor de K
   algoritmoKmeans = KMeans( init='k-means++',  n_clusters=3, n_init=10,   max_iter=300, random_state=42 )
     
   #Se entrena el algoritmo
   algoritmoKmeans.fit(dataKmedias2023) 
     
   #Utilicemos los datos de muestras y verifiquemos en que cluster se encuentran
   muestra_prediccion = algoritmoKmeans.predict(dataKmedias2023)
     
   #score = metrics.adjusted_rand_score()
     
   #Se obtiene los datos de los centroides y las etiquetas
   centroides, etiquetas = algoritmoKmeans.cluster_centers_, algoritmoKmeans.labels_
   
   #Se asigna el numero de grupo en un array proveniente de la variable etiquetas
   for e in range(len(etiquetas)):
        NumGrupo2023.append(etiquetas[e] + 1)
    
   #Se crea un dataframe con el numero de grupo (etiquetas)
   NumGrupo2023_df = pd.DataFrame(data=NumGrupo2023, columns=['Grupo'])  
     
   ### GRAFICAR LOS DATOS JUNTO A LOS RESULTADOS ###
   # Se aplica la reducción de dimensionalidad al conjunto de datos
   modelo_pca = PCA(n_components = 2)
   modelo_pca.fit(dataKmedias2023)
   pca = modelo_pca.transform(dataKmedias2023) 
     
   #Se crea dataframe para resultados de pca (reduccion del conjunto de datos)
   pca_df = pd.DataFrame( data=pca, columns=['PCA_X', 'PCA_Y']) 
   #se crea dataframe concadenando los estados, pca y el numero de grupo (etiqueta)
   pca_finalDF = pd.concat([estados, pca_df , NumGrupo2023_df ], axis=1)
   #se crea dataframe concadenando los estados y el numero de grupo (etiqueta) para la tabla
   pca_finalDFtable = pd.concat([estados, NumGrupo2023_df ], axis=1)
   

   #Se aplicar la reducción de dimsensionalidad a los centroides
   #modelo_pca = PCA(n_components = 2)
   #modelo_pca.fit(centroides)
   centroides_pca = modelo_pca.transform(centroides) 

   #Se crea dataframe para resultados de pca centroides 
   centroides_pca_df = pd.DataFrame(data= centroides_pca, columns=['Puntaje_X','Puntaje_Y'])
   #Se agrega columna con valores  
   centroides_pca_df['Centroide']= ['Centroide 1', 'Centroide 2', 'Centroide 3']
   #Se ordenan las columnas
   centroides_pca_df = centroides_pca_df.reindex(columns=['Centroide','Puntaje_X','Puntaje_Y'])
 
 
   # Se define los colores de cada clúster y simbolos de centroides
   #colores = ['blue', 'brown', 'purple']
   colores = ['Grupo 1', 'Grupo 2', 'Grupo 3']
   symbols = ['circle-dot']
   colorRed =['black'] 
   
   #Se asignan los colores a cada clústeres
   colores_cluster = [colores[etiquetas[i]] for i in range(len(pca))]
   
   #Se grafican los dataFrames generados
   #fig1 = px.scatter(pca, pca[:, 0], pca[:, 1], color=colores_cluster)
   fig1 = px.scatter(pca_finalDF, x="PCA_X" , y="PCA_Y", color=colores_cluster, hover_name="Estado")
   
   fig2 = px.scatter(centroides_pca , centroides_pca[:, 0], centroides_pca[:, 1], 
                     color_discrete_sequence= colorRed , symbol_sequence = symbols)
   fig2.update_traces(marker={'size': 12})
   

   #Se guadan los datos en una variable para que sea fácil escribir el código
   xvector = modelo_pca.components_[0] * max(pca[:,0])
   yvector = modelo_pca.components_[1] * max(pca[:,1])
   columnas = dataKmedias2023.columns
   
   factores_pca_df= pd.DataFrame({'Factores':columnas, 'PCA_X':xvector, 'PCA_Y':yvector})

   
   fig = go.Figure(data = fig1.data + fig2.data)
   
   fig.update_layout(
    title="",
    xaxis_title="W1-Componentes principales, eje X ",
    yaxis_title="w2-Componentes principales, eje Y ",
    legend_title="",
    )

   """
   #Se grafican los vectores 
   i=0
   for i in range(len(columnas)):
       fig.add_trace(go.Scatter(
       x=[0, xvector[i]],
       y=[0, yvector[i]],
       mode="lines+markers+text",
       name=columnas[i],
       text=["", ""],
       textposition="top center"
       ))
   """
   agrupamientoKMedias2021 =  html.Div( 
                                children=[ 
                                 
                                            
        html.H3('Agrupamiento K medias período 2023', style={'textAlign': 'center'}), 
        dcc.Graph(  figure=fig),
    
        
         #dbc.Spinner(children=[ dcc.Graph(id="loading-output", figure=fig)], size="lg", color="primary", type="border", fullscreen=True,),
        
        
        html.Br(),
        html.H4('Tabla de Estados con número de agrupación', style={'textAlign': 'center'}), 
        html.Hr(),
        html.Br(),
        dbc.Table.from_dataframe(pca_finalDFtable.sort_values(by="Grupo"), striped=True, bordered=True, hover=True, style= TABLE_AGRUPAMIENTO_STYLE),
        
        
      # html.Br(),
      # html.H5('Tabla de puntaje factores', style={'textAlign': 'center'}), 
      # html.Hr(),
      # html.Br(),
      # dbc.Table.from_dataframe(factores_pca_df, striped=True, bordered=True, hover=True, style= TABLE_AGRUPAMIENTO_STYLE),
        
      # html.Br(),
      # html.H5('Tabla de puntaje de centroides', style={'textAlign': 'center'}), 
      # html.Hr(),
      # html.Br(),
      # dbc.Table.from_dataframe(centroides_pca_df, striped=True, bordered=True, hover=True, style= TABLE_AGRUPAMIENTO_STYLE),
        
        ], style  = GRAFICA_STYLE,     
         
        )#cierre div
   return  agrupamientoKMedias2021 

"""
def mezclaGaussiana ():
   mezclaGaussiana =  html.Div( children=[ 
        html.H3('Mezcla gausianna periodo 2019 - 2021', style={'textAlign': 'center'}), 
         dcc.Graph() 
         ], style  = GRAFICA_STYLE 
        )
   return  mezclaGaussiana 
"""

def generate_Agrupamiento (): 
    graficaAgrupamiento = html.Div( children=[ 
        html.H3('Gráfica de agrupamiento periodo 2019 - 2021', style={'textAlign': 'center'}), 
        html.Br(),
        dcc.Graph(),
        ], style  = AGRUPAMIENTO_STYLE 
        )
    return  graficaAgrupamiento   

presentacion = html.Div( children=[
    html.H1('Universidad Politécnica de Puebla', style={"margin-left": "6rem",'textAlign': 'rigth'}), 
    html.Br(),
    html.Br(),
   #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
    html.Img(src=app.get_asset_url('logo.png'), style={ 'textAlign': 'center','height':'80%', 'width':'80%'}),
    #html.H2('Tema de investigación: ', style={'textAlign': 'lef'}) ,    
    #html.H3('Ciencia de datos para analizar el abandono escolar en el nivel ', style={'textAlign': 'center'}),
    #html.H3( ' medio superior en México', style={'textAlign': 'center'}), 
    #html.Br(),
    #html.H4( 'Por:', style={'textAlign': 'Left'}), 
    html.H6( 'I.S.C. Carlos Antonio García Pérez', style={'textAlign': 'right'}),
    #html.Br(),
    #html.Br(),
    html.H6( 'Director: Dr. Javier Caldera Miguel.', style={'textAlign': 'right'}), 
    html.H6( 'Co-Director: Dr. Jorge de la Calleja Mora', style={'textAlign': 'right'}), 
    ],
    
    style=CONTENT_STYLE 
    ) #fin de presentacion 

   
sidebar = html.Div(
    children=[
        html.H3("UPPue", className="display-4"),
        html.Hr(),
        html.H4(
            "Menú", className="lead"
        ),
        
        dbc.Nav(
            [
                dbc.NavLink("Inicio", href="/", active="exact"),
            #   dbc.NavLink("Conjunto Datos 2020", href="/page-1", active="exact"),
                html.Details([
                    html.Summary("Conjuntos de datos"), 
                    dbc.NavLink("Variables periodo 2023 ", href="/page-1", active="exact"),
                    
                    ],   ), #termina Details 
               
                  html.Details([
                    html.Summary("Gráfica por Estados"),   
                    dbc.NavLink("Aguascalientes", href="/page-2", active="exact"),
                    dbc.NavLink("Baja California", href="/page-3", active="exact"),
                    dbc.NavLink("Baja California Sur", href="/page-4", active="exact"),
                    dbc.NavLink("Campeche", href="/page-5", active="exact"),
                    dbc.NavLink("Coahuila de Zaragoza", href="/page-6", active="exact"),
                    dbc.NavLink("Colima", href="/page-7", active="exact"),
                    dbc.NavLink("Chiapas", href="/page-8", active="exact"),
                    dbc.NavLink("Chihuahua", href="/page-9", active="exact"),
                    dbc.NavLink("Ciudad de México", href="/page-10", active="exact"),
                    dbc.NavLink("Durango", href="/page-11", active="exact"),
                    dbc.NavLink("Guanajuato", href="/page-12", active="exact"),
                    dbc.NavLink("Guerrero", href="/page-13", active="exact"),  
                    dbc.NavLink("Hidalgo", href="/page-14", active="exact"), 
                    dbc.NavLink("Jalisco", href="/page-15", active="exact"),
                    dbc.NavLink("Estado de Mexico", href="/page-16", active="exact"),
                    dbc.NavLink("Michoacán de Ocampo", href="/page-17", active="exact"),
                    dbc.NavLink("Morelos", href="/page-18", active="exact"),  
                    dbc.NavLink("Nayarit", href="/page-19", active="exact"),
                    dbc.NavLink("Nuevo León", href="/page-20", active="exact"),
                    dbc.NavLink("Oaxaca", href="/page-21", active="exact"),
                    dbc.NavLink("Puebla", href="/page-22", active="exact"),  
                    dbc.NavLink("Querétaro", href="/page-23", active="exact"), 
                    dbc.NavLink("Quintana Roo", href="/page-24", active="exact"),
                    dbc.NavLink("San Luis Potosi", href="/page-25", active="exact"),
                    dbc.NavLink("Sinaloa", href="/page-26", active="exact"),  
                    dbc.NavLink("Sonora", href="/page-27", active="exact"),
                    dbc.NavLink("Tabasco", href="/page-28", active="exact"),
                    dbc.NavLink("Tamaulipas", href="/page-29", active="exact"),  
                    dbc.NavLink("Tlaxcala", href="/page-30", active="exact"), 
                    dbc.NavLink("Veracruz de Ignacio de la Llave", href="/page-31", active="exact"),
                    dbc.NavLink("Yucatán", href="/page-32", active="exact"),
                    dbc.NavLink("Zacatecas", href="/page-33", active="exact"), 
                    ],  ), #termina Details
                   
                  html.Details([
                    html.Summary("Agrupamiento"),   
                    dbc.NavLink(" K medias", href="/page-34", active="exact"),
                 
                     ],   ), #termina Details
              
                   # html.Details([
                   # html.Summary("Mezcla Gaussiana (GMM)"),    
                   # dbc.NavLink("Mezcla Gaussiana (GMM)", href="/page-106", active="exact"), 
                   # ],  ), #termina Details
               
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE) 
            
#app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
app.layout = html.Div([dcc.Location(id="url"), sidebar,  
                       html.Div( [  dcc.Loading( children=[ html.Div([  html.Div(  content  ) ]) ],color="#119DFF", type="dot", fullscreen=True,)])  ])  

"""
@app.callback(
    Output("download_Excel2012", "data"),
    #Input("btn_Excel2012", "n_clicks"),
    #Output(component_id='download_Excel2012',  "data"),  
    Input(component_id='btn_Excel2012', component_property='n_clicks'),
  
   prevent_initial_call=True,
   suppress_callback_exceptions=True,
)
def func(n_clicks):
    return dcc.send_data_frame(dataframe2012.to_excel, "datos2012.xlsx", sheet_name="datos2012")

"""

@app.callback(
Output("page-content", "children"), 
[Input("url", "pathname")
])

def render_page_content(pathname):
    if pathname == "/":
        return presentacion
    elif pathname == "/page-1":
         return generate_table2023(dataframe2023)
    elif pathname == "/page-2":
         return generate_grafica2023("Aguascalientes")
    elif pathname == "/page-3":
         return generate_grafica2023("Baja California") 
    elif pathname == "/page-4":
         return generate_grafica2023("Baja California Sur") 
    elif pathname == "/page-5":
         return generate_grafica2023("Campeche")     
    elif pathname == "/page-6":
         return generate_grafica2023("Coahuila de Zaragoza")  
    elif pathname == "/page-7":
         return generate_grafica2023("Colima")  
    elif pathname == "/page-8":
         return generate_grafica2023("Chiapas")  
    elif pathname == "/page-9":
         return generate_grafica2023("Chihuahua") 
    elif pathname == "/page-10":
         return generate_grafica2023("Ciudad de Mexico") 
    elif pathname == "/page-11":
         return generate_grafica2023("Durango") 
    elif pathname == "/page-12":
         return generate_grafica2023("Guanajuato") 
    elif pathname == "/page-13":
         return generate_grafica2023("Guerrero") 
    elif pathname == "/page-14":
         return generate_grafica2023("Hidalgo") 
    elif pathname == "/page-15":
         return generate_grafica2023("Jalisco") 
    elif pathname == "/page-16":
         return generate_grafica2023("Estado de Mexico") 
    elif pathname == "/page-17":
         return generate_grafica2023("Michoacan de Ocampo") 
    elif pathname == "/page-18":
         return generate_grafica2023("Morelos") 
    elif pathname == "/page-19":
         return generate_grafica2023("Nayarit")     
    elif pathname == "/page-20":
         return generate_grafica2023("Nuevo Leon")
    elif pathname == "/page-21":
         return generate_grafica2023("Oaxaca")
    elif pathname == "/page-22":
         return generate_grafica2023("Puebla")
    elif pathname == "/page-23":
         return generate_grafica2023("Queretaro") 
    elif pathname == "/page-24":
         return generate_grafica2023("Quintana Roo") 
    elif pathname == "/page-25":
         return generate_grafica2023("San Luis Potosi")      
    elif pathname == "/page-26":
         return generate_grafica2023("Sinaloa")     
    elif pathname == "/page-27":
         return generate_grafica2023("Sonora")     
    elif pathname == "/page-28":
         return generate_grafica2023("Tabasco")  
    elif pathname == "/page-29":
         return generate_grafica2023("Tamaulipas")  
    elif pathname == "/page-30":
         return generate_grafica2023("Tlaxcala")
    elif pathname == "/page-31":
         return generate_grafica2023("Veracruz de Ignacio de la Llave")
    elif pathname == "/page-32":
         return generate_grafica2023("Yucatan")
    elif pathname == "/page-33":
         return generate_grafica2023("Zacatecas")
    elif pathname == "/page-34":
         return agrupamientoKMEANS2023()
        
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
  
     #print(dataframe2020)
     app.run_server(debug=True, port=8051) # Esblece puerto 
     #app.run_server(debug=True)  