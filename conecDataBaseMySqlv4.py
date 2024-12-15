# -*- coding: utf-8 -*-
"""
Created on Tue May 31 18:35:23 2022

@author: PC-GARCIA
"""

import pandas as pd
import pymysql

class classConecionDB:


    def __init__(self):
        print("CONSTRUCTOR ")	 

    def obtenerDataSet2023 (self):
        try:
         conexion = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='conjunto2023')
         
         cursor = conexion.cursor()
         #sql = "select * from factores2023"
         sql = ("select  id, nombreEstado as \'Estado\' , numHabitantes as \'Número de Habitantes[1]\' , desempleo as \'Desempleo[2] %\' , pobreza as \'Pobreza[2] %\'," 
		       "alcoholismo as \'Alcoholismo[2] %\', drogadiccion as \'Drogadicción[2] %\', relacionPandillas as \'Alguna Relación con Pandillas[2] %\',"
               "depresion as \'Depresión[2] %\',  ansiedad as \'Ansiedad[2] %\', divorcios as \'Divorcios[2] %\' , embarazosTempranos as \'Embarazos Tempranos[4] %\'"
               "From  factores2023")
            
         cursor.execute(sql)
         result = cursor.fetchall()
         print(result)
         df = pd.read_sql(sql,conexion )
         return df
         print("Conectado ")	
        except (pymysql.err.OperationalError, pymysql.err.InternalError) as e:
         print("Ocurrió un error al conectar: ", e)
        except pymysql.Error as error:
         print("Error en pymysql %d: %s" %(error.args[0], error.args[1]))   
        finally :
         if conexion:   
            conexion.close()
       
  
    def selectEstado2023(self, varEstado):
        Estado = varEstado
        try:
          conexion = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='conjunto2023')
          cursor = conexion.cursor()
          if Estado=="Aguascalientes":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
        	    depresion, ansiedad, divorcios, embarazosTempranos \
        	    FROM factores2023 where nombreEstado =\"Aguascalientes\"";     
          elif Estado=="Baja California":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
        	    depresion, ansiedad, divorcios, embarazosTempranos \
        	    FROM factores2023 where nombreEstado =\"Baja California\""; 
          elif Estado=="Baja California Sur":
            sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
         	    depresion, ansiedad, divorcios, embarazosTempranos \
         	    FROM factores2023 where nombreEstado =\"Baja California Sur\""; 
          elif Estado=="Campeche":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
        	    depresion, ansiedad, divorcios, embarazosTempranos \
        	    FROM factores2023 where nombreEstado =\"Campeche\""; 
          elif Estado=="Coahuila de Zaragoza":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
           	    depresion, ansiedad, divorcios, embarazosTempranos \
           	    FROM factores2023 where nombreEstado =\"Coahuila de Zaragoza\""; 
          elif Estado=="Colima":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
        	    depresion, ansiedad, divorcios, embarazosTempranos \
        	    FROM factores2023 where nombreEstado =\"Colima\""; 
          elif Estado=="Chiapas":
            sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
         	    depresion, ansiedad, divorcios, embarazosTempranos \
         	    FROM factores2023 where nombreEstado =\"Chiapas\""; 
          elif Estado=="Chihuahua":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
        	    depresion, ansiedad, divorcios, embarazosTempranos \
        	    FROM factores2023 where nombreEstado =\"Chihuahua\""; 
          elif Estado=="Ciudad de Mexico":
           sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
        	    depresion, ansiedad, divorcios, embarazosTempranos \
        	    FROM factores2023 where nombreEstado =\"Ciudad de Mexico\""; 
          elif Estado=="Durango":
            sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
         	    depresion, ansiedad, divorcios, embarazosTempranos \
         	    FROM factores2023 where nombreEstado =\"Durango\""; 
          elif Estado=="Guanajuato":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
           	    depresion, ansiedad, divorcios, embarazosTempranos \
           	    FROM factores2023 where nombreEstado =\"Guanajuato\""; 
          elif Estado=="Guerrero":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
           	    depresion, ansiedad, divorcios, embarazosTempranos \
           	    FROM factores2023 where nombreEstado =\"Guerrero\""; 
          elif Estado=="Hidalgo":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
           	    depresion, ansiedad, divorcios, embarazosTempranos \
           	    FROM factores2023 where nombreEstado =\"Hidalgo\"";      
          elif Estado=="Jalisco":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
           	    depresion, ansiedad, divorcios, embarazosTempranos \
           	    FROM factores2023 where nombreEstado =\"Jalisco\"";  
          elif Estado=="Estado de Mexico":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
                depresion, ansiedad, divorcios, embarazosTempranos \
                FROM factores2023 where nombreEstado =\"Estado de Mexico\"";  
          elif Estado=="Michoacan de Ocampo":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Michoacan de Ocampo\"";        
          elif Estado=="Morelos":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
         	    depresion, ansiedad, divorcios, embarazosTempranos \
         	    FROM factores2023 where nombreEstado =\"Morelos\"";  
          elif Estado=="Nayarit":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
           	    depresion, ansiedad, divorcios, embarazosTempranos \
           	    FROM factores2023 where nombreEstado =\"Nayarit\""; 
          elif Estado=="Nuevo Leon":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
            	depresion, ansiedad, divorcios, embarazosTempranos \
            	FROM factores2023 where nombreEstado =\"Nuevo Leon\"";  
          elif Estado=="Oaxaca":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
            	depresion, ansiedad, divorcios, embarazosTempranos \
            	FROM factores2023 where nombreEstado =\"Oaxaca\"";        
          elif Estado=="Puebla":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
            	depresion, ansiedad, divorcios, embarazosTempranos \
            	FROM factores2023 where nombreEstado =\"Puebla\"";        
          elif Estado=="Queretaro":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
              	depresion, ansiedad, divorcios, embarazosTempranos \
               	FROM factores2023 where nombreEstado =\"Queretaro\"";    
          elif Estado=="Quintana Roo":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Quintana Roo\"";     
          elif Estado=="San Luis Potosi":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"San Luis Potosi\""; 
          elif Estado=="Sinaloa":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Sinaloa\"";         
          elif Estado=="Sonora":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Sonora\"";      
          elif Estado=="Tabasco":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Tabasco\""; 
          elif Estado=="Tamaulipas":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Tamaulipas\""; 
          elif Estado=="Tlaxcala":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Tlaxcala\""; 
          elif Estado=="Veracruz de Ignacio de la Llave":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
               depresion, ansiedad, divorcios, embarazosTempranos \
               FROM factores2023 where nombreEstado =\"Veracruz de Ignacio de la Llave\"";              
          elif Estado=="Yucatan":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
                depresion, ansiedad, divorcios, embarazosTempranos \
                FROM factores2023 where nombreEstado =\"Yucatan\"";       
          elif Estado=="Zacatecas":
              sql = "SELECT desempleo, pobreza, alcoholismo, drogadiccion , relacionPandillas,\
                depresion, ansiedad, divorcios, embarazosTempranos \
                FROM factores2023 where nombreEstado =\"Zacatecas\"";   
          else:
              return False
          cursor.execute(sql)
             #result = cursor.fetchall()
             #print(result)  
          df = pd.read_sql(sql,conexion)
                  #  df = pd.read_sql(result )
          return df
          print("Conectado ")	
        except (pymysql.err.OperationalError, pymysql.err.InternalError) as e:
          print("Ocurrió un error al conectar: ", e)
        except pymysql.Error as error:
         print("Error en pymysql %d: %s" %(error.args[0], error.args[1]))   
        finally :
         if conexion:   
            conexion.close()        
