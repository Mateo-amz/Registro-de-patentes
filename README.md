# Registro de patentes

Final project for the Building AI course

## Summary

Se pretende lograr un registro automático de ingreso de vehículos mediante la lectura de patentes (placas) utilizando IA. El registro deberá escribirse en una planilla de cálculo o archivo .csv.
El proyecto podrá ampliarse para incorporar características como registro de características de vehículos o cantidad de ocupantes visualizado.

(The objective of this project is an automatic registration of vehicle license plates, using AI for license plate and vehicle recognition. The results will be saved in a .csv file. This project can be updated to add some other feature, such as the number of vehicles.)


## Background

Este proyecto pretende resolver los siguientes puntos:
* Registro automatico de vehículos que ingresan a un sector, reduciendo la carga del personal de seguridad.
* Mejorar la seguridad y reducir el estrés laboral al disminuir la carga de trabajo del personal.

(This proyect attempt to solve:
* Automatic vehicle registration.
* Improve safety and reduce work stress.)

## How is it used?

El sistema cuenta como mínimo con una computadora que analiza imágenes tomadas con una cámara de seguridad convencional.
El procesamiento debe realizarse on-site.

El código cuenta de dos funciones: main.py y util.py. Además, necesita de un video (mp4) para ser analizado y un modelo entrenado para reconocer las patentes correspondientes al video. Se incluye un pequeño modelo entrenado para patentes Argentinas, pero este modelo puede ser optimizado mediante un dataset más grande. Además, se requiere tener descargado el módulo "sort", el cual debe estár en una carpeta en la misma ubicación que las demás funciones y bajo el nombre "sort" (https://github.com/abewley/sort). Información más completa puede enocntrarse en las fuentes al final del README.
El funcionamiento del programa se realiza ejecutando la función main.py (en mi caso, utilizando Visual Studio)


(This system uses a computer to read images from a camera. All image processing will be on site.
The code has two functions: main.py and util.py. In addition, it needs a video (mp4) to be analyzed and a trained model to recognize the patents corresponding to the video. A small model trained for Argentine patents is included, but this model can be optimized using a larger dataset.
The operation of the program is done by executing the main.py function (in my case, using Visual Studio)
In addition, it is required to have the "sort" module downloaded, which must be in a folder in the same location as the other functions and under the name "sort" (https://github.com/abewley/sort). More complete information can be found in the sources at the end of the README.)

## Data sources and AI methods

Los datos para entrenamiento serán obtenidos mediante cámaras convencionales, al inicio se usarán imágenes capturadas a partir de un celular, para luego comenzar a utilizar video en vivo mediante una cámara de seguridad de 2 MP o 4 MP.
Se considera que este tipo de IA se encuentra desarrollada en otros casos, por lo tanto se realizará una busqueda para subir a hombros de estos autores más avanzados en la temática, considerando siempre el correcto uso de copyright.

(The training data will be obtained from cameras. Many authors on the web contribute very useful information for this project, this will be used with the corresponding copyright.)

## Challenges

El requerimiento de procesamiento on-site podría ser un desafío, al requerir un sistema optimizado capaz de funcionar en tiempo real en una computadora estándar.

(In situ processing could be a challenge, because a computer will be needed to do all the processing in real time.)

## What next?

A futuro, se propone incorporar más posibilidades bajo la misma plataforma, como por ejemplo, el registro de características de los vehículos o un contador.

(In the future, it is possible to add some functions, such as car feature recognition or car counting.)

## Acknowledgments

Las fuentes de inspiración para este proyecto son la necesidad de dicho dispositivo en la empresa donde trabajo, la curiosidad por las nuevas posibilidades que brinda la IA, y los conocimientos adquiridos tanto en el cursado de Building AI como en el curso denominado "IA aplicado a la ingenieria usando Python" dictado en la Universidad Nacional de Misiones por el Dr. Matías Krujoski e ing. Axel Skrauba.

(The sources of inspiration for this project are the need for said device in the company where I work, the curiosity about the new possibilities that AI offers, and the knowledge acquired both in the Building AI course and in the course called "AI applied to engineering using Python" taught at the National University of Misiones by Dr. Matías Krujoski and ing. Axel Skrauba.)

Además, de lo anterior, las principales fuentes de información para la realización de este proyecto son:
* github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8
* ultralytics.com






