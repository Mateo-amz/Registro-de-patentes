from ultralytics import YOLO #YOLO!
import cv2  #Para trabajar con videos
import numpy as np

from util import get_car, read_license_plate, write_csv, write_csv2

""" Pinta lineas con el formato
x1,y,1     x2,y1
x1,y2      x2,y2 """
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=1, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    color2=(0,0,255)
    thickness2=2
    cv2.line(img, (x1, y1), (x1, y2), color2, thickness2)  #-- top-left
    cv2.line(img, (x1, y1), (x2 , y1), color2, thickness2)

    cv2.line(img, (x1, y2), (x2, y2), color2, thickness2)  #-- bottom-left
    cv2.line(img, (x2, y2), (x2, y1), color2, thickness2)
    
    return img



#Bliblioteca para tracking de vehículos
from sort.sort import *
mot_tracker=Sort()

#Se usan dos modelos de YOLO, uno para detectar autos y otro para patentes
coco_model=YOLO('yolov8n.pt')
#El siguiente modelo se debe ubicar en models/license_plate_detector.pt
license_plate_detector=YOLO('B:/Mega/Building AI/Final/v5.0_Prueba_modelo_arg/best.pt')

#Video que se utiliza:
#!Poner ruta completa (ej: B:/Mega/Building AI/Final/v1.0/sample.mp4)
#cap = cv2.VideoCapture("B:/Mega/Building AI/Final/v1.0/sample.mp4")
cap = cv2.VideoCapture("B:/Mega/Building AI/Final/v5.3_csv3/video2.mp4")
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("rtsp://admin:ebycam555@192.168.10.15:554/Streaming/channels/1/")

#Lista de categorías en YOLO que se consideran vehículos:
#2 (auto) ; 3 (moto) ; 5 (colectivo) ; 7 (camión)
vehicles =[2, 3, 5, 7]

#Lectura de frames
#autos_actuales={}   #Guarda la lista de autos en analisis actualmente 
frames_espera = 75 #Cuando un auto (car_id) se deja de ver en esta cantidad de frames, se actualiza el csv
autos= {} #Uso para cargar solo autos y patentes
results = {}
results_max_score = {}
frame_nmr=-1
ret=True
new=0   #Determina si ya se inicializó el csv

while ret:
    
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:

        

        if frame_nmr > 5000:
            break
        #frame = cv2.resize(frame, (1280, 720))
        results[frame_nmr] = {} #Diccionario donde se guardan los resultados que después se pasan a la función write_csv
        detections=coco_model(frame)[0]
        detections_=[]  #Para guardar las detecciónes validas (las de vehículos)
        #Información de detecciónes:
        for detection in detections.boxes.data.tolist():
            #cada deteccupib tuebe -> Ubicación (x,y), score (certeza) y categoría 
            x1, y1, x2, y2, score, class_id = detection
            #Solo me quedo con las deteccións aceptadas en la variable vehicles:
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score]) #guardo solo las detecciónes de vehículos

        #TRACKING DE VEHÍCULOS en movimiento (se usa SORT)
        #Se añade información de rastreo a la matriz de los autos
        #cv2.imshow("a",frame)
        #cv2.waitKey(1)
        for i in range(len(detections_)):
            draw_border(frame, (int(detections_[i][0]), int(detections_[i][1])), (int(detections_[i][2]), int(detections_[i][3])), (0, 255, 0), 25, line_length_x=200, line_length_y=200)
            #frame2 = cv2.resize(frame, (1280, 720))
            cv2.imshow("a",frame)
            cv2.waitKey(1)

        track_ids = mot_tracker.update(np.asarray(detections_))


        #DETECCIÓN DE PATENTES:
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            #Unión auto-patente:
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                #Al detectar un auto, sumo un frame a todos, pero reseteo el detectado
                # for i in autos_actuales.keys():  
                #     autos_actuales[i]["Frames"]=autos_actuales[i]["Frames"]+1
                #     if autos_actuales[i]["Frames"]==frames_espera:
                #         output_path="B:/Mega/Building AI/Final/v5.2_csv2/Autos.csv"
                #         write_csv2(autos,car_id, output_path)
                #         print("ENVIADO A CSV")                
                # autos_actuales[car_id]["Frames"]=0
                
                """try:
                    if car_id not in autos:
                    #if not autos[car_id]:
                        #autos[car_id]={car_id:""}
                        autos[car_id]=""
                        print("AGREGADO")
                except:
                    print("NO SE PUDO") """
                #Recorte de patente:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                #Procesamiento de la patente:
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                #_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 190, 255, cv2.THRESH_BINARY_INV) #(la primera variable no se usa)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 190, 255, cv2.THRESH_BINARY_INV) #(la primera variable no se usa)
                #threshold original -> 64
                #threshold original -> 160 -> maso bien
                #threshold original -> 180 -> no reconoce el OCR
                #threshold original -> 190 -> 5/7 errores (mejor)
                #threshold original -> 195 -> 6/7 errores (mejor)
                #threshold original -> 210 -> deja de reconocer el OCR
                
                #cv2.imshow("Recorte",license_plate_crop)
                cv2.imshow("Recorte",license_plate_crop_thresh)
                cv2.waitKey(1)
                
                #OCR:
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                #Se guardan los datos que luego se exportan
                if license_plate_text is not None: #Solos se guarda si se detectó patente y se leyó
                    results[frame_nmr][car_id] = {"car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                                                "license_plate": {"bbox":[x1, y1, x2, y2], 
                                                                    "text":license_plate_text, 
                                                                    "bbox_score":score, 
                                                                    "text_score":license_plate_text_score}}
                    #"autos" es un diccionario que guarda solo los ids de autos con patentes
                    try:
                        if car_id not in autos:
                            
                            autos[car_id]={"Patente": {"text":"" , "score":0}, "Frames":0}
                            autos[car_id]["Patente"]["text"]=results[frame_nmr][car_id]["license_plate"]["text"]
                            print("AGREGADO")
                            
                        else:
                            
                            if autos[car_id]["Patente"]["score"] < results[frame_nmr][car_id]["license_plate"]["text_score"]:
                                autos[car_id]["Patente"]["score"] = results[frame_nmr][car_id]["license_plate"]["text_score"]
                                autos[car_id]["Patente"]["text"]=results[frame_nmr][car_id]["license_plate"]["text"]
                            #print("PATENTE GUARDADA: ",autos[car_id]["Patente"]["score"]) 
                    except:
                        print("NO SE PUDO")
                    
                    print(autos)
                    for i in autos.keys():  
                        autos[i]["Frames"]=autos[i]["Frames"]+1
                        if autos[i]["Frames"]==frames_espera:
                            output_path="B:/Mega/Building AI/Final/v5.3_csv3/Autos.csv"
                            write_csv2(autos,car_id, output_path, new)
                            new=1
                            print("ENVIADO A CSV")
                    print("AUTOS CAR_ID FRAMES", autos[car_id]["Frames"])                
                    autos[car_id]["Frames"]=0

#Como hago el tracking de autos, puedo llamara a la función que escribe el CSV solo cuando un auto desaparece
#Solo que debería agregar contenido al CSV ya existente 
#Por ahora puedo probar hacer un CSV por cada vehículo, el csv se guarda solo cuando el vehículo desaparece o cuando pasan 
#cierta cantidad de frames

#Reviso results y me quedo solo con el mejor score de patente para un auto
print("-------------------------")

""" print(results)
for frame_nmr in results.keys(): #para todos los frames
    for car_id in results[frame_nmr].keys(): #para todos los autos dentro de cada frame
        print("CAR_ID: ",car_id)
        print("Patente: ",results[frame_nmr][car_id]["license_plate"]["text"])
        print("Score: ",results[frame_nmr][car_id]["license_plate"]["text_score"]) """

#Guardado de resultados:
#write_csv(results, "B:/Mega/Building AI/Final/v5.1_csv/test.csv")
print("DICCIONARIO DE AUTOS: ",autos)
#print("Autos actuales: ", autos_actuales)
