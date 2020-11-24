# Parcial
# Procesamiento de imagenes y visión
# Manuela Bravo Soto

# IMPORTACIONES
import numpy as np # Del módulo numpy
import cv2 # Del módulo opencv-python
import os # Del módulo os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from hough import *
from orientation_estimate import *

#CLASE Bandera
class Bandera:

    # CONSTRUCTOR
    # Recibe como parámetros la dirección de la imagen y su nombre
    def __init__(self,path,image_name):
        self.path = path # Se guarda la dirección de la imagen en self
        self.image_name = image_name # Se guarda el nombre de la imagen en self
        self.path_file = os.path.join(self.path, self.image_name) # Se unen los componentes para tener una dirección completa a la imagen
        self.imageRGB = cv2.imread(self.path_file) # Se lee la imagen de la dirección dada y se guarda en self
        #cv2.imshow("Image", self.imageRGB)
        #cv2.waitKey(0)

    # MÉTODO PARA OBTENER EL NÚMERO DE COLORES DE LA BANDERA
    def Colores(self):
        def intraclusterdist(image, centers, labels, rows, cols):
            dist = 0
            label_idx = 0
            for i in range(rows):
                for j in range(cols):
                    centroid = centers[labels[label_idx]]
                    point = image[i, j, :]
                    dist += np.sqrt(
                        np.power(point[0] - centroid[0], 2) + np.power(point[1] - centroid[1], 2) + np.power(
                            point[2] - centroid[2], 2))
                    label_idx += 1
            return dist

        hist = cv2.calcHist([self.imageRGB], [0], None, [256], [0, 256])
        #plt.plot(hist)
        #plt.show()
        #cv2.waitKey(0)
        array_peaks = np.array(hist)
        max = np.max(array_peaks)
        peaks = []
        for i in range(len(hist)):
            ispeak = True
            if i - 1 > 0:
                ispeak &= (array_peaks[i] > 1.8 * array_peaks[i - 1])
            if i + 1 < len(hist):
                ispeak &= (array_peaks[i] > 1.8 * array_peaks[i + 1])

            ispeak &= (array_peaks[i] > 0.05 * max)
            if ispeak:
                peaks.append(i)
        self.peaks = peaks
        self.numpeaks = len(peaks)

        image = np.array(self.imageRGB, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        distance = []
        for ncolors in range(1, self.numpeaks+1):
            image_array_sample = shuffle(image_array, random_state=0)[:10000]
            model = KMeans(n_clusters=ncolors, random_state=0).fit(image_array_sample)
            labels = model.predict(image_array)
            centers = model.cluster_centers_
            dist = intraclusterdist(image, centers, labels, rows, cols)
            distance.append(dist)
        quantity = len(distance)
        return quantity

    # MÉTODO PARA OBTENER EL PORCENTAJE DE CADA COLOR PRESENTE
    def Porcentaje(self):
        lower_bound = []
        upper_bound = []
        count = 0
        percent = []
        for n_peaks in range(self.numpeaks):
            lower_bound.append(np.array([self.peaks[n_peaks] - 10, 0, 0]))
            upper_bound.append(np.array([self.peaks[n_peaks] + 10, 255, 255]))
            mask = cv2.inRange(self.imageRGB, lower_bound[count], upper_bound[count])
            height, width = mask.shape[:2]
            num_pixels = height * width
            count_color = cv2.countNonZero(mask)
            percent.append(int((count_color / num_pixels) * 100))
            count = count + 1
        if len(percent) == 3:
            percent.append(0)
        elif len(percent) == 2:
            percent.append(0)
            percent.append(0)
        elif len(percent) == 1:
            percent.append(0)
            percent.append(0)
            percent.append(0)
        return percent

    # MÉTODO PARA OBTENER LA ORIENTACIÓN DE LAS FRANJAS
    def Orientacion(self):
        high_thresh = 300
        bw_edges = cv2.Canny(self.imageRGB, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough1 = hough(bw_edges)
        accumulator = hough1.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hough.find_peaks(bw_edges, accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = self.imageRGB.shape[:2]
        image_draw = np.copy(self.imageRGB)
        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough1.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough1.center_x
            y0 = b * rho + hough1.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) < 80:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
            elif np.abs(theta_) > 100:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [255, 0, 255], thickness=2)
            else:
                if theta_ > 0:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 0], thickness=2)
                else:
                    image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)

        cv2.imshow("frame", bw_edges)
        cv2.imshow("lines", image_draw)
        cv2.waitKey(0)
        [theta_data, M] = orientation_map(bw_edges, 7)
        theta_data += np.pi / 2
        theta_data /= np.pi
        theta_uint8 = theta_data * 255
        theta_uint8 = np.uint8(theta_uint8)
        print(len(theta_data),len(theta_data[0]))
        print(theta_uint8[1][2])
        band = True
        '''
        for x in range(len(theta_data)):
            for vl in range(len(theta_data[0])):
                if theta_uint8[x][vl] != 127:
                    print('Hola')
                else:
                    str = 'vertical'
        '''
        for x in range(len(theta_data)):
            if not(all(i == theta_uint8[x][0] for i in list)):
                band = False
                break
        if band == True:
            str = 'vertical'
        else:
            band = True
            for x in range(len(theta_data[0])):
                if not(all(i == theta_uint8[0][x] for i in list)):
                    band = False
                if band == False:
                    break
            if band == True:
                str = 'horizontal'
            else:
                str = 'mixto'
        theta_uint8 = cv2.applyColorMap(theta_uint8, cv2.COLORMAP_JET)
        theta_view = np.zeros(theta_uint8.shape)
        theta_view = np.uint8(theta_view)
        theta_view[M > 0.2] = theta_uint8[M > 0.2]
        cv2.imshow("Image", theta_view)
        cv2.waitKey(0)
        return str

if __name__ == '__main__':
    path = '/Users/mbrav/OneDrive/Desktop/parcial'
    image_name = 'flag5.png'
    flag = Bandera(path,image_name)
    #quantity = flag.Colores()
    #print(quantity)
    #color_percent = flag.Porcentaje()
    #print(color_percent)
    orient = flag.Orientacion()
    print(orient)