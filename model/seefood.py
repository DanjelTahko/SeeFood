from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC
from sklearn import neighbors, gaussian_process, naive_bayes

from skimage.restoration import denoise_tv_chambolle
from skimage.measure import find_contours
from skimage.filters import threshold_li
from skimage.transform import resize, rescale
from skimage.color import rgb2gray
from skimage.io import imread

import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import GridSearchCV

IMG_FORMAT = ['.png', 'jpeg', '.jpg']

class SeeFood:

    def __init__(self, dataset:str=None) -> None:

        
        """ 
            takes directory name where csv file with same name exists as argument

            raise FileNotFoundError if dataset path doesnt exist

           use foo=SeeFood().createData(training_data_dir) for creating dataset 
        """
        if (dataset != None):
            os.listdir(f'{os.getcwd()}/data/dataset/{dataset}')
            dataset = pd.read_csv(f'{os.getcwd()}/data/dataset/{dataset}/{dataset}.csv')

        self.dataset = dataset


    def createData(self, data_dir:str|list[str]) -> None:

        """get path with all training data and create dataset"""
        if (not isinstance(data_dir, list)):
            data_dir = [data_dir]

        images = []
        name = ''
        start = time.time()
        for _dir in data_dir:
            path = f'{os.getcwd()}/data/train/{_dir}'
            for img_file in os.listdir(path):
                if img_file[-4:].lower() not in IMG_FORMAT:
                    continue
                img = self.resizeImage(path + '/' + img_file)
                img_array = (np.array(img) * 100).flatten()
                images.append([*img_array, _dir, img_file])
            name += _dir + '_'
        name = name[:-1]

        columns = [f'px{i}' for i in range(60*60)] + ['name', 'filename']
        self.dataset = pd.DataFrame(images, columns=columns)
        os.makedirs(f'{os.getcwd()}/data/dataset/{name}')
        self.dataset.to_csv(f'{os.getcwd()}/data/dataset/{name}/{name}.csv',index=False)
        stop = time.time()
        tajm = stop - start
        print("\nSuccessfully created dataset")
        print(f"Took : {int(tajm/60)} min {tajm%60:.2}\nCreated dataset of {len(images)} images")


    def resizeImage(self, image_path:str) -> np.ndarray:
        
        """read image file from path, crop, resize and convert to numpy array """

        
        image = imread(image_path)

        denoised = denoise_tv_chambolle(image, channel_axis=True)
        try:
            gray_img = rgb2gray(denoised)
        except ValueError as e:
            print(f'Error with : {image_path}\n{e}')
            denoised = denoised[:, :, :-1]
            gray_img = rgb2gray(denoised)

        threshold = threshold_li(gray_img)
        img =  (gray_img > threshold).astype(int)
    
        contours = find_contours(img)

        x_min, y_min = img.shape[1], img.shape[0]
        x_max = y_max = 0
        for contour in contours:
            x = contour[:, 1]
            y = contour[:, 0]
            x_min = min(x_min, x.min())
            x_max = max(x_max, x.max())
            y_min = min(y_min, y.min())
            y_max = max(y_max, y.max())

        xmin = int(x_min)
        ymin = int(y_min)
        xmax = int(x_max + .9)
        ymax = int(y_max + .9)

        cropped = img[ymin:ymax, xmin:xmax]
        resized = resize(cropped, (60, 60), mode='constant', cval=1)

        return resized

    def predictData(self, filename:str): 

        path = f'{os.getcwd()}/static/images/{filename}'
        p_img = self.resizeImage(path)
        img_array = (np.array(p_img) * 100).flatten()

        X = self.dataset.iloc[:, :3600]
        Y = self.dataset['name']

        dtc = neighbors.KNeighborsClassifier()
        dtc.fit(X, Y)
        y_pred = dtc.predict(img_array.reshape(1,-1))

        #svc = SVC(kernel='poly', random_state=1)
        #svc.fit(X, Y)
        #y_pred = svc.predict(img_array.reshape(1,-1))
        print(f"-> Predicted {filename} = {y_pred}")
        #print(f"-> Prediction Report:\n{classification_report(y_test, y_pred)}")
        if (y_pred[0] != 'hotdog'):
            y_pred[0] = 'not hotdog'
        return y_pred

 

    def trainTestModelsReport(self):

        """If you want to test different models on your dataset"""
        
        """train model with data"""
        X = self.dataset.iloc[:, :3600]
        Y = self.dataset['name']

        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1)

        accuracy = ['model', 0.0]

        """Linear models"""
        from sklearn import linear_model 
        print("\n\n----------------------------------------------------------------\n")
        print("####### Linear Models #######\n")
        dtc = linear_model.RidgeClassifier(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> RidgeClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'RidgeClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = linear_model.SGDClassifier(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> SGDClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'SGDClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = linear_model.Perceptron(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> Perceptron\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'Perceptron'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = linear_model.PassiveAggressiveClassifier(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> PassiveAggressiveClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'PassiveAggressiveClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        print("----------------------------------------------------------------\n")
        print("####### Support Vector Machines #######\n")
        from sklearn import svm 

        dtc = svm.SVC(kernel='linear', random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> SVC (Linear)\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'SVC (Linear)'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = svm.SVC(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> SVC (RBF)\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'SVC (RBF)'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = svm.SVC(kernel='poly', random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> SVC (Poly)\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'SVC (Poly)'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = svm.NuSVC(kernel='linear', random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> NuSVC (Linear)\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'NuSVC (Linear)'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = svm.NuSVC(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> NuSVC (RBF)\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'NuSVC (RBF)'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = svm.NuSVC(kernel='poly', random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> NuSVC (Poly)\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'NuSVC (Poly)'
            accuracy[1] = accuracy_score(y_test, y_pred)

        print("----------------------------------------------------------------\n")
        print("####### Nearest Neighbors #######\n")

        dtc = neighbors.KNeighborsClassifier()
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> KNeighborsClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'KNeighborsClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = neighbors.NearestCentroid()
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> NearestCentroid\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'NearestCentroid'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = gaussian_process.GaussianProcessClassifier()
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> GaussianProcessClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'GaussianProcessClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = naive_bayes.ComplementNB()
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> ComplementNB\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'ComplementNB'
            accuracy[1] = accuracy_score(y_test, y_pred)


        print("----------------------------------------------------------------\n")

        dtc = DecisionTreeClassifier(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> DecisionTreeClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'DecisionTreeClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        dtc = RandomForestClassifier(random_state=1)
        dtc.fit(x_train, y_train)
        y_pred = dtc.predict(x_test)
        print(f"-> RandomForestClassifier\nReport:\n{classification_report(y_test, y_pred, zero_division=0)}")
        if (accuracy[1] < accuracy_score(y_test, y_pred)):
            accuracy[0] = 'RandomForestClassifier'
            accuracy[1] = accuracy_score(y_test, y_pred)

        print(f"\n{accuracy[0]} got best score with : {accuracy[1]:.4}")
