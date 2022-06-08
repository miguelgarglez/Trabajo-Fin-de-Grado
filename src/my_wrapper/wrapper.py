import mxnet as mx
from mxnet import gluon, autograd, metric, nd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from abc import ABC, abstractmethod

class BaseGluonModel(BaseEstimator, ABC):
    def __init__(self, model_function, loss_function, init_function=None, batch_size=50,
        learning_rate=0.001, epochs=10, optimizer='sgd', alpha=1.0, verbose=False, format_img_data=None):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.model_function = model_function
        # funcion que debe pasarse como argumento, que crea un modelo personalizado
        # y devuelve una instancia de un modelo propio de gluon
        self.model = model_function()
        self.optimizer = optimizer
        self.loss_function= loss_function
        self.init_function = init_function
        self.verbose = verbose
        #self.loss_values = []
        self.format_img_data = None
        if format_img_data:
            if len(format_img_data) == 2:
                self.format_img_data = (1, format_img_data[0], format_img_data[1])
            elif len(format_img_data) == 3:
                self.format_img_data = format_img_data
            else:
                raise Exception("Introduce a tuple, (dim1, dim2) or (channel, dim1, dim2) are the desired shapes")


    def format_data(x):
        """
        Función de la clase utilizada para formatear los datos
        de forma adecuada a MXNet, convirtiéndolos a 'mxnet.numpy.ndarray'
        Args:
            x (numpy.ndarray): datos a formatear
            **kwargs: otros argumentos
        Returns:
            mxnet.numpy.ndarray: array con el formato adecuado a MXNet

        """
        X = mx.nd.array(x).as_np_ndarray()

        return X

    def format_img_data(x, dims):
        """
        Función de la clase utilizada para formatear los datos
        de una imagen y adecuarlos a una entrada de red convolucional.
        Args:
            x (np.ndarray): datos a formatear
            **kwargs: otros argumentos
        Returns:
            mxnet.numpy.ndarray: array con el formato adecuado a MXNet
                                 con las dimensiones adecuadas

        """
        channels = dims[0]
        dim1 = dims[1]
        dim2 = dims[2]
        X = x.reshape(-1, channels, dim1, dim2)

        return X

    def fit(self, X, y, **kwargs):
        """
        Método que entrena el modelo dados unos datos y sus respectivas salidas 
        Args:
            X (mxnet.numpy.ndarray): datos usados por el modelo para el entrenamiento
            y (mxnet.numpy.ndarray): salidas reales de los datos usadas por el modelo para el entrenamiento
            **kwargs: otros argumentos
        Returns:
            None

        """
        X = BaseGluonModel.format_data(X)
        y = BaseGluonModel.format_data(y)

        # si se tiene que redimensionar
        if self.format_img_data:
            X = BaseGluonModel.format_img_data(X, self.format_img_data)
            
        dataset = gluon.data.ArrayDataset(X, y)
        data_iter = gluon.data.DataLoader(dataset, batch_size=self.batch_size)

        if self.init_function:
            self.model.initialize(self.init_function)
        else:
            self.model.initialize()


        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer,
            {'learning_rate': self.learning_rate, 'wd': self.alpha})
        
        # bucle de entrenamiento
        for epoch in range(self.epochs + 1):
            for X_i, y_i in data_iter:
                with autograd.record():
                    l = self.loss_function(self.model(X_i), y_i)
                l.backward()
                trainer.step(self.batch_size)
            if self.verbose:
                if epoch % 10 == 0:
                    l = self.loss_function(self.model(X), y)
                    print(f'epoch {epoch}, loss {l.mean().asnumpy():f}')


    @abstractmethod
    def predict(self, x, **kwargs):
        """
        Predice la salida para los datos (x)
        
        Args:
            x (numpy.ndarray): datos usados por el modelo para predecir
            **kwargs: otros argumentos
        Returns:
            mxnet.numpy.ndarray: las predicciones obtenidas

        """
        pass



class GluonRegressor(BaseGluonModel, RegressorMixin):
    
    def predict(self, x, **kwargs):
        """
        Predice una salida para los datos (x)
        
        Args:
            x (mxnet.numpy.ndarray): datos usados por el modelo para predecir
            **kwargs: otros argumentos
        Returns:
            mxnet.numpy.ndarray: las predicciones obtenidas

        """
        X = BaseGluonModel.format_data(x)

        ret = np.array(self.model(X))

        return ret


class GluonClassifier(BaseGluonModel, ClassifierMixin):

    def predict(self, x, **kwargs):
        """
        Predice una salida para los datos (x)
        
        Args:
            x (mxnet.numpy.ndarray): datos usados por el modelo para predecir
            **kwargs: otros argumentos
        Returns:
            mxnet.numpy.ndarray: las predicciones obtenidas

        """
        x = BaseGluonModel.format_data(x)
        if self.format_img_data:
            x = BaseGluonModel.format_img_data(x, self.format_img_data)

        ret = np.array(self.model(x).argmax(axis=1)).astype(np.int64)

        return ret




