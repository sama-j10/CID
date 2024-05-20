# HANDS-ON 3  Polynomial Regression
#Morales Miranda Samantha

import numpy as np
import mpmath
mpmath.mp.dps = 50 # ajustar precision 

class DataSet:
    # Constructor
    def __init__(self):
        self.x = np.array([108.0, 115.0, 106.0, 97.0, 95.0, 91.0, 97.0, 83.0, 83.0, 78.0, 54.0, 67.0, 56.0, 53.0, 61.0, 115.0, 81.0, 78.0, 30.0, 45.0, 99.0, 32.0, 25.0, 28.0, 90.0, 89.0])
        self.y = np.array([[95.0], [96.0], [95.0], [97.0], [93.0], [94.0], [95.0], [93.0], [92.0], [86.0], [73.0], [80.0], [65.0], [69.0], [77.0], [96.0], [87.0], [89.0], [60.0], [63.0], [95.0], [61.0], [55.0], [56.0], [94.0], [93.0]])

    # Getters
    def get_x(self):
        return self.x
    def get_y(self):
        #convierte a precision arbitraria
        y_mpmath = np.array([[mpmath.mpf(val) for val in row] for row in self.y])
        return y_mpmath
    
class Algebra:
    # Constructor
    def __init__(self, degree):
        self.degree = degree  
        self.result = 0
        self.matrix = []

    def create_matrix(self, x):
        for i in range(len(x)):
            fila = []
            for colum in range(self.degree+1):
                fila.append(x[i]**colum)
            self.matrix.append(fila)
        x_mpmath = np.array([[mpmath.mpf(val) for val in row] for row in self.matrix]) #convierte a precision arbitraria
        return x_mpmath
    
    def transpose(self, m):
        num_rows = len(m)
        num_columns = len(m[0])  
        self.matrix = []

        for i in range(num_columns): 
            fila = []
            for j in range(num_rows): 
                fila.append(m[j][i])
            self.matrix.append(fila)

        return self.matrix
    
    def inverse(self, m):
        n = len(m)
    
        m_identity = [[0] * n for _ in range(n)]  #matriz identidad
        for i in range(n):
            m_identity[i][i] = 1
        
        a_matrix = [row + m_identity[row_num] for row_num, row in enumerate(m)]  #matriz aumentada
        
        for col in range(n): #gauss-jordan
            max_row = max(range(col, n), key=lambda i: abs(a_matrix[i][col])) 
            a_matrix[col], a_matrix[max_row] = a_matrix[max_row], a_matrix[col]
            
            diag = a_matrix[col][col]
            a_matrix[col] = [val / diag for val in a_matrix[col]]
            
            for row in range(n):
                if row != col:
                    scale = a_matrix[row][col]
                    a_matrix[row] = [val - scale * a_matrix[col][idx] for idx, val in enumerate(a_matrix[row])]
        
        inverse_matrix = [row[n:] for row in a_matrix]
        return inverse_matrix
        
    
    def matrix_multiply(self, m1 ,m2):
        self.result = m1 @ m2
        return self.result


class Polynomial_R:
    # Constructor
    def __init__(self):
        self.result = 0

    def calculate_beta(self, xt_x_inve, xt_y):
        res_float = xt_x_inve @ xt_y
        self.result= np.array([[float(val) for val in row] for row in res_float])
        return self.result
    
    def r_squared(self, y_true, x_matrix, coeficientes):
        #calcula las predicciones de y , de acuerdo al modelo
        y_pred = np.dot(x_matrix, coeficientes)
        # Calcular la suma total de los cuadrados
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        # Calcular la suma de los cuadrados de los residuos
        ss_res = np.sum((y_true - y_pred)**2)
        # Calcular R^2
        r_squared = 1 - (ss_res / ss_total)
        return r_squared
    
    def correlation_coefficient(self, r_squared):
        correlation_coefficient = r_squared**0.5
        return correlation_coefficient

    def prediction(self, x_values, coeficientes):
        for x in x_values:
            y_pred = 0
            for y in range(len(coeficientes)):
                y_pred += coeficientes[y]*(x**y)
            print("Para x = ", x, "y_pred = ", y_pred)
        print('\n')

    def to_print_equation(self, coeficientes, degree):
        print("\nEcuación de Regresión Polinomial de grado "+ str(degree) + ':')
        eq = '\n\ty = '
        for i in range(degree+1):
            if i == 0:
                eq += str(coeficientes[i])
            else: 
                eq += ' + '+ str(coeficientes[i]) + ' x' + '^' +str(i)
        print(eq+'\n')

             
# main

data = DataSet()  
regression = Polynomial_R()
random_values = [109, 90, 120]  # Valores para predecir y

#Aproximacion Lineal
degree = 1   #se pone el grado que se desea calcular
lineal = Algebra(degree)

# crea la matriz X 
x_matrix = lineal.create_matrix(data.get_x())

# Realizar las operaciones
x_trans = lineal.transpose(x_matrix)
xt_x = lineal.matrix_multiply(x_trans, x_matrix)
xt_x_float = xt_x.astype(float)
xt_x_inv = np.linalg.inv(xt_x_float)
xt_y = lineal.matrix_multiply(x_trans, data.get_y())

#calcular beta
coeficientes = regression.calculate_beta(xt_x_inv, xt_y)

np.set_printoptions(suppress=True)# quita notacion cientifica

# Ecuación de Regresión Polinomial
regression.to_print_equation(coeficientes, degree)

#predicciones
regression.prediction( random_values, coeficientes)

# Calcular R^2
r_squared = regression.r_squared(data.get_y(),x_matrix, coeficientes)
print("Coeficiente de determinación (R^2):  %",round(r_squared*100,2))

# Calcular coeficiente de correlación
corr_coefficient = regression.correlation_coefficient(r_squared)
print("Coeficiente de Correlación: %", round(corr_coefficient*100,2))






