# HANDS-ON 2  Simple Lineal Regression
#Morales Miranda Samantha

import random

class DataSet:
    # Constructor
    def __init__(self):
        self.x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.y = [2, 4, 6, 8, 10, 12, 14, 16, 18]

    # Getters
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    
class Discrete_Maths:
    # Constructor
    def __init__(self, n):
        self.n = n
        self.result = 0

    # Methods
    def sum_of(self, data):
        result = sum(data)
        return result
    def sum_quad(self, data):
        result = sum(i**2 for i in data)
        return result
    def sumX_sumX(self, data):
        result = sum(x*x for x in data)
        return result
    def sum_XY(self, dataX, dataY):
        result = sum(x*y for x,y in zip(dataX,dataY))
        return result
    

class SLR:
    # Constructor
    def __init__(self):
        self._beta0 = 0
        self._beta1 = 0
        self._correlation = 0
        self._determination = 0
    
    #Setters
    def set_beta0(self, data):
        self._beta0 = data
    def set_beta1(self, data):
        self._beta1 = data
    def set_correlation(self, data):
        self._correlation = data
    def set_determination(self, data):
        self._determination = data

    #Getters
    def get_beta0(self):
        return self._beta0
    def get_beta1(self):
        return self._beta1
    def get_correlation(self):
        return self._correlation
    def get_determination(self):
        return self._determination
    
    #Methods
    def to_compute_beta0(self, sum_y, beta_1, sum_x, n):
        beta_0 = (sum_y - (beta_1 * sum_x)) / n
        return beta_0
    
    def to_compute_beta1(self, sum_y, sum_x, n, sum_xy, square_x):
        beta_1 = ((n * sum_xy) - (sum_x * sum_y)) / ((n * square_x) - (sum_x * sum_x))
        return beta_1
    
    def correlation_c(self, sum_xy,sum_x, sum_y, square_x, square_y, n):
        correlation = (n*sum_xy-sum_x*sum_y)/((n * square_x - sum_x ** 2) * (n * square_y - sum_y ** 2)) ** 0.5
        return correlation
    
    def determination_c(self, correlation):
        determination = correlation**2
        return determination
    
    def to_predict(self, x):
        result = self._beta0 + (self._beta1 * x)
        return result
    
    def to_print_equation(self, x, result):
        return f"Y = {self._beta0:.2f} + {self._beta1:.2f} * {x} = {result:.2f}\n\n"
    
#main
    
data_set = DataSet()
math = Discrete_Maths(len(data_set.get_x()))
slr = SLR()

#calculate betas
beta1=slr.to_compute_beta1(math.sum_of(data_set.get_y()),math.sum_of(data_set.get_x()),math.n,math.sum_XY(data_set.get_x(),data_set.get_y()),math.sum_quad(data_set.get_x()))
slr.set_beta1(beta1)

beta0=slr.to_compute_beta0(math.sum_of(data_set.get_y()),slr.get_beta1(),math.sum_of(data_set.get_x()),math.n)
slr.set_beta0(beta0)

#calculate coefficientes
correlation = slr.correlation_c(math.sum_XY(data_set.get_x(),data_set.get_y()),math.sum_of(data_set.get_x()),math.sum_of(data_set.get_y()),
                                  math.sum_quad(data_set.get_x()),math.sum_quad(data_set.get_y()),math.n)
slr.set_correlation(correlation)
determination = slr.determination_c(slr.get_correlation())
slr.set_determination(determination)

#predict a new y value with a random x
random_x = random.randint(0, 100)
new_value = slr.to_predict(random_x)

#results
print("\n\tHANDS-ON 2\n")
print("Beta 0 = ",f"{slr.get_beta0():.2f}")
print("Beta 1 = ",f"{slr.get_beta1():.2f}")
print("Correlation Coefficient = ",slr.get_correlation())
print("Coefficient of Determination = ",slr.get_determination())
print("\nNew Predicted Value:")
print(slr.to_print_equation(random_x, new_value))
