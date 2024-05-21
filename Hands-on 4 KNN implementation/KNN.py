from collections import Counter

class Object:
    def __init__(self, x, y, tag): 
        self.x = x
        self.y = y
        self.tag = tag
        self.distance = 'null'
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_color(self):
        return self.color
    
    def get_distance(self):
        return self.distance

class Calculate:
    # Constructor
    def __init__(self): 
        self.result = 0

    def calculate_mean(self, data):
        return sum(data) / len(data)

    def calculate_std(self, data):
        mean = self.calculate_mean(data)
        self.result = sum((x - mean) ** 2 for x in data) / len(data)
        return self.result ** 0.5
    
    def calculate_euclidean_distance(self, test_dot, train_data):
        for dot in train_data:
            distance = ((test_dot.x - dot.x) ** 2 + (test_dot.y - dot.y) ** 2) ** 0.5
            dot.distance = distance
            #print (distance)
        return train_data  


class KNN:
    # Constructor
    def __init__(self, k): 
        self.k = k
        self.nearest_neighbors = []
    
    def get_k(self):
        return self.k
    
    def get_nn(self):
        return self.nearest_neighbors

    def train_data(self, x, y, tags, obj):
        data = []
        s_x = []
        s_y = []
        for i, j in zip(x, y):
            # Estandarizamos x
            nx = (i - obj.calculate_mean(x)) / obj.calculate_std(x)
            s_x.append(nx)
            
            # Estandarizamos y
            ny = (j - obj.calculate_mean(y)) / obj.calculate_std(y)
            s_y.append(ny)
       
        #creamos los objetos con sus tributos
        for i in range(len(x)):
            dot = Object(x[i], y[i], tags[i])
            data.append(dot)
        
        return data


    def get_sorted_by_distance(self, train_data):
        sorted_data = sorted(train_data, key=lambda dot: dot.distance)
        return sorted_data

    def to_predict(self, data_set, new_dot, k, obj):
        for i in range(k):
            self.nearest_neighbors.append(data_set[i])
        
        count_tags = Counter(neighbor.tag for neighbor in self.nearest_neighbors)
        most_common_tag = count_tags.most_common(1)[0][0]

        new_dot.tag = most_common_tag
        return new_dot

    
    def print_results(self, data, result):
        print("\nThe nearest neighbors are:")
        for i in range(self.get_k()):
            print(f"\tNeighbor {i + 1}: (x = {data[i].x}, y = {data[i].y}, tag = {data[i].tag}, distance = {data[i].distance:.2f})")
        print(f"\nThe new value is: {result.tag}")


# train data
x=[158,158,158,160,160,163,163,160,163,165,165,165,168,168,168,170,170,170]
y=[58,59,63,59,60,60,61,64,64,61,62,65,62,63,66,63,64,68]
tags=['M','M','M','M','M','M','M','L','L','L','L','L','L','L','L','L','L','L']

knn = KNN(5)
cal = Calculate()

#creating the dots from the train data
train_data = knn.train_data(x, y, tags, cal)

#new value to predict
new_entry = Object(161, 61, '?')

#calculate the distances between the data_train and the dot to predict
train_data = cal.calculate_euclidean_distance(new_entry, train_data)
# Ordenar los datos por distancia
train_data = knn.get_sorted_by_distance(train_data)
result = knn.to_predict(train_data, new_entry, knn.get_k(), cal)

#print results
knn.print_results(knn.get_nn(), result)
