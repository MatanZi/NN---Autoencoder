from ImageProduce import *
from NueralNetwork import *
from random import randint


def split_data(data_set, percent):
    data_num = int(len(data_set)*percent)
    train_data = []
    test_data = []
    counter = 0
    lines = set()
    while counter != data_num:
        line = randint(0, data_num)
        if line not in lines:
            counter += 1
            lines.add(line)
            train_data.append(data_set[line])

    rest_lines = set(range(0, data_num)) - lines
    for line in rest_lines:
        test_data.append(data_set[line])

    return train_data, test_data


if __name__ == '__main__':
    img_handler = ImageProduce('Photo of Lena in ppm.jpg', 16)
    img_handler2 = ImageProduce('Photo of Richard in ppm.jpg', 16)
    data_set_lena = img_handler.matrix
    data_set_richard = img_handler2.matrix
    n_inputs = len(data_set_lena[0])
    n_outputs = n_inputs
    l_rate = 0.05
    epochs = 30
    hidden_layers = 10
    neural_network = NeuralNetwork(n_inputs, n_outputs, l_rate, epochs, hidden_layers)
    train_error = neural_network.train_network(data_set_lena)
    neural_network.draw_error_graph(train_error, "Train", "Epoch", "EX1_Lena_Train")

#----------- test Lena photo --------------------
    predict_list, test_error = neural_network.test_predict(data_set_lena)
    neural_network.draw_error_graph(test_error, "Test" , "Row" , "EX1_Lena_Test")
    img_handler.reconstruct(predict_list)

#----------- test Monkey photo --------------------
    predict_list, test_error = neural_network.test_predict(data_set_richard)
    neural_network.draw_error_graph(test_error, "Test" , "Row" , "EX1_Richard_Test")
    img_handler2.reconstruct(predict_list)



