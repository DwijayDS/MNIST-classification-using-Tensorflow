'''Importing libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''Extracting training data'''
train_data = pd.read_csv('mnist_train.csv', header=None)
train_data1=np.array(train_data)
train_image=train_data1[:,1:]/255
train_label=train_data1[:,0]

'''Extracting testing data'''
test_data = pd.read_csv('mnist_test.csv', header=None)
test_data1=np.array(test_data)
test_image=test_data1[:,1:]/255
test_label=test_data1[:,0]


'''Setting placeholder'''
n_inputs = 28*28 #MNIST
batch_size = 50
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


def neuron_layer(X, n_neurons, name, activation=None):
    '''Function to define working of perceptrons of one layer of neural network'''
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros(n_neurons), name="bias")
        output = tf.matmul(X,W) + b
        if activation is not None:
            return activation(output)
        else:
            return output

        
def take_next_batch(current_batch_no):
    '''
    Function to return images and their labels of certain batch size 
    '''
    
    out_img=train_image[current_batch_no:(current_batch_no + batch_size), : ]
    out_label=train_label[current_batch_no:(current_batch_no + batch_size)]
    current_batch_no += batch_size
    #print(len(out_img))
    return (out_img,out_label,current_batch_no)


def NeuralNetworkStructure(X):
    with tf.name_scope("ANN"):
        '''Defining sizes and variable'''
        n_inputs = 28*28 #MNIST
        n_hidden1 = 300
        n_hidden2 = 100
        n_outputs = 10
        '''Defining hidden layer'''
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        output_without_activation = neuron_layer(hidden2, n_outputs, name="outputs")

        return output_without_activation

def train_neural_network():
    '''Master fuction to train the neural network defined'''
    '''Calling the neural networkl function'''
    logits = NeuralNetworkStructure(X)
    '''Starting with finding loss'''
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        '''cost function'''
        loss=tf.reduce_mean(cross_entropy, name="loss")
    '''Minimizing cost function using gradient descent algorithm'''
    learning_rate=0.01

    with tf.name_scope("grad"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_output=optimizer.minimize(loss)
    '''Accuracy:- .in_top_k() function checks if highest logit corresponds to target class'''
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    '''saving coputation graph'''
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    '''initializing'''
    n_epochs = 40
    
    '''starting session'''
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            current_batch_no=0
            for iteration in range(len(train_image)//batch_size):
                epoch_x,epoch_y,current_batch_no = take_next_batch(current_batch_no)
                sess.run(training_output, feed_dict={X: epoch_x, y: epoch_y})
            acc_train = accuracy.eval(feed_dict={X: epoch_x, y: epoch_y})
            acc_test = accuracy.eval(feed_dict={X: test_image, y: test_label})

            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path=saver.save(sess, "./final_model_graph")



print("Input t for training and c for checking the neural network")
choice = input("Choice:")
if choice == 't':
    train_neural_network()
else:
    logits = NeuralNetworkStructure(X)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./final_model_graph")
        X_new = test_image[8].reshape(1,28*28)
        Z = logits.eval(feed_dict={X: X_new})
        ypred = np.argmax(Z,axis=1)
                






    

