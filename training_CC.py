import tensorflow as tf
import model_b_modified.config_CC as config
import model_b_modified.model_CC as model
import model_b_modified.utils_CC as utils
import cv2
import numpy as np
from numpy import matrix
import re
import glob
import pandas as pd
import cv2
import sys

def input_data():
    file_name = config.file_path["train_xlsx"]
    list_of_patient = []
    df = pd.read_excel(io=file_name)
    patient_id = df['patient_id'].tolist()
    target_value = df["assessment"].tolist()
    count = 0
    for i in range(len(patient_id)):
        count += 1
        list_of_patient.append([patient_id[i], target_value[i]])
    #print(count)
    list_patient_train_data =[]
    array_images = []
    array_target = []
    for i in range(len(list_of_patient)):
        for file in glob.glob(config.folder_path["train_dataset_glob"]):
            if str(str(list_of_patient[i][0]).split("_")[-1]) in str(file):
                key_word = str(str(list_of_patient[i][0]).split("_")[-1])
                target_element = str(list_of_patient[i][1])
                try:
                    temp_path_CC = config.folder_path["train_dataset"] + "_" + key_word + "_" + "RIGHT" + "_" + "CC" + ".png"
                except Exception as error:
                    print(error)
                train_x1_rcc = 0
                try:
                    train_x1_rcc = cv2.imread(temp_path_CC,0)
                except Exception as e:
                    print(e)
                if train_x1_rcc is None:
                    try:
                        temp_path_MLO = config.folder_path["train_dataset"] + "_" + key_word + "_" + "LEFT" + "_" + "CC" + ".png"
                    except Exception as error:
                        print(error)
                    try:
                        train_x1_rcc = cv2.imread(temp_path_MLO, 0)
                        train_x1_rcc = cv2.flip(train_x1_rcc, 1)
                    except Exception as e:
                        print(e)
                #print(temp_path_CC, temp_path_MLO, target_element)
                # image = cv2.imread(temp_path_MLO,0)
                list_patient_train_data.append(str(list_of_patient[i][0]).split("_")[-1])
                try:
                    #train_x1_rcc = cv2.imread(temp_path_CC, 0)
                    train_x1_rcc = cv2.resize(train_x1_rcc, (2000, 2600))
                    train_x1_rcc = np.array(train_x1_rcc).reshape(1, 2000, 2600, 1)
                    array_images.append(train_x1_rcc)

                    if target_element == 0:
                        train_y = matrix([[1, 0, 0]])

                    if target_element == 1:
                        train_y = matrix([[0, 1, 0]])
                    if target_element == 2:
                        train_y = matrix([[0, 0, 1]])
                    else:
                        train_y = matrix([[0, 1, 0]])

                    array_target.append(train_y)
                except Exception as error:
                    print(error)
    print("List of Train patient ", list_patient_train_data)
    return array_images, array_target
list_patient_test_data = []
def test_data():
    file_name = config.folder_path["test_dataset"]
    list_of_patient = []
    df = pd.read_excel(io=file_name)
    patient_id = df['patient_id'].tolist()
    target_value = df["assessment"].tolist()
    count = 0
    for i in range(len(patient_id)):
        count += 1
        list_of_patient.append([patient_id[i], target_value[i]])
    #print(count)
    test_array_images = []
    test_array_target = []
    for i in range(len(list_of_patient)):
        for file in glob.glob(config.folder_path["test_dataset_glob"]):
            if str(str(list_of_patient[i][0]).split("_")[-1]) in str(file):
                key_word = str(str(list_of_patient[i][0]).split("_")[-1])
                target_element = str(list_of_patient[i][1])
                temp_path_RCC = config.folder_path["test_dataset"] + "_" + key_word + "_" + "RIGHT" + "_" + "CC" + ".png"
                temp_path_LCC = config.folder_path["test_dataset"] + "_" + key_word + "_" + "LEFT" + "_" + "CC" + ".png"

                list_patient_test_data.append(str(list_of_patient[i][0]).split("_")[-1])
                try:
                    train_x1_rcc = cv2.imread(temp_path_RCC,0)
                except Exception as e:
                    print(e)

                if train_x1_rcc is None:
                    try:
                        train_x1_rcc = cv2.imread(temp_path_LCC, 0)
                        train_x1_rcc = cv2.flip(train_x1_rcc, 1)
                    except Exception as e:
                        print(e)

                try:
                    #train_x1_rcc = cv2.imread(temp_path_CC, 0)
                    train_x1_rcc = cv2.resize(train_x1_rcc, (2000, 2600))

                    train_x1_rcc = np.array(train_x1_rcc).reshape(1, 2000, 2600, 1)
                    test_array_images.append(train_x1_rcc)

                    if target_element == 0:
                        train_y = matrix([[1, 0, 0]])

                    if target_element == 1:
                        train_y = matrix([[0, 1, 0]])
                    if target_element == 2:
                        train_y = matrix([[0, 0, 1]])
                    else:
                        train_y = matrix([[0, 1, 0]])

                    test_array_target.append(train_y)
                except Exception as error:
                    print(error)
    print("List of test patients ", str(list_of_patient[i][0]).split("_")[-1])
    return test_array_images, test_array_target


#Parameters
training_iters = 10
learning_rate = 0.001
batch_size = 1
no_epochs = 10
n_classes = 3

x1_cc = tf.placeholder(tf.float32, shape=[1, 2000, 2600, 1])
x = (x1_cc)
y = tf.placeholder(tf.float32, shape=(1, 3))

prediction = model.baseline(x, parameters=None)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.99, epsilon=0.1)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
train2 = optimizer2.minimize(cost)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    init.run()
    # sess.run(init)
    train_loss = []
    test_loss = []
    valid_loss = []
    valid_accuracy = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter(config.model_ckpts["model_output"], sess.graph)
    array_images, array_target = input_data()
    array_images_test, array_target_test = test_data()
    print("Size of Dataset: ")
    print("Training Set : ", len(array_images))
    print("Validation Set : ", len(array_images_test) // 2)
    print("Test Set : ", len(array_images_test) // 2)
    iterator = 0
    for epoch in range(no_epochs):
        for j in range(len(array_images) - 1):
        #for j in range(2):
            train_x1_rcc = array_images[j]
            train_y = array_target[j]
            #opt = sess.run(optimizer, feed_dict={x1_cc: train_x1_rcc, y: train_y})
            feed_dict = {x1_cc: train_x1_rcc, y: train_y}
            sess.run(train, feed_dict)
            loss, acc = sess.run([cost, accuracy],
                                 feed_dict={x1_cc: train_x1_rcc, y: train_y})
            train_accuracy.append(acc)
            train_loss.append(loss)

        for k in range((len(array_images_test) // 2) - 1):
        #for k in range(2):
            test_X = array_images_test[k]
            test_y = array_target[k]
            validation_acc, validation_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})

            valid_loss.append(validation_loss)

            valid_accuracy.append(validation_acc)
        overall_valid_accuracy = np.mean(valid_accuracy)
        overall_train_accuracy =  np.mean(train_accuracy)
        overall_valid_loss = np.mean(valid_loss)
        overall_train_loss = np.mean(train_loss)
        print("Epoch ", epoch, " Training Accuracy: ", overall_train_accuracy, " Training Loss: ", overall_train_loss, " Validation Accuracy: ", overall_valid_accuracy, " Validation Loss: ", overall_valid_loss)
        for l in range((len(array_images_test) // 2), len(array_images_test), 1):
        #for l in range(2):
            test_X = array_images_test[l]
            test_y = array_target[l]
            test_acc_temp, test_loss_temp = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})

            test_loss.append(test_loss_temp)
            test_accuracy.append(test_acc_temp)
        overall_test_accuracy = np.mean(test_accuracy)
        overall_test_loss = np.mean(test_loss)
        print(" Test Accuracy: ", overall_test_accuracy, " Test Loss: ", overall_test_loss)

    summary_writer.close()

    savePath = saver.save(sess, config.model_ckpts["saved_model"])
