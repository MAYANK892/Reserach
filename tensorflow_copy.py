from __future__ import division
from scipy.signal import butter, lfilter
import os
import numpy as np
import math
import pandas as pd
from scipy import signal
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


# Loading the data
seed = 100
tf.set_random_seed(seed)
np.random.seed(seed)

x_df=pd.read_csv('/home/vaneet/PPG_Experiment1_June30/Predicting_heartrate/X_green_unfilter_50sec.csv', header=None)
y_df=pd.read_csv('/home/vaneet/PPG_Experiment1_June30/Predicting_heartrate/Actual Heart Rate_green_unfilter_50sec.csv', header=None)

print(x_df.isnull().sum().sum())
print(y_df.isnull().sum().sum())
# print(x_df.shape)
# print(y_df.shape)
#Converting the data into np array

x_vals=np.array(x_df)
y_vals=np.array(y_df)

x_vals= x_vals / (x_vals.max(axis=0) + np.spacing(0))


print x_vals.shape[0]
print y_vals.shape[0]
n_col_x=np.shape(x_vals)[1]
n_col_y=np.shape(y_vals)[1]




#Defining placeholders for x and y
x_data = tf.placeholder(shape=[None, n_col_x], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, n_col_y], dtype=tf.float32)

print(len(x_vals))


learningRate = tf.placeholder(dtype=tf.float32)

#Defining the test data and train data
#-------Check if I am doing this splitting right---------#


#Setting variables for deep learning
# hi represent the ith hidden layer
h1 = 500
h2 = 500
h3 = 500
#h4 = 50
#h5 = 25

#dont change these
m=6
p=n_col_x
q=n_col_y

#Ai's defines the variables for weights and bi's defines the bias for each of the hidden layer


# iF change number of layers, dont just change shape, CHANGE EVERYTHING!
A1 = tf.Variable(tf.random_uniform(shape=[p, h1], maxval=tf.sqrt(m/(p+h1+0.0)), minval=-tf.sqrt(m/(p+h1+0.0))))
b1 = tf.Variable(tf.constant([0.0]*h1))
A2 = tf.Variable(tf.random_uniform(shape=[h1,h2],maxval=tf.sqrt(m/(h1+h2+0.0)),minval=-tf.sqrt(m/(h1+h2+0.0))))
b2 = tf.Variable(tf.constant([0.0]*h2))
A3 = tf.Variable(tf.random_uniform(shape=[h2,h3],maxval=tf.sqrt(m/(h2+h3+0.0)),minval=-tf.sqrt(m/(h2+h3+0.0)))) #tf.Variable(tf.random_normal(shape=[h2,h3],stddev= 2))
b3 = tf.Variable(tf.constant([0.0]*h3))
A4 = tf.Variable(tf.random_uniform(shape=[h3,q],maxval=tf.sqrt(m/(h3+q+0.0)),minval=-tf.sqrt(m/(h3+q+0.0)))) #tf.Variable(tf.random_normal(shape=[h2,h3],stddev= 2))
#b4 = tf.Variable(tf.constant([-10.0]*h4))
#A5 = tf.Variable(tf.random_uniform(shape=[h4,h5],maxval=tf.sqrt(m/(h5+h4+0.0)),minval=-tf.sqrt(m/(h5+h4+0.0))))
#b5 = tf.Variable(tf.constant([-10.0]*h5))
#A6 = tf.Variable(tf.random_uniform(shape=[h5,q],maxval=tf.sqrt(m/(q+h5+0.0)),minval=-tf.sqrt(m/(q+h5+0.0))))

#Recording outputs from each layer
output1 = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
output2 = tf.nn.relu(tf.add((tf.matmul(output1, A2)),b2))
#output2 = ((tf.matmul(output1, A2)))
output3 = tf.nn.relu(tf.add(tf.matmul(output2, A3), b3))
#output3 = tf.matmul(output2, A3) # Output 3 without any activation and bias
#output4 = tf.nn.relu(tf.add(tf.matmul(output3, A4), b4))
output4 =((tf.matmul(output3, A4)))
#output5 = tf.nn.relu(tf.add(tf.matmul(output4, A5),b5))  #Output 5 gives the predictions
#output6=tf.nn.relu(tf.matmul(output5,A6))
#Defining loss function


#Change the output if you change number of hidden layers
loss=tf.reduce_mean(tf.squared_difference(output4,y_data))
my_opt=tf.train.AdamOptimizer(learningRate)

#I am not defining dropouts as of now

#saver = tf.train.Saver()


train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



#Initializing the loss vectors
loss_vec=[]
test_loss=[]


#Add list for looping through

loop_index=[0,4,6,8,12,14,16]
#loop_index=[14,16]
name_list=["Andy","Edgar","Ivan","Mayank","Mohit","Taa"]
folder_path="/home/vaneet/PPG_Experiment1_June30/Predicting_heartrate/7Sep/200iter/"
min_loss_vec=[]

n_row_x=(x_vals).shape[0]
n_row_y=(y_vals).shape[0]
percentage_error_MPE = []
percentage_error_RMSE=[]
total_prediction=[]
total_predictions_train=[]
percentage_error_MPE_train=[]
percentage_error_RMSE_train=[]
print n_row_x==n_row_y
if n_row_x==n_row_y:
    for i in range(0,len(loop_index)-1): #This for loop is doing the job of cross validation thing



        # init = tf.global_variables_initializer()
        # sess = tf.Session()
        # sess.run(init)

        start=loop_index[i]
        end=loop_index[i+1]


        #By Person

        # x_vals_train = x_vals[10:160,:]
        x_vals_test = x_vals[start:end, ]
        x_vals_train = np.delete(x_vals, np.s_[start:end], 0)
        # y_vals_train = y_vals[10:160,:]
        y_vals_test = y_vals[start:end, ]
        y_vals_train = np.delete(y_vals, np.s_[start:end], 0)

        epoch = 1
        rate = 0.00001
        num_iter = 200
        for l in range(num_iter):
            batch_size = int(np.random.uniform(2,12, 1))  # should never be bigger than number of rows.

            # Choose random indices for batch selection
            rand_index = np.random.choice(len(x_vals_train), size=batch_size)
            # Get random batch
            rand_x = x_vals_train[rand_index]
            rand_y = y_vals_train[rand_index]



            if l % 100 == 0:
                epoch += 1.0
                rate = 0.00001 / epoch

                # Run the training step
            sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y, learningRate: rate})
            # Get and store the train loss
            temp_loss = sess.run(loss, feed_dict={x_data: x_vals_train, y_data: y_vals_train})
            loss_vec.append(temp_loss)
            # Get and store the test lossfrom __future__ import division
            test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_data: y_vals_test})
            test_loss.append(test_temp_loss)
            print (l)
            print(temp_loss)
            print (test_temp_loss)

            # print A1
            # print A2
            # print A3
            # print A4



        a = plt.figure(1)

        plt.plot(loss_vec, 'k-', label='Train Loss')
        plt.plot(test_loss, 'r--', label='Test Loss')
        plt.title('Loss per iteration')

        plt.xlabel('Iteration')
        plt.ylabel('Loss values')
        plt.legend(loc='upper right')
        print i
        a.savefig(folder_path + name_list[i] + "_Loss2.eps", format='eps', dpi=300)

        np.savetxt(
            folder_path +name_list[i]+ 'train_loss.csv',
            loss_vec, delimiter=',')

        np.savetxt(
            folder_path +name_list[i]+ 'test_loss.csv',
            test_loss, delimiter=',')
        #a.show()
        a.clear()


        min_loss_vec.append(min(test_loss))

        loss_vec = []
        test_loss = []

        predictions = sess.run(output4, feed_dict={x_data: x_vals_test})
        predictions_train=sess.run(output4, feed_dict={x_data:x_vals_train})
        #print (predictions)
        #print(y_vals_test)
        # raw_input()

        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []
        # Mean percent error and RMSE
        for i in range(len(predictions)):
            my_val1 = ((predictions[i] - y_vals_test[i]) ** 2)
            my_val2 = abs(predictions[i] - y_vals_test[i]) / y_vals_test[i]
            temp1.append(my_val1)
            print('RATIO SPECS:')
            print predictions[i]
            print y_vals_test[i]
            temp2.append(my_val2)
            total_prediction.append(predictions[i])

        for i in range(len(predictions_train)):
            my_val3 = ((predictions_train[i] - y_vals_train[i]) ** 2)
            my_val4 = abs(predictions_train[i] - y_vals_train[i]) / y_vals_train[i]
            temp3.append(my_val3)
            temp4.append(my_val4)
            total_predictions_train.append(predictions_train[i])

        perc_error_RMSE = math.sqrt(math.fsum(temp1) / len(predictions))


        perc_error_MPE = (100 / len(predictions)) * math.fsum(temp2)

        perc_error_RMSE_train = math.sqrt(math.fsum(temp3) / len(predictions_train))

        perc_error_MPE_train = (100 / len(predictions_train)) * math.fsum(temp4)

        print "The mean percentage error is: " + str(perc_error_MPE)
        print "THe RMSE IS: " + str(perc_error_RMSE)
        percentage_error_MPE.append(perc_error_MPE)
        percentage_error_RMSE.append(perc_error_RMSE)

        percentage_error_MPE_train.append(perc_error_MPE_train)
        percentage_error_RMSE_train.append(perc_error_RMSE_train)


    print percentage_error_MPE, percentage_error_RMSE
    print percentage_error_MPE_train,percentage_error_RMSE_train



    np.savetxt(folder_path+'prediction_values.csv', total_prediction, delimiter=',')

    np.savetxt(folder_path + 'prediction_values_train.csv',
        total_predictions_train, delimiter=',')

    np.savetxt(folder_path +'Percentage errors_MPE.csv',
            percentage_error_MPE, delimiter=',')

    np.savetxt(folder_path + 'Percentage errors_RMSE.csv',
            percentage_error_RMSE, delimiter=',')

    np.savetxt(folder_path + 'Percentage errors_MPE_train.csv',
        percentage_error_MPE_train, delimiter=',')

    np.savetxt(folder_path + 'Percentage errors_RMSE_train.csv',
        percentage_error_RMSE_train, delimiter=',')
        #By Activity

        # span=(end-start)//10
        #
        # test_start=start
        #
        #
        # for j in range(span):
        #
        #     print j
        #
        #     # x_vals_train = x_vals[10:160,:]
        #     x_vals_test = x_vals[test_start:test_start+10, ]
        #     x_vals_train = np.delete(x_vals, np.s_[start:end], 0)
        #     # y_vals_train = y_vals[10:160,:]
        #     y_vals_test = y_vals[test_start:test_start+10, ]
        #     y_vals_train = np.delete(y_vals, np.s_[start:end], 0)
        #
        #     test_start=test_start+10
        #
        #     epoch = 1
        #     rate = 0.00001
        #     num_iter=1000
        #     for l in range(num_iter):
        #         batch_size = int(np.random.uniform(5, 100, 1))  # should never be bigger than number of rows.
        #
        #         # Choose random indices for batch selection
        #         rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        #         # Get random batch
        #         rand_x = x_vals_train[rand_index]
        #         rand_y = y_vals_train[rand_index]
        #
        #         if l % 100 == 0:
        #             epoch += 1.0
        #             rate = 0.00001 / epoch
        #
        #             # Run the training step
        #         sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y, learningRate: rate})
        #         # Get and store the train loss
        #         temp_loss = sess.run(loss, feed_dict={x_data: x_vals_train, y_data: y_vals_train})
        #         loss_vec.append(temp_loss)
        #         # Get and store the test lossfrom __future__ import division
        #         test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_data: y_vals_test})
        #         test_loss.append(test_temp_loss)
        #         print (l)
        #         print(temp_loss)
        #         print (test_temp_loss)
        #         # if i==99999:
        #         #     save_path = saver.save(sess, "/home/vaneet/Desktop/.ckpt")
        #
        #     a = plt.figure(1)
        #     # a=plt.figure(i)
        #     plt.plot(loss_vec[num_iter//40:], 'k-', label='Train Loss')
        #     plt.plot(test_loss[num_iter//40:], 'r--', label='Test Loss')
        #     plt.title('Loss per Generation')
        #
        #     plt.xlabel('Generation')
        #     plt.ylabel('Loss')
        #     plt.legend(loc='upper right')
        #     print i
        #     a.savefig("/home/vaneet/PPG_Experiment1_June30/Predicting_heartrate/04Aug/By_Activity_1000iter/" + name_list[i]+"_"+str(j)+"_Loss2.png")
        #     a.clear()
        #
        #     min_loss_vec.append(min(test_loss))
        #
        #     loss_vec = []
        #     test_loss = []
        #
        #     predictions = sess.run(output4, feed_dict={x_data: x_vals_test})
        #     # print (predictions)
        #     # print(y_vals_test)
        #     # raw_input()
        #
        #     temp = []
        #     temp2=[]
        #     for ii in range(len(predictions)):
        #         my_val = (((predictions[ii] - y_vals_test[ii]) ** 2) / (y_vals_test[ii]) ** 2)
        #         temp.append(my_val)
        #         temp2.append((y_vals_test[ii]) ** 2)
        #
        #     perc_error = (100 / len(predictions)) * math.sqrt(math.fsum(temp))
        #     print "The mean percentage error is: " + str(perc_error)
        #     percentage_error.append(perc_error)
        #     print percentage_error
        #     print min_loss_vec
        #     np.savetxt('/home/vaneet/PPG_Experiment1_June30/Predicting_heartrate/04Aug/By_Activity_1000iter/Percentage errors.csv',
        #                percentage_error, delimiter=',')


#For taking the mean of the values
# final_prediction=[]
# final_original=[]
# k=1
# for i in range(0,1800,k):
#     temp_final_predictions=np.mean(predictions[i:(i+k)])
#     final_prediction.append(temp_final_predictions)
#     temp_original=np.mean(y_vals_test[i:(i+k)])
#     final_original.append(temp_original)
# print(len(final_prediction))
# print(len(final_original))



##FFT for both actual and predicted

#Power spectrum
#
# rate = 30.0
# #x = np.sin(2*np.pi*4*t) + np.sin(2*np.pi*7*t) + np.random.randn(len(t))*0.2
# p_spec = 20*np.log10(np.abs(np.fft.rfft(y_vals_test)))
# f = np.linspace(0, rate/2, len(p_spec))

# plt.plot(f, p_spec)
# plt.show()
#
# print predictions
# f, Pxx_den=signal.periodogram(predictions,fs=30)
# print f
# print len(f)
#
# print Pxx_den
# print len(Pxx_den)

# plt.semilogy(f, Pxx_den)
#
# #plt.ylim([1e-7, 1e2])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()


# ps = np.abs(np.fft.rfft(y_vals_test))**2
# time_step = 1 / 30
# freqs= np.fft.rfftfreq(y_vals_test.size, time_step)
# idx = np.argsort(freqs)
#
# plt.plot(freqs[idx], ps[idx])
# plt.show()
""""
fs = 30 # 1 ns -> 1 GHz
cutoff =2 # 10 MHz
B, A = butter(1, cutoff / (fs / 2), btype='low') # 1st order Butterworth low-pass
filtered_signal = lfilter(B, A, predictions, axis=0)
#
b=plt.figure('Taa Pushup_filtered')# ps = np.abs(np.fft.rfft(y_vals_test))**2
# time_step = 1 / 30
# freqs= np.fft.rfftfreq(y_vals_test.size, time_step)
# idx = np.argsort(freqs)
#
plt.plot(filtered_signal[50:350])
b.show()


# plt.plot(abs(predicted_fft))
# b.show()
#
c=plt.figure('Taa Pushup1')
plt.plot(y_vals_test[50:350])
c.show()
d=plt.figure('Taa Pushup1.predicted')
plt.plot(predictions[50:350])
d.show()

#
# d=plt.figure('Taa Still')
# plt.plot(predictions[2000:2300])
# d.show()
#
# e=plt.figure('Taa still1')
# plt.plot(y_vals_test[2000:2300])
# e.show()

raw_input()

# SSE=math.fsum((predictions-y_vals_test)**2)
# RMSE=math.sqrt(SSE/len(y_vals_test))
# Error_percentage=(SSE/(math.fsum(y_vals_test)**2))*100
#
# print(Error_percentage)
"""
