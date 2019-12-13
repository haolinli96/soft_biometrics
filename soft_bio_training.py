import  tensorflow as tf
import  data_prep as train_data
import  cv2
import  random
import  matplotlib.pyplot as plt
import  numpy as np
from sklearn.metrics import confusion_matrix, classification_report

train_epochs=4000
batch_size = 9  #batch
drop_prob = 0.35 #dropout probability
learning_rate=0.00001


# shape is a list 
# [3, 3, 3, 32] means 3 * 3 kernal input 3 chanel(like rgb image) output 32 chanel 
# initialize weight
def weight_init(shape):
    weight = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weight)

# initialize bias
def bias_init(shape):
    bias = tf.random_normal(shape, dtype=tf.float32)
    return tf.Variable(bias)

# input images with size 224*224
# input labels of race 1-asian 2-black 3-caucasian 4-hispanic
images_input = tf.placeholder(tf.float32, [None,224*224*3], name='input_images')
labels_input = tf.placeholder(tf.float32, [None,2], name='input_labels')

# initialize fully connected layer !!!!!!!!!!!!!!!!!!!!!!!!!
def fch_init(layer1,layer2,const=1):
    min = -const * (6.0 / (layer1 + layer2));
    max = -min;
    weight = tf.random_uniform([layer1, layer2], minval=min, maxval=max, dtype=tf.float32)
    return tf.Variable(weight)

def conv2d(images,weight,stride):
    return tf.nn.conv2d(images, weight, strides=[1,stride,stride,1], padding='SAME')

def depthwise_conv2d(images, weight, stride):
    # returns NHWC
    return tf.nn.depthwise_conv2d(images, weight, strides=[1, stride, stride, 1], padding='SAME')

def avg_pool7x7(images,tname):
    # 2d??
    return tf.nn.avg_pool(images, ksize=[1,7,7,1], strides=[1,7,7,1], padding='SAME', name=tname)

def bn_relu(images, tname):
    bn = tf.layers.batch_normalization(images, training=True)
    relu = tf.nn.relu(bn, name=tname)
    return relu
# reshape image to 224*224*3
x_input = tf.reshape(images_input, [-1,224,224,3])

def MBNet(images, cin, cout, stride, tname):
    # input images is NHWC why 1024?
    dw_w = weight_init([3,3,cin,1]) # last dimension is channel multiplier here
    dw_b = bias_init([cin])
    conv_w = weight_init([1,1,cin,cout])
    conv_b = bias_init([cout])
    dw_conv = depthwise_conv2d(images, dw_w, stride) + dw_b
    bn_relu_m = bn_relu(dw_conv, tname+'m')
    conv_f = conv2d(bn_relu_m, conv_w, 1) + conv_b
    bn_relu_f = bn_relu(conv_f, tname+'f')
    return bn_relu_f


# layer1 input 224 * 224 * 3
# 32 filters, each 3 * 3 * 3
w1 = weight_init([3,3,3,32])
b1 = bias_init([32])
# output NHWC  N Height Width Channel
conv_1 = conv2d(x_input, w1, 2)+b1     #strides
bn_relu_1 = bn_relu(conv_1, 'bn_relu_1')
#max_pool_1 = max_pool2x2(relu_1,'max_pool_1')

# layer2 input 112 * 112 * 32
# 32 dw filters, each 3 * 3 * 32
# 64 conv filters each 1 * 1 * 32
layer_2 = MBNet(bn_relu_1, 32, 64, 1, 'bn_relu_2')

# layer3 input 112 * 112 * 64
layer_3 = MBNet(layer_2, 64, 128, 2, 'bn_relu_3')
layer_3_2 = MBNet(layer_3, 128, 128, 1, 'bn_relu_3_2')

# layer4 input 56 * 56 * 128
layer_4 = MBNet(layer_3_2, 128, 256, 2, 'bn_relu_4')
layer_4_2 = MBNet(layer_4, 256, 256, 1, 'bn_relu_4_2')

# layer5 input 28 * 28 * 256 
layer_5 = MBNet(layer_4_2, 256, 512, 2, 'bn_relu_5')
layer_5_2 = MBNet(layer_5, 512, 512, 1, 'bn_relu_5_2')
layer_5_3 = MBNet(layer_5_2, 512, 512, 1, 'bn_relu_5_3')
layer_5_4 = MBNet(layer_5_3, 512, 512, 1, 'bn_relu_5_4')
layer_5_5 = MBNet(layer_5_4, 512, 512, 1, 'bn_relu_5_5')
layer_5_6 = MBNet(layer_5_5, 512, 512, 1, 'bn_relu_5_6')

# layer6 input 14 * 14 * 512
layer_6 = MBNet(layer_5_6, 512, 1024, 2, 'bn_relu_6')
layer_6_2 = MBNet(layer_6, 1024, 1024, 1, 'bn_relu_6_2')

# average pool input 7 * 7 * 1024
# output 1* 1* 1024
avg_pool_7 = avg_pool7x7(layer_6_2, 'avg_pool_7')


# fully connected layers
# reshape it to 1 dimension
f_input = tf.reshape(avg_pool_7,[-1,1*1*1024])

# fc1 1024 to 256
f_w1= fch_init(1024, 256)
f_b1 = bias_init([256])
f_r1 = tf.matmul(f_input,f_w1) + f_b1
f_relu_r1 = tf.nn.relu(f_r1)
f_dropout_r1 = tf.nn.dropout(f_relu_r1,drop_prob)

# fc2 256 to 2/4 output
f_w2 = fch_init(256,2)
f_b2 = bias_init([2])
f_r2 = tf.matmul(f_dropout_r1,f_w2) + f_b2

f_softmax = tf.nn.softmax(f_r2,name='f_softmax')


# crossentrophy
cross_entry =  tf.reduce_mean(tf.reduce_sum(-labels_input*tf.log(f_softmax)))
optimizer  = tf.train.AdamOptimizer(learning_rate).minimize(cross_entry)

#calculating preccision & loss
arg1 = tf.argmax(labels_input,1)
arg2 = tf.argmax(f_softmax,1)
cos = tf.equal(arg1,arg2)
acc = tf.reduce_mean(tf.cast(cos,dtype=tf.float32))




# start
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)




Cost = []
Accuracy=[]
for i in range(train_epochs):
    idx=random.randint(0,len(train_data.images)-20)
    batch= random.randint(6,18)
    train_input = train_data.images[idx:(idx+batch)]
    train_labels = train_data.labels[idx:(idx+batch)]
    result,acc1,cross_entry_r,cos1,f_softmax1,bn_relu_1_r= sess.run([optimizer,acc,cross_entry,cos,f_softmax,bn_relu_1],feed_dict={images_input:train_input,labels_input:train_labels})
    print (acc1)
    Cost.append(cross_entry_r)
    Accuracy.append(acc1)

# loss function curve
fig1,ax1 = plt.subplots(figsize=(10,7))
plt.plot(Cost)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cost')
plt.title('Cross Loss')
plt.grid()
plt.show()

# precission curve
fig7,ax7 = plt.subplots(figsize=(10,7))
plt.plot(Accuracy)
ax7.set_xlabel('Epochs')
ax7.set_ylabel('Accuracy Rate')
plt.title('Train Accuracy Rate')
plt.grid()
plt.show()


# run test
arg2_r = sess.run(arg2,feed_dict={images_input:train_data.test_images,labels_input:train_data.test_labels})
arg1_r = sess.run(arg1,feed_dict={images_input:train_data.test_images,labels_input:train_data.test_labels})

print (classification_report(arg1_r, arg2_r))

# save model
saver = tf.compat.v1.train.Saver()
saver.save(sess, './model/MBNet-gender-v2.0')