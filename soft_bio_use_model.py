# using existing model
import tensorflow as tf
import numpy as np
import  data_prep as train_data
import  matplotlib.pyplot as plt
import  os

model_name = "MBNet-gender-v1.0"

np.set_printoptions(suppress=True)

# get a testing image
input_image = train_data.images[0:1]
labels = train_data.labels[0:1]
fig2,ax2 = plt.subplots(figsize=(2,2))
ax2.imshow(np.reshape(input_image, (224, 224, 3)))
plt.savefig('input.png')


sess = tf.Session()
graph_path=os.path.abspath("model/" + model_name + '.meta')
model=os.path.abspath('model')

server = tf.train.import_meta_graph(graph_path)
server.restore(sess,tf.train.latest_checkpoint(model))

graph = tf.get_default_graph()

# feed_dict
x = graph.get_tensor_by_name('input_images:0')
y = graph.get_tensor_by_name('input_labels:0')
feed_dict={x:input_image,y:labels}


# layer1
bn_relu_1 = graph.get_tensor_by_name('bn_relu_1:0')

# layer_2
bn_relu_2m = graph.get_tensor_by_name('bn_relu_2m:0')
bn_relu_2f = graph.get_tensor_by_name('bn_relu_2f:0')

# layer_3
bn_relu_3m = graph.get_tensor_by_name('bn_relu_3m:0')
bn_relu_3f = graph.get_tensor_by_name('bn_relu_3f:0')
bn_relu_3_2m = graph.get_tensor_by_name('bn_relu_3_2m:0')
bn_relu_3_2f = graph.get_tensor_by_name('bn_relu_3_2f:0')

# layer_4
bn_relu_4m = graph.get_tensor_by_name('bn_relu_4m:0')
bn_relu_4f = graph.get_tensor_by_name('bn_relu_4f:0')
bn_relu_4_2m = graph.get_tensor_by_name('bn_relu_4_2m:0')
bn_relu_4_2f = graph.get_tensor_by_name('bn_relu_4_2f:0')

# layer_5
bn_relu_5m = graph.get_tensor_by_name('bn_relu_5m:0')
bn_relu_5f = graph.get_tensor_by_name('bn_relu_5f:0')
bn_relu_5_2m = graph.get_tensor_by_name('bn_relu_5_2m:0')
bn_relu_5_2f = graph.get_tensor_by_name('bn_relu_5_2f:0')
bn_relu_5_3m = graph.get_tensor_by_name('bn_relu_5_3m:0')
bn_relu_5_3f = graph.get_tensor_by_name('bn_relu_5_3f:0')
bn_relu_5_4m = graph.get_tensor_by_name('bn_relu_5_4m:0')
bn_relu_5_4f = graph.get_tensor_by_name('bn_relu_5_4f:0')
bn_relu_5_5m = graph.get_tensor_by_name('bn_relu_5_5m:0')
bn_relu_5_5f = graph.get_tensor_by_name('bn_relu_5_5f:0')
bn_relu_5_6m = graph.get_tensor_by_name('bn_relu_5_6m:0')
bn_relu_5_6f = graph.get_tensor_by_name('bn_relu_5_6f:0')

# layer_6
bn_relu_6m = graph.get_tensor_by_name('bn_relu_6m:0')
bn_relu_6f = graph.get_tensor_by_name('bn_relu_6f:0')
bn_relu_6_2m = graph.get_tensor_by_name('bn_relu_6_2m:0')
bn_relu_6_2f = graph.get_tensor_by_name('bn_relu_6_2f:0')

# average pool 7
avg_pool_7 = graph.get_tensor_by_name('avg_pool_7:0')

# last fully connected output
f_softmax = graph.get_tensor_by_name('f_softmax:0')


#relu_1_r,max_pool_1_,relu_2,max_pool_2,relu_3,max_pool_3,f_softmax=sess.run([relu_1,max_pool_1,relu_2,max_pool_2,relu_3,max_pool_3,f_softmax],feed_dict)



#----------------------------------visualize-------------------------------


# layer1 visualization
r_bn_relu_1 = sess.run(bn_relu_1, feed_dict)
r_tranpose_1 = sess.run(tf.transpose(r_bn_relu_1,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_1[i][0])
plt.title('Conv1 32*112*112')
plt.savefig('Conv1.png')

# layer2 visualization
r_bn_relu_2m = sess.run(bn_relu_2m, feed_dict)
r_tranpose_2m = sess.run(tf.transpose(r_bn_relu_2m,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_2m[i][0])
plt.title('Depthwise_Conv2 32*112*112')
plt.savefig('Depthwise_Conv2.png')
r_bn_relu_2f = sess.run(bn_relu_2f, feed_dict)
r_tranpose_2f = sess.run(tf.transpose(r_bn_relu_2f,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_2f[i][0])
plt.title('1*1_Conv2 64*112*112')
plt.savefig('Conv2.png')

# layer3 first MBNet visualization
r_bn_relu_3m = sess.run(bn_relu_3m, feed_dict)
r_tranpose_3m = sess.run(tf.transpose(r_bn_relu_3m,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_3m[i][0])
plt.title('Depthwise_Conv3 64*56*56')
plt.savefig('Depthwise_Conv3.png')
r_bn_relu_3f = sess.run(bn_relu_3f, feed_dict)
r_tranpose_3f = sess.run(tf.transpose(r_bn_relu_3f,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_3f[i][0])
plt.title('1*1_Conv3 128*56*56')
plt.savefig('Conv3.png')

# layer4 first MBNet visualization
r_bn_relu_4m = sess.run(bn_relu_4m, feed_dict)
r_tranpose_4m = sess.run(tf.transpose(r_bn_relu_4m,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_4m[i][0])
plt.title('Depthwise_Conv4 128*28*28')
plt.savefig('Depthwise_Conv4.png')
r_bn_relu_4f = sess.run(bn_relu_4f, feed_dict)
r_tranpose_4f = sess.run(tf.transpose(r_bn_relu_4f,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_4f[i][0])
plt.title('1*1_Conv4 256*28*28')
plt.savefig('Conv4.png')


# layer5 first MBNet visualization
r_bn_relu_5m = sess.run(bn_relu_5m, feed_dict)
r_tranpose_5m = sess.run(tf.transpose(r_bn_relu_5m,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_5m[i][0])
plt.title('Depthwise_Conv5 256*14*14')
plt.savefig('Depthwise_Conv5.png')
r_bn_relu_5f = sess.run(bn_relu_5f, feed_dict)
r_tranpose_5f = sess.run(tf.transpose(r_bn_relu_5f,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_5f[i][0])
plt.title('1*1_Conv5 512*14*14')
plt.savefig('Conv5.png')

# layer6 first MBNet visualization
r_bn_relu_6m = sess.run(bn_relu_6m, feed_dict)
r_tranpose_6m = sess.run(tf.transpose(r_bn_relu_6m,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_6m[i][0])
plt.title('Depthwise_Conv6 512*7*7')
plt.savefig('Depthwise_Conv6.png')
r_bn_relu_6f = sess.run(bn_relu_6f, feed_dict)
r_tranpose_6f = sess.run(tf.transpose(r_bn_relu_6f,[3,0,1,2]))
fig, ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_6f[i][0])
plt.title('1*1_Conv6 1024*7*7')
plt.savefig('Conv6.png')


# 7 * 7 pool
r_avg_pool_7 = sess.run(avg_pool_7,feed_dict)
r_tranpose_7 = sess.run(tf.transpose(r_avg_pool_7,[3,0,1,2]))
fig,ax = plt.subplots(nrows=1,ncols=16,figsize=(16,1))
for i in range(16):
    ax[i].imshow(r_tranpose_7[i][0])
plt.title('7*7_Avg_Pool7 1024*1*1')
plt.savefig('Avg_Pool.png')


print (sess.run(f_softmax,feed_dict))