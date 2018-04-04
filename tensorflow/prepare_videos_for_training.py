import os
import tensorflow as tf, sys
import numpy as np
import cv2


WIDTH, HEIGHT = 224, 224



def get_immediate_subdirectories(a_dir):
    return [name for name in sorted(os.listdir(a_dir))
            if os.path.isdir(os.path.join(a_dir, name))]

def import_single_frame(image_path):
    WIDTH, HEIGHT = 224, 224
    image = cv2.imread(image_path)
    if cv2.countNonZero(cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )) == 0:
        return False
    image = cv2.resize(image,  (WIDTH, HEIGHT))
    return image

def load_netModel(path_model):
    print("loadin model")
    
    # Unpersists graph from file
    with tf.gfile.FastGFile("pre_trained_models/standalone.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        return graph_def


path= 'video_frames'
output_path = 'video_converted_to_numpy/'
net_model_path = "pre_trained_models/standalone.pb"

graph_def = load_netModel(net_model_path)




video_directories_list = get_immediate_subdirectories(path)

for video_dir in video_directories_list:
    print('working on video '+ video_dir +'...')
    with tf.Session() as sess:
        fullyConn_tensor = tf.get_default_graph().get_tensor_by_name("fc7/fc7:0")#final_result
        video_np_array= []
        
        for frame in sorted(os.listdir(path +'/'+ video_dir)):

            img = import_single_frame(path +'/'+ video_dir+'/'+frame)
            if isinstance(img, (list,np.ndarray)):
                predictions = sess.run(fullyConn_tensor,    {'input:0': [img]})
                video_np_array.append(predictions)

                #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))
        video_np_array= np.array(video_np_array)
        np.save(output_path + video_dir[:-8],video_np_array)
        print(video_np_array.shape)
          
        
        
        
        

    
