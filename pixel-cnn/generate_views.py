"""
__author__ = Sohaib Kiani
August 2021
"""

import fast_pixel_cnn_pp.model as model
import fast_pixel_cnn_pp.fast_nn as fast_nn
import fast_pixel_cnn_pp.plotting as plotting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import argparse
import time
import os
import json
import math
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    default=100,
    help='Number of images to generate simultaneously')
parser.add_argument(
    '-i',
    '--image_size',
    type=int,
    default=32,
    help='Height and width of the image')
parser.add_argument(
    '-l',
    '--num_labels',
    type=int,
    default=16,
    help='Height and width of the image')
parser.add_argument(
    '-g',
    '--nr_gpu',
    type=int,
    default=1,
    help='No. of gpu')
parser.add_argument(
    '-s', '--seed', type=int, default=2702, help='Seed for random generation')
parser.add_argument(
    '-c',
    '--checkpoint',
    type=str,
    default='nothing',
    help='Location of the pretrained checkpoint')
parser.add_argument(
    '-v',
    '--save_dir',
    type=str,
    default='out/',
    help='Location to save generated images to')
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    default='imagenet',
    help='Location to save generated images to')
parser.add_argument(
    '-n',
    '--num_samples',
    type=int,
    default=-1,
    help='Number of Samples')
args = parser.parse_args()

N_gen=4
data_dir = os.path.join( os.getcwd(), f'data/{args.dataset}' )
args.checkpoint = f"Model_{args.dataset}/params_{args.dataset}.ckpt"
if args.dataset == 'imagenet':
    args.num_labels = 16
if args.dataset == 'cifar':
    args.num_labels = 10
if args.dataset == 'gtsrb':
    args.num_labels = 43
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

def load_data(file):
    ndata = {}
    if args.dataset == 'imagenet':
        if file != 'test_c':
            data_dir_a = os.path.join(data_dir,'Adversarial')
            data = pickle.load(open(f'{data_dir_a}/{file}', mode='rb'))
            ndata['x'] = data['data']
            ndata['y'] = data['tlabel']
            ndata['label'] = data['alabel']
        else:
            data = pickle.load(open(f'{data_dir}/{file}', mode='rb'))
            ndata['x'] = data['x']
            ndata['y'] = data['y']
            ndata['label'] = data['y']
    elif args.dataset == 'cifar':
        if file != 'test_c':
            data_dir_a = os.path.join(data_dir,'Adversarial')
            data = pickle.load(open(f'{data_dir_a}/{file}', mode='rb'))
            ndata['x'] = np.squeeze(data['data'])
            ndata['y'] = np.squeeze(data['tlabel'])
            ndata['label'] = np.squeeze(data['alabel'])
        else:
            data = pickle.load(open(f'{data_dir}/{file}', mode='rb'))
            ndata['x'] = data['x']
            ndata['y'] = data['y']
            ndata['label'] = data['y']
    elif args.dataset == 'gtsrb':
        if file != 'test_c':
            data_dir_a = os.path.join(data_dir,'Adversarial')
            data = pickle.load(open(f'{data_dir_a}/{file}', mode='rb'))
            ndata['x'] = data['data']
            ndata['y'] = data['tlabel']
            ndata['label'] = data['alabel']
        else:
            data = pickle.load(open(f'{data_dir}/{file}', mode='rb'))
            ndata['x'] = data['x']
            ndata['y'] = data['y']
            ndata['label'] = data['y']
    if args.num_samples == -1 :
        args.num_samples = int(ndata['x'].shape[0]/args.batch_size) * args.batch_size

    ndata['x'] = np.cast[np.float32]((ndata['x'] - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    print(ndata.keys())
    print ("Range of data should be -1-1 and actual is: ",str(np.min(ndata['x']))+" "+str(np.max(ndata['x'])))
    print (ndata['x'].shape)
    print ("Generating Number of Samples",args.num_samples)
    assert ndata['x'].shape[0] >= args.num_samples
    assert np.max(ndata['x']) <=1
    return ndata

def init_seed(f_index,data):
    ##### New Stuff #####
    obs_shape = data['x'].shape[1:]
    x_gen = np.zeros((args.batch_size * N_gen ,) + obs_shape, dtype=np.float32)
    lab = data['label'][f_index:f_index+args.batch_size]
    #X_gen contain 12 copies of each image. Those 12 copies are not consecutive
    #First batch_size=16 image will be first 16 samples of x_gen, and this will concatenated N_gen times
    # The next gpu will contain same order with different set og images
    index=0
    t_index=f_index
    r_valid = [8,16,24,8]
    for i in range(0,N_gen):
        for row in range(0,32):
            x_gen[index:index+args.batch_size,row,:,:] = data['x'][t_index:t_index + args.batch_size,row,:,:]
        index += args.batch_size
    return x_gen,lab

def generate_Image_rowWise(lab,output_images,start_r):
    starting_row = 0
    if starting_row < 0:
        starting_row = 0
    for row in range(starting_row,args.image_size):
        # Implicit downshift.
        if row > start_r + 8:
            break
        if row == 0:
            x_row_input = np.zeros((args.batch_size, 1, args.image_size, input_channels))
            #x_row_input[:,:,:,:3] =  output_images[:, (row):row+1, :, :]
        else:
            x_row_input = output_images[:, (row - 1):row, :, :]
            x_row_input = np.concatenate(
                (x_row_input, np.ones(
                    (args.batch_size, 1, args.image_size, 1))),
                axis=3)
        sess.run(v_stack, {row_input: x_row_input, row_id: row,ys:lab})
        for col in range(0,args.image_size):
            # Implicit rightshift.
            if col == 0:
                x_pixel_input = np.zeros(
                    (args.batch_size, 1, 1, input_channels))
            else:
                x_pixel_input = output_images[:, row:(row + 1),(col - 1):col, :]
                x_pixel_input = np.concatenate(
                    (x_pixel_input, np.ones((args.batch_size, 1, 1, 1))),
                    axis=3)
            feed_dict = {
                row_id: row,
                col_id: col,
                pixel_input: x_pixel_input,
                ys:lab
            }
            pixel_output = sess.run(sample, feed_dict)
            if row >= start_r:
                output_images[:, row:(row + 1),col:(col + 1), :] = pixel_output

g = tf.compat.v1.Graph()
with g.as_default():
    print('Creating model')
    input_channels = 4  # 3 channels for RGB and 1 channel of all ones
    image_size = (args.batch_size, args.image_size, args.image_size,input_channels)

    row_input = tf.compat.v1.placeholder(
        tf.compat.v1.float32, [args.batch_size, 1, args.image_size, input_channels],
        name='row_input')
    pixel_input = tf.compat.v1.placeholder(
        tf.compat.v1.float32, [args.batch_size, 1, 1, input_channels],
        name='pixel_input')
    row_id = tf.compat.v1.placeholder(tf.compat.v1.int32, [], name='row_id')
    col_id = tf.compat.v1.placeholder(tf.compat.v1.int32, [], name='col_id')
    ema = tf.compat.v1.train.ExponentialMovingAverage(0.9995)

    ys = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=(args.batch_size,))
    hs = tf.compat.v1.one_hot(ys, args.num_labels)

    model_spec = tf.compat.v1.make_template('model', model.model_spec)
    sample, fast_nn_out, v_stack = model_spec(
        row_input, pixel_input, row_id, col_id, image_size,hs, seed=args.seed)

    all_cache_variables = [
        v for v in tf.compat.v1.global_variables() if 'cache' in v.name
    ]
    initialize_cache = tf.compat.v1.variables_initializer(all_cache_variables)
    reset_cache = fast_nn.reset_cache_op()

    vars_to_restore = {
        k: v
        for k, v in ema.variables_to_restore().items() if 'cache' not in k
    }
    saver = tf.compat.v1.train.Saver(vars_to_restore)
    sess = tf.compat.v1.Session()
    sess.run(initialize_cache)
    print('Loading checkpoint %s' % args.checkpoint)
    saver.restore(sess, args.checkpoint)

    file = ['test_c','pgd_E4','pgd_E8','pgd_E16','fgsm_E4','fgsm_E8','fgsm_E16','mim_E4','mim_E8','mim_E16','deepFool','CW']
    for filename in file:
        print (data_dir)
        data = load_data(filename)
        sample_x = []
        print ("Total Iterations:",args.num_samples/args.batch_size)
        for i in range(0,args.num_samples,args.batch_size):
            print ("Sampled: ",i)
            x,lab = init_seed(i,data)
            num_eval_examples = x.shape[0]
            eval_batch_size = args.batch_size
            batch = 0
            print('Generating')
            sess.run(reset_cache)
            start_time = time.time()
            #Generating first three quarters
            start_r = [8,16,24]
            for g in range(0,3):
                print (lab)
                generate_Image_rowWise(lab, x[g*args.batch_size:(g+1)*args.batch_size],start_r[g])
                sess.run(reset_cache)
            step = [8,16,24,33]
            for g in range(3):
                print (g,flush=True)
                x[args.batch_size * 3: args.batch_size * 4,step[g]:step[g+1],:,:] = x[g*args.batch_size:(g+1)*args.batch_size,step[g]:step[g+1],:,:]
            sample_x.append(x)
            end_time = time.time()
            print('Time taken to generate %d images: %.2f seconds' %
                  (args.batch_size, end_time - start_time))
            batch += 1
        new_x = []
        sample_x = np.concatenate(sample_x,axis=0)
        sample_x = np.asarray(sample_x)
        print (sample_x.shape)
        track_sample_gpu=0
        gpu_sample = 1
        k=0
        for i in range(args.num_samples):
            new_x.append(data['x'][i,:,:,:])
            for g in range(N_gen):
                new_x.append(sample_x[k+(g*args.batch_size),:,:,:])
            k+=1
            if k == track_sample_gpu + args.batch_size:
                k = gpu_sample * args.batch_size * N_gen
                track_sample_gpu = k
                gpu_sample += 1
        new_x=np.asarray(new_x)
        print (np.shape(new_x))

        img_tile1 = plotting.img_tile(new_x[:10*(N_gen+1)], aspect_ratio=1.0,tile_shape=[10,N_gen+1], border_color=1.0, stretch=True)
        plotting.plot_img(img_tile1)
        if not os.path.exists("out/"):
            os.makedirs(f"out/")
        plotting.plt.savefig(os.path.join(args.save_dir, f'{filename}.png'))
        plotting.plt.close('all')
        out_data = {'x':new_x,'y':data['y'],'y_adv':data['label']}
        save_file = os.path.join(data_dir,'Generated')
        if not os.path.exists(save_file):
            print(f"Creating Directory {save_file}")
            os.makedirs(save_file)
        print ('Saving',f'{save_file}/g_{filename}')
        pickle.dump(out_data, open(f'{save_file}/g_{filename}', 'wb'))


#plt.show()
