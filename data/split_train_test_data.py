import h5py
import os
import numpy as np

classes = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '6QAM1']


def split_train_test_data(source_h5, target_dir, samples=2000, train_ratio=0.8, num_classes=24, seed=123):
    np.random.seed(seed)
    os.makedirs(target_dir + '/train', exist_ok=True)
    os.makedirs(target_dir + '/val', exist_ok=True)
    f = h5py.File(source_h5, 'r')
    print(f['X'].shape)
    modu_snr_size = samples
    train_size = int(np.floor(samples * train_ratio))
    test_size = int(samples - train_size)
    # modu_list = [3, 10, 11, 23]
    modu_group = [
        [3],    # FM
        [7],    # 8ASK
        [16],   # ASK
        [9],    # PSK
    ]
    modu_idx = 0
    for label in range(num_classes):
        print(modu_idx)
        modu_list = modu_group[label]
        for modu in modu_list:
            X_train_modu, X_test_modu = np.zeros((train_size*26, 1024, 2)), np.zeros((test_size*26, 1024, 2))
            Y_train_modu, Y_test_modu = np.zeros((train_size*26, 1)), np.zeros((test_size*26, 1))
            Z_train_modu, Z_test_modu = np.zeros((train_size*26, 1)), np.zeros((test_size*26, 1))
            print('part ',modu)
            start_modu = modu*106496
            for snr in range(26):
                print('snr ', snr)
                start_snr = start_modu + snr*4096
                idx_list = np.random.choice(range(0,4096),size=modu_snr_size,replace=False)
                X = f['X'][start_snr:start_snr+4096][idx_list]
                Y = np.ones((modu_snr_size, 1), int)*label
                Z = f['Z'][start_snr:start_snr+4096][idx_list]

                #X = X[:,0:768,:]
                X_train_modu[snr*train_size:(snr+1)*train_size] = X[:train_size]
                
                X_test_modu[snr*test_size:(snr+1)*test_size] = X[train_size:]
                Y_train_modu[snr*train_size:(snr+1)*train_size] = Y[:train_size]
                Y_test_modu[snr*test_size:(snr+1)*test_size] = Y[train_size:]
                Z_train_modu[snr*train_size:(snr+1)*train_size] = Z[:train_size]
                Z_test_modu[snr*test_size:(snr+1)*test_size] = Z[train_size:]

            print(X_train_modu.shape, Y_train_modu.shape, Z_train_modu.shape)
            print(X_test_modu.shape, Y_test_modu.shape, Z_test_modu.shape)
            # print(X_train[:1])
            print(train_size)

            train_filename = target_dir + 'train/' + 'modu_' + str(modu_idx) + '.h5'
            f_train = h5py.File(train_filename,'w')
            f_train['X'] = X_train_modu
            f_train['Y'] = Y_train_modu
            f_train['Z'] = Z_train_modu

            print('X shape:',f_train['X'].shape)
            print('Y shape:',f_train['Y'].shape)
            print('Z shape:',f_train['Z'].shape)
            f_train.close()

            test_filename = target_dir + 'val/' + 'modu_' + str(modu_idx) + '.h5'
            f_test = h5py.File(test_filename,'w')
            f_test['X'] = X_test_modu
            f_test['Y'] = Y_test_modu
            f_test['Z'] = Z_test_modu

            print('X shape:',f_test['X'].shape)
            print('Y shape:',f_test['Y'].shape)
            print('Z shape:',f_test['Z'].shape)

        modu_idx += 1

    ### 
    # for i in range()
    # 不同SNR

# def merge_train_test_data(source_dir='./Dataset', target_dir='./merge_dataset', num_classes=4, seed=123):
#     np.random.seed(seed)
#     modu_list = [
#         [3],             # FM
#         [10,16,17,21],   # AMs
#         [7, 11],         # ASKs
#         [2,14,18,20,23], # QAMs
#     ]

#     for label in range(num_classes):
#         out_f_train = h5py.File(target_dir+'/train'+'/modu'+str(label),'w')
#         out_f_val   = h5py.File(target_dir+'/val'+'/modu'+str(label),'w')

#         X_train, X_val = None, None
#         Y_train, Y_val = None, None
#         Z_train, Z_val = None, None
        
#         for modu in modu_list[label]:
#             f_train = h5py.File(source_dir + '/train'+'/modu_'+str(modu)+'.h5', 'r')
#             f_val = h5py.File(source_dir + '/val'+'/modu_'+str(modu)+'.h5', 'r')

#         out_f_train.close()
#         out_f_val.close()

split_train_test_data('./datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5', './datasets/sub_dataset/', samples=2000, num_classes=4)
# merge_train_test_data()