import ntpath
import  keras
import numpy as np
import os 
import random 
import s3fs

    
class groovy_data_generator(keras.utils.Sequence):

    def __init__(self, batch_size, image_dir, train_location, image_size, num_unique_labels, fs=None) :
        """
        if using cloud services, make sure your image_dir input ends in a / since there is no 
        safe s3fs method for joining file directory and file name like os.path.join(dir,fname) 
        in local cases.
        """
        self.train_location = train_location

        if self.train_location == "local":
            self.image_filenames = [os.path.join(image_dir,fname) for fname in os.listdir(image_dir)]
        else:
            self.fs = s3fs.S3FileSystem() 
            self.image_filenames = [image_dir+ fname for fname in self.fs.ls(image_dir)]
        random.shuffle(self.image_filenames)

        self.batch_size = batch_size

        self.image_size = image_size

        self.num_unique_labels = num_unique_labels

    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx) :
        classes = []
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        if batch_x == []:
            raise Exception("Image Dir Contains {} Files. You are trying to index {} to {}".format(len(self.image_filenames),
                                                                                                  idx*self.batch_size,
                                                                                                   (idx+1)*self.batch_size))
        
        data_tensor_shape = (self.batch_size, self.image_size [0], self.image_size [1], self.image_size [2])
        data_tensor = np.zeros(data_tensor_shape)

        labels = np.zeros([self.batch_size, self.num_unique_labels]) 
        for ct,file_path in enumerate(batch_x):
            label = int(ntpath.basename(file_path).split("_")[0]) 
            labels[ct][label] = 1
            classes.append(label)
            if self.train_location == "local":
                data = np.load(file_path)

            else:
                with self.fs.open(file_path) as f:
                    data = np.load(f)

            data_tensor[ct,:,:,:] = data
            
#         unique, counts = np.unique(classes, return_counts=True)
#         ct_dict = dict(zip(unique, counts))
#         print("This tensor contains labels from the following groups")
#         print(ct_dict)
        return data_tensor, labels
        

