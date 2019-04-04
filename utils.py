import numpy as np
import pandas as pd

# Create submission DataFrame
def create_submission(csv_name, predictions, image_ids):
    """
    csv_name -> string for csv ('XXXXXXX.csv')
    predictions -> numpyarray of size (num_examples, height, width)
                In this case (num_examples, 512, 512)
    image_ids -> numpyarray or list of size (num_examples,)
    
    predictions[i] should be the prediciton of road for image_id[i]
    """
    sub = pd.DataFrame()
    sub['ImageId'] = image_ids
    encodings = []
    num_images = len(image_ids)
    for i in range(num_images):
        if (i+1) % (num_images//10) == 0:
            print(i, num_images)
        encodings.append(rle_encoding(predictions[i]))
        
    sub['EncodedPixels'] = encodings
    sub['Height'] = [512]*num_images
    sub['Width'] = [512]*num_images
    sub.to_csv(csv_name, index=False)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    """
    x = numpyarray of size (height, width) representing the mask of an image
    if x[i,j] == 0:
        image[i,j] is not a road pixel
    if x[i,j] != 0:
        image[i,j] is a road pixel
    """
    dots = np.where(x.T.flatten() != 0)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): 
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def get_img_id(img_path):
    
    img_basename = os.path.basename(img_path)
    img_id = os.path.splitext(img_basename)[0][:-len('_sat')]
    return img_id

def image_gen(img_paths, img_size=(img_height, img_width), paths_to_train):

    for img_path in img_paths:
        
        img_id = get_img_id(img_path)
        mask_path = os.path.join('comp-540-spring-2019/train', img_id + '_msk.png')
        
        img = imread(img_path) / 255.
        mask = rgb2gray(imread(mask_path))
        
        img = resize(img, img_size, preserve_range=True)
        mask = resize(mask, img_size, mode='constant', preserve_range=True)
        mask = (mask >= 0.5).astype(float)
        
        yield img, mask

def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * (K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    return score

def image_batch_generator(img_paths, batchsize = batch_size):
    
    while True:
        
        ig = image_gen(img_paths)
        batch_img, batch_mask = [], []
        
        for img, mask in ig:

            batch_img.append(img)
            batch_mask.append(mask)

            if len(batch_img) == batchsize:
                
                yield np.stack(batch_img, axis=0), np.expand_dims(np.stack(batch_mask, axis=0),axis = -1)
                batch_img, batch_mask = [], []
        
        if len(batch_img) != 0:
            yield np.stack(batch_img, axis=0), np.expand_dims(np.stack(batch_mask, axis=0),axis = -1)
            batch_img, batch_mask = [], []

def calc_steps(data_len, batchsize):
    
    return (data_len + batchsize - 1) // batchsize