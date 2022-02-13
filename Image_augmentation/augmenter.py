import argparse
import os
import albumentations as A
import cv2
import tqdm

def augment(trainning_folder, transforms_folder):
    
    ext = ['.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp', '.apng', '.avif', '.gif', '.png', '.svg', '.webp']

    for root, dirs, files in os.walk(trainning_folder):
        for categ in dirs:
            categ_path = os.path.join(root, categ)

            for image_name in os.listdir(categ_path):

                im_path = os.path.join(categ_path, image_name)
                if os.path.isfile(im_path) and image_name.endswith(tuple(ext)):

                    image = cv2.imread(im_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Transform image with the augmentation pipelines we have imported
                    i=0
                    for trans_file in os.listdir(transforms_folder):
                        i+=1
                        # Load transfomers
                        transform = A.load(os.path.join(transforms_folder, trans_file), data_format='yaml')
                        # Augment image
                        transformed_image = transform(image=image)['image']
                        # Save image
                        extension = os.path.splitext(image_name)[1]
                        cv2.imwrite(categ_path + '/' + image_name.split(extension)[0] + '_aug' + str(i) + extension, transformed_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runnnig...")
    parser.add_argument(
        "--trainning-folder", 
        "-I",
        default="", 
        help="Enter trainning images folder path"
    )
    parser.add_argument(
        "--transforms-folder", 
        "-T",
        default="", 
        help="Enter transformers folder path"
    )
    args = parser.parse_args()
    augment(args.trainning_folder, args.transforms_folder)
        
