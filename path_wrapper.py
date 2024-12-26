import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='', type=str,
                    help='The folder path')
parser.add_argument('--flist_file', default='SIDD_train_input.flist', type=str,
                    help='The train filename.')
parser.add_argument('--keywords', default='', type=str,
                    help='match the image name')
parser.add_argument('--subfolders', action='store_true', help='Wrapping multiple subfolders inside. Default: False')

if __name__ == "__main__":

    args = parser.parse_args()

    flist = open(args.flist_file, "w")
    if(args.subfolders):
        for dir_name in os.listdir(args.folder_path):
            dir_path = os.path.join(args.folder_path, dir_name)
            if os.path.isdir(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith(('.png', '.jpg', '.jpeg', '.gif')) and args.keywords in file:
                            file_path = os.path.join(root, file)
                            flist.write(file_path + '\n')
        
    else:    
        for root, dirs, files in os.walk(args.folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(root, file)
                    flist.write(image_path + '\n')
    
    print('Image paths have been written to', args.flist_file)