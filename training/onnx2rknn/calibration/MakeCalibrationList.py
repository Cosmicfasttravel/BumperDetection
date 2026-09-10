import os
import glob

def generate_calibration_file(folder_name='imgs', output_file='calibration_list.txt'):
    if not os.path.exists(folder_name):
        return
        
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in valid_extensions:
        search_path = os.path.join(folder_name, '**', ext)
        found_files = glob.glob(search_path, recursive=True)
        image_paths.extend(found_files)
        
    if not image_paths:
        return

    absolute_paths = [os.path.abspath(p) for p in image_paths]
    absolute_paths.sort()
    
    with open(output_file, 'w') as f:
        for path in absolute_paths:
            f.write(path + '\n')

if __name__ == '__main__':
    generate_calibration_file()
