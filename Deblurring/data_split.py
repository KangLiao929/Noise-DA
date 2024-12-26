import argparse

def split_path(args):
    with open(args.source_file_path, 'r', encoding='utf-8') as source_file, \
        open(args.output_file_path1, 'w', encoding='utf-8') as output_file1, \
        open(args.output_file_path2, 'w', encoding='utf-8') as output_file2:
        for line in source_file:
            parts = line.strip().split()
            if len(parts) == 2:
                new_line1 = args.base_path + parts[0] + '\n'
                new_line2 = args.base_path + parts[1] + '\n'
                output_file1.write(new_line1)
                output_file2.write(new_line2)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file_path', type=str, default='./datasets/split_names/RealBlur_J_test_list.txt')
    parser.add_argument('--output_file_path1', type=str, default='NoiseDA/flist_name/deblur/RealBlur_J_test_gt.flist')
    parser.add_argument('--output_file_path2', type=str, default='NoiseDA/flist_name/deblur/RealBlur_J_test_input.flist')
    parser.add_argument('--base_path', type=str, default='./datasets/deblur/RealBlur_J/')
    args = parser.parse_args()
    split_path(args)
    print('Splitting Done.')