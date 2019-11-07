from random import randrange

def get_splits():
    src_file_NASA = '/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/raw_data/NASA_GTF.txt'
    file_list = []
    with open(src_file_NASA, "r") as read_file:
        line = read_file.readline()
        while line:
            num_boxes = line.split(' : ')[1]
            if int(num_boxes) > 0:
                file_name = line.split(' : ')[0][:-4]
                file_list.append(file_name)
            line = read_file.readline()
    src_file_students = '/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/raw_data/StudentsLabels.txt'
    with open(src_file_students, "r") as read_file:
        line = read_file.readline()
        while line:
            file_name = line.split()[0][:-4]
            if file_name not in file_list:
                file_list.append(file_name)
            line = read_file.readline()
    print(len(file_list))
    train = [file_list.pop(randrange(len(file_list))) for _ in range(int(0.6*len(file_list)))]
    valid = [file_list.pop(randrange(len(file_list))) for _ in range(int(0.5*len(file_list)))]
    test = file_list
    split_list = [train, valid, test]
    for split in split_list:
        if split == train:
            new_file = 'train'
        elif split == valid:
            new_file = 'valid'
        else:
            new_file = 'test'
        with open(
            '/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/Names/' + f'{new_file}' + '.txt',
            "w+") as write_file:
            for file_name in split:
                write_file.write(file_name)
                write_file.write("\n")

if __name__ == "__main__":
    get_splits()