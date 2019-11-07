def gen_Annotations_dir_NASA():
    # annotations_dir = '../Annotations'
    src_file = '/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/raw_data/NASA_GTF.txt'
    # Load all annotations and convert to the appropriate form
    with open(src_file, "r") as read_file:
        line = read_file.readline() #create one new file per line
        while line:
            num_boxes = line.split(' : ')[1]
            if int(num_boxes) > 0:
                line_list = line.split(' : ')[3:]
                new_file_name = line.split(' : ')[0][:-4]
                line_to_write = []
                with open('/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/Annotations/' + new_file_name + '.txt', "w+") as write_file:
                    for element in line_list:
                        if element == '2' or element == line_list[-1]:
                            if len(line_to_write) == 4:
                                for bounding_box in line_to_write:
                                    write_file.write("%s " % bounding_box)
                                write_file.write("1 \n")
                                line_to_write = []
                        else:
                            line_to_write.append(element)
            line = read_file.readline()

def gen_Annotations_dir_student():
    src_file = '/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/raw_data/StudentsLabels.txt'
    with open(src_file, "r") as read_file:
        line = read_file.readline() #create one new file per line
        while line:
            line_list = line.split()[2:]
            write_file_name = line.split()[0][:-4]
            line_to_write = []
            with open('/Users/mfinkels/repos/tf-Faster-RCNN-craterDetection/project/Data/LRO-equator/Annotations/' + write_file_name + '.txt', "a") as write_file:
                for element in line_list:
                    if len(line_to_write) == 4:
                        for top_left_coord in line_to_write[:2]:
                            write_file.write("%s " % top_left_coord)
                        bottom_right = [int(line_to_write[0]) + int(line_to_write[2]),
                                        int(line_to_write[1]) + int(line_to_write[3])]
                        for bottom_right_coord in bottom_right:
                            write_file.write("%s " % str(bottom_right_coord))
                        write_file.write("1 \n")
                        line_to_write = []
                    line_to_write.append(element)
            line = read_file.readline()


gen_Annotations_dir_NASA()
gen_Annotations_dir_student()
