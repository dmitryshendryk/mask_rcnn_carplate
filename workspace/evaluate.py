import os
import skimage.draw
import datetime


def evaluate_carplate(img_folder, model, root_dir):
    now = datetime.datetime.now()
    if not os.path.exists(root_dir + "/eval_results"):
        os.mkdir(root_dir + "/eval_results" )

    eval_path = root_dir + '/eval_results/' + 'plate_' + now.isoformat()
    eval_path_txt = open(eval_path, 'w')

    check_file = root_dir + '/eval_real_data/' + 'eval.txt'
    
    name_check = {}
    with open(check_file) as fp:
        for cnt, line in enumerate(fp):
            val,key = line.split(" ")[0],  line.split(" ")[1].split("\n")[0] 
            name_check[val] = key
    print(name_check)
    class_names = ["BG, carplate"]
    correct = 0
    error = 0

    if img_folder:

        imgs = os.listdir(img_folder)
        for img in imgs:
            image = skimage.io.imread(img_folder + '/' + img)

            r = model.detect([image], verbose=1)[0]

            if  len(r['scores']) != 0 and name_check[img] == 'True':
                 eval_path_txt.write(img + '  ' + 'Correct' + '\n')
                 correct += 1
            elif len(r['scores']) != 0 and name_check[img] == 'False' :
                 eval_path_txt.write(img + '  ' + 'Error' + '\n')
                 error += 1
            elif len(r['scores']) == 0 and name_check[img] == 'True':
                 eval_path_txt.write(img + '  ' + 'Error' + '\n')
                 error += 1
            elif len(r['scores']) == 0 and name_check[img] == 'False':
                 eval_path_txt.write(img + '  ' + 'Correct' + '\n')
                 correct += 1



            
            print(r['class_ids'], r['scores'])
        print(("The total accuracy is: " +  str(int((correct/len(imgs)) * 100)) + '% '))
        print(('Total error is: ' + str(int((error/len(imgs)) * 100)) + '% '))
        print('Images accurate: ' + str(correct))
        print('Images failed: ' + str(error))
        print('Total images processed: ' + str(correct + error))
        eval_path_txt.write("The total accuracy is: " +  str(int((correct/len(imgs)) * 100)) + '% ' +'\n' )
        eval_path_txt.write('Images accurate: ' + str(correct) +'\n')
        eval_path_txt.write('Images failed: ' + str(error) +'\n')
        eval_path_txt.write('Total images processed: ' + str(correct + error) +'\n')

def evaluate_numbers(img_folder, model, root_dir):
    now = datetime.datetime.now()
    
    if not os.path.exists(root_dir + "/eval_results"):
        os.mkdir(root_dir + "/eval_results/number_plate" )

    eval_path = root_dir + '/eval_results/number_plate/' + 'number_' + now.isoformat()
    eval_path_txt = open(eval_path, 'w')

    class_names = ['BG','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'J', 'K', 'L', 'M', 'N', 'P',  'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    if img_folder:

        imgs = os.listdir(img_folder)

        for img in imgs:
            image = skimage.io.imread(img_folder + '/' + img)

            r = model.detect([image], verbose=1)[0]
            print()

def create_eval_dataset(root_dir, file, img_folder):
    if not os.path.exists(root_dir + "/eval_real_data"):
        os.mkdir(root_dir + "/eval_real_data" )

    eval_path = root_dir + '/eval_real_data' + '/' + file
    eval_path_txt = open(eval_path, 'w')
    if img_folder:

        imgs = os.listdir(img_folder)
        imgs.sort(key=lambda x: int(os.path.splitext(x)[0]))
        for img in imgs:
            # image = skimage.io.imread(img_folder+'/'+img)
            eval_path_txt.write(img + '\n')


