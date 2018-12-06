import os 
import generate_car_plate as gc
import gen_number_letters as gn 

if __name__ == "__main__":
    print("Welcome to data generator for the car plate, please choose options below\n")
    print("1. Generate the car plate dataset")
    print("2. Generete the numbers and letter for car plate ")
    print("3. Generate common txt")

    ROOT_DIR = os.path.abspath("../")

    data_path = os.path.join(ROOT_DIR, 'car_plate_data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    while True:

        choose = input('Enter number > ')

        if (choose == str(1)):
            print('Input how big is dataset')
            size, val = input("Enter training number and validation number > ").split()
            print('Start generate the data for car plate')
            gc.run(int(size), int(val))
            break
        elif (choose == str(2)):
            print('Input how big is dataset')
            size, val = input("Enter training number and validation number > ").split()
            print('Start generate the data for the numbers and letters')
            gn.run(int(size), int(val))
            break
        elif (choose == str(3)):
            print('Input how big is dataset')
            size, val = input("Enter training number and validation number > ").split()
            print('Start generate the data for the numbers and letters')
            gc.run(int(size), int(val), txt_file_train='general_train.txt', txt_file_test='general_test.txt', folder_train='train',folder_test='test')
            gn.run(int(size), int(val), txt_file_train='general_train.txt', txt_file_test='general_test.txt', folder_train='train', folder_test='test')
            break

        else:
            print('Error, please choose 1 or 2 option, try again')
