import os,cv2,sys,time
import numpy as np
from Libs import Utils as u
from Libs.Efx import Efx

w,h,color = 330,330,3

train_folder = 'dataset/train'
test_folder = 'dataset/test'
eval_folder = 'dataset/eval'

#  Limieta a quantidade de imagens
sets = [(test_folder,99999),(eval_folder,99999),(train_folder,99999)] 
#sets = [(train_folder,99999)] 

def rotate():
    for set in sets:
        rotate_lim = set[1]
        print('Rotate')
        print('--------------------------------------------------------------------------------------------')
        print(set)
        classes = u.get_paths(os.scandir(set[0]))
        for out_path in classes:
            files = u.get_paths(os.scandir(out_path))
            qtd = len(files)
            for f in files:
                f_name = f.split('/')[-1]
                f_name = f_name[:-4]

                efx = Efx(cv2.imread(f))

                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, -2)
                    qtd += 1
                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, -3)
                    qtd += 1
                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, -5)
                    qtd += 1
                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, -7)
                    qtd += 1

                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, 2)
                    qtd += 1
                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, 3)
                    qtd += 1
                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, 5)
                    qtd += 1
                if qtd < rotate_lim:
                    efx.rotate(out_path, f_name, 7)
                    qtd += 1

                if qtd >= rotate_lim:
                    break


def brightness():
    for set in sets:
        bright_lim = set[1]
        print('Brightness e Contraste')
        print('--------------------------------------------------------------------------------------------')
        print(set)
        classes = u.get_paths(os.scandir(set[0]))
        for out_path in classes:
            files = u.get_paths(os.scandir(out_path))
            qtd = len(files)
            for f in files:
                f_name = f.split('/')[-1]
                f_name = f_name[:-4]

                efx = Efx(cv2.imread(f))

                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, 15, 1)
                    qtd += 1
                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, -15, 1)
                    qtd += 1
                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, 30, 1)
                    qtd += 1
                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, -30, 1)
                    qtd += 1

                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, 0, 0.8)
                    qtd += 1
                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, 0, 0.9)
                    qtd += 1
                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, 0, 1.1)
                    qtd += 1
                if qtd < bright_lim:
                    efx.filter_brightnes_contrast(out_path, f_name, 0, 1.2)
                    qtd += 1

                if qtd >= bright_lim:
                    break


def gerar_dataset():
    for set in sets:

        imgs = []
        labels = []
        classes = []
        img_count = 0

        total = u.qtd_files(set[0])

        print(total)
        p = int(total/100) * 2

        set_name = set[0].split('/')[-1]
        output_path = '.'

        print('Gerar dataset')
        print('--------------------------------------------------------------------------------------------')
        print(set[0])
        classes_path = u.get_paths(os.scandir(set[0]))
        for out_path in classes_path:
            files = u.get_paths(os.scandir(out_path))
            classe = out_path.split('/')[-1]
            print('_________________________________________')

            for f in files:
                if img_count % p == 0:
                    print('{} - {} / {} \t\t {:.2f}%'.format(classe,img_count,total,img_count*100/total))

                cv_img = cv2.imread(f)
                try:


                    if cv_img.shape[2] != 3:
                        print("N�o � RGB")
                        print("Shape:", cv_img.shape, "\nPath:", f)
                    else:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                        if classe not in classes:
                            classes.append(classe)

                        labels.append(classe)

                        img = cv2.resize(cv_img, (w, h))
                        img = img.reshape(w * h * color)
                        imgs.append(img)

                        img_count += 1
                except:
                    print(f)

        print('{} - {} / {} \t\t {:.2f}%'.format(classe,img_count,total,img_count*100/total))

        np_i = np.array(imgs)
        np_l = np.array(labels)

        print('Shape:', np_i.shape)
        print('W*H*C: ', w * h * color)
        print('Classes:', classes)
        print('{} imagens'.format(img_count))
        print('Salvando')

        out_file = os.path.join(output_path,set_name+'_qtd_classes('+str(len(classes)) + ')')
        np.savez_compressed(out_file , imgs=np_i, labels=np_l, classes=classes)
        print('--------------------------------------------------------------------------------------------')
        with open('labels.txt','w') as f:
            for c in classes:
                f.writelines(c + "\n")
def help():
    print('------------------------------------')
    for set in sets:
        print('')
        print(u.get_name(os.scandir(set[0])))

    print('------------------------------------')
    print('-r Rota��o')
    print('-b Brilho')
    print('-g gerar dataset')
    exit()

if __name__ == '__main__':

    if len(sys.argv) == 1:
        help()
    else:
        r = False
        b = False
        g = False

        for i in range(1,len(sys.argv)):
            if sys.argv[i] == '-r': # roracionar
                r = True
            elif sys.argv[i] == '-b': # brilho
                b = True
            elif sys.argv[i] == '-g': # gerar dataset
                g = True
            else:
                print('parametro invalido')
                help()
        if r:
            rotate()
        if b:
            brightness()
        if g:
            tini=time.time()
            gerar_dataset()
            print('{:.2f} ms'.format(time.time()-tini))