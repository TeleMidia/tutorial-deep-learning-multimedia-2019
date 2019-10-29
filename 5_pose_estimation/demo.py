import sys
sys.path.append('./')

from inference import Inference
import cv2

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass

tensorflow_shutup()

# Caminho para o arquivo original
img_full_name = '0.jpg'
img_dir = './images/'
save_dir = './out/'
img_name, extension = img_full_name.split('.')
extension = '.' + extension
'''
 Instanciando um objeto da classe Inference 
 Essa classe constroi o grafo de execução dos modelos Stacked Hourglass e YOLO
 Também carrega os pesos treinados desses modelos.
'''
inf = Inference(config_file = 'config_tiny.cfg', model = 'hg_refined_tiny_200', yoloModel = 'YOLO_small.ckpt')

# Carregando uma imagem RGB 
img = cv2.imread(img_dir+img_full_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Caso a imagem não seja do tamanho 256x256, redimensionar.
if img.shape != (256,256,3):
    print('Wrong shape. Resizing.')
    img = cv2.resize(img, (256,256))

# Mostrando as dimensões, deve ser (256X256X3) == (alturaXlarguraXnumero_de_canais_de_cor)
print('Img shape: ',img.shape)

'''
 Chamando a função que prediz a pose, onde os parametros são:
 thresh : limiar de corte para o nivel de confiança para o keypoint
 pltJ : plotar keypoints (ou Joints)
 pltL : plotar segmentos (ou Limbs)
'''
new_img = inf.pltSkeleton(img, thresh = 0.5, pltJ = True, pltL = True)
new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(save_dir+img_name+'_prediction'+extension,new_img)

# Nessa função a predição para na fase de detecção de pessoas
bb = inf.pltBoundingBoxes(img)
bb = cv2.cvtColor(bb, cv2.COLOR_RGB2BGR)
cv2.imwrite(save_dir+img_name.split('.')[0]+'_bb'+extension,bb)

# Nessa função a predição para quando temos os heatmaps, a função retorna eles
hm = inf.predictHM(img)
hm = cv2.cvtColor(hm, cv2.COLOR_RGB2BGR)
cv2.imwrite(save_dir+img_name.split('.')[0]+'_hm'+extension,hm)

print(f'Done! Check {save_dir} for the results!')