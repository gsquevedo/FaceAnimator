import imageio
import torch
from tqdm import tqdm
from animate import normalize_kp
from demo import load_checkpoints
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import cv2
import os
import argparse
import sys

def is_executable():
    return getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')

# Argumentos do programa
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", required=True, help="Path to image to animate")
ap.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint")
ap.add_argument("-v", "--input_video", required=False, help="Path to video input")
args = vars(ap.parse_args())

config_path = 'config/vox-256.yaml' if not is_executable() else './vox-256.yaml'

# Carregar imagem de origem e checkpoints
print("[INFO] Loading source image and checkpoint...")
source_path = args['input_image']
checkpoint_path = args['checkpoint']
if args['input_video']:
    video_path = args['input_video']
else:
    video_path = None

source_image = imageio.imread(source_path)
source_image = resize(source_image, (256, 256))[..., :3]
generator, kp_detector = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path)

# Carregar o detector de rosto
print(cv2.data.haarcascades)

# Verificar se está no modo de executável ou ambiente local
if is_executable():
    # No executável, o arquivo Haar Cascade deve ser localizado de forma diferente
    haar_cascade_path = os.path.join(sys._MEIPASS, 'cv2', 'data', 'haarcascade_frontalface_default.xml')
else:
    # No ambiente local, usamos o caminho padrão
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Criação do diretório de saída
if not os.path.exists('output'):
    os.mkdir('output')

relative = True
adapt_movement_scale = True
cpu = True

# Configuração da câmera ou vídeo
if video_path:
    cap = cv2.VideoCapture(video_path)
    print("[INFO] Loading video from the given path")
else:
    cap = cv2.VideoCapture(0)
    print("[INFO] Initializing front camera...")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter('output/test.avi', fourcc, 12, (256 * 3, 256), True)

cv2_source = cv2.cvtColor(source_image.astype('float32'), cv2.COLOR_BGR2RGB)
with torch.no_grad():
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)
    count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            # Detectar rosto na imagem
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) > 0:
                x, y, w, h = faces[0]  # Seleciona o primeiro rosto detectado

                # Expandir os limites do recorte para incluir mais área ao redor do rosto
                margin = 0.3  # Porcentagem de margem adicional
                x_margin = int(w * margin)
                y_margin = int(h * margin)

                # Garantir que os limites não excedam os da imagem original
                x1 = max(0, x - x_margin)
                y1 = max(0, y - y_margin)
                x2 = min(frame.shape[1], x + w + x_margin)
                y2 = min(frame.shape[0], y + h + y_margin)

                face_frame = frame[y1:y2, x1:x2]  # Recorta a região expandida
                face_frame = resize(face_frame, (256, 256))[..., :3]
            else:
                # Usa o quadro completo se nenhum rosto for detectado
                face_frame = resize(frame, (256, 256))[..., :3]

            if count == 0:
                source_image1 = face_frame
                source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                kp_driving_initial = kp_detector(source1)

            frame_test = torch.tensor(face_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            driving_frame = frame_test
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source,
                                   kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial,
                                   use_relative_movement=relative,
                                   use_relative_jacobian=relative,
                                   adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            joinedFrame = np.concatenate((cv2_source, im, face_frame), axis=1)

            cv2.imshow('Test', joinedFrame)
            out1.write(img_as_ubyte(joinedFrame))
            count += 1
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out1.release()
    cv2.destroyAllWindows()
