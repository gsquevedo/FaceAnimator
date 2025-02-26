import imageio.v2 as imageio 
import torch
from tqdm import tqdm
from animate import normalize_kp
from demo import load_checkpoints
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, exposure
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
print("[INFO] Carregando imagem de origem e checkpoint...")
source_path = args['input_image']
checkpoint_path = args['checkpoint']
video_path = args.get('input_video', None)

source_image = imageio.imread(source_path)
source_image = resize(source_image, (256, 256))[..., :3]

generator, kp_detector = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path)

# Criar diretório de saída
os.makedirs('output', exist_ok=True)

relative = True
adapt_movement_scale = True
cpu = True

# Configuração da câmera ou vídeo
cap = cv2.VideoCapture(video_path if video_path else 0)
print(f"[INFO] Usando {'vídeo' if video_path else 'webcam'} para animação...")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter('output/test.avi', fourcc, 12, (256 * 3, 256), True)

# Converte a imagem original para OpenCV
cv2_source = (source_image * 255).astype(np.uint8)
cv2_source = cv2.cvtColor(cv2_source, cv2.COLOR_RGB2BGR)

with torch.no_grad():
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        if not video_path:
            # Ajuste o fator de escala para "afastar" a imagem
            scale_factor = 1.5  # Ajuste esse valor para mais ou menos afastamento
            new_size = int(256 * scale_factor)

            center_h, center_w = h // 2, w // 2
            cropped_frame = frame[max(0, center_h - new_size // 2):min(h, center_h + new_size // 2),
                                  max(0, center_w - new_size // 2):min(w, center_w + new_size // 2)]

            # Redimensionar para 256x256 novamente
            frame_resized = resize(cropped_frame, (256, 256), mode='reflect', anti_aliasing=True)[..., :3]
        else:
            cropped_frame = frame

        frame_resized = (frame_resized * 255).astype(np.uint8)

        if count == 0:
            source_image1 = frame_resized.astype(np.float32) / 255.0
            source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            kp_driving_initial = kp_detector(source1)

        # Processar o quadro do vídeo
        frame_test = frame_resized.astype(np.float32) / 255.0
        driving_frame = torch.tensor(frame_test[np.newaxis]).permute(0, 3, 1, 2)
        if not cpu:
            driving_frame = driving_frame.cuda()
        kp_driving = kp_detector(driving_frame)

        # Normalizar os pontos-chave faciais
        kp_norm = normalize_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
            use_relative_movement=relative,
            use_relative_jacobian=relative,
            adapt_movement_scale=adapt_movement_scale
        )

        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
        im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

        # Evitar imagens brancas: normalizar corretamente
        im = np.clip(im, 0, 1)
        im = (im * 255).astype(np.uint8)

        # Converter para formato OpenCV corretamente
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # Criar um frame combinado
        joinedFrame = np.concatenate((cv2_source, im, frame_resized), axis=1)

        # Exibir a animação
        cv2.imshow('Test', joinedFrame)

        # Salvar corretamente
        out1.write(joinedFrame)

        count += 1
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    out1.release()
    cv2.destroyAllWindows()
