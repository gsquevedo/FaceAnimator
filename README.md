# Animação Facial com First Order Motion Model (FOMM)

Este projeto utiliza o modelo de aprendizado profundo **First Order Motion Model (FOMM)** para animar imagens estáticas com base em movimentos capturados de um vídeo ou de outra imagem. O modelo dissocia informações de aparência e movimento, utilizando uma formulação auto-supervisionada que permite aprender pontos-chave de animação sem depender de anotações dispendiosas.

## Como Funciona

O FOMM extrai pontos-chave tanto da imagem ou vídeo fonte quanto da imagem alvo e aplica transformações afins locais para deformar a imagem alvo, reproduzindo de forma precisa o movimento presente na fonte. Isso permite lidar com movimentos complexos e grandes mudanças na pose dos objetos.

O modelo utilizado é baseado no **VoxCeleb**, um conjunto de dados de rostos humanos falando, e a configuração **vox-adv-256.yaml**, que melhora a qualidade da síntese facial com redes adversariais generativas (GANs). O checkpoint utilizado, **vox-adv-cpk.pth.tar**, contém os pesos necessários para realizar a animação.

## Requisitos do Sistema

- **GPU Nvidia** (recomendado para aceleração do treinamento e animação)

## 🛠️ Passos para Configuração  

### 📌 Passo 1: Instalar os módulos necessários  

- **Instalar dependências**:  
  ```bash
  pip install -r requirements.txt
  ```  
- **Instalar PyTorch e Torchvision**:  
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```  

---

### 📌 Passo 2: Baixar arquivos necessários  

Baixe o arquivo **cascade**, os pesos da rede neural e o modelo treinado.  
Os arquivos devem ser salvos em uma pasta chamada **extract**.  

📩 **Para obter os arquivos, entre em contato com a desenvolvedora pelo e-mail:**  
**soaresgabriele365@gmail.com**  

---

### 📌 Passo 3: Executar o projeto  

#### 📷 Rodar a aplicação com a câmera ao vivo:  
```bash
python image_animation.py -i caminho_para_imagem -c caminho_para_checkpoint
```  
**Exemplo**:  
```bash
python image_animation.py -i ./Inputs/Monalisa.png -c ./checkpoints/vox-cpk.pth.tar
```  

#### 🎥 Rodar a aplicação com um arquivo de vídeo:  
```bash
python image_animation.py -i caminho_para_imagem -c caminho_para_checkpoint -v caminho_para_video
```  
**Exemplo**:  
```bash
python image_animation.py -i ./Inputs/Monalisa.png -c ./checkpoints/vox-cpk.pth.tar -v ./video_input/test1.mp4
```  

---

![Demonstração](animate.gif)  

### 🔗 Projeto original  
*Criado por* [anandpawara](https://github.com/anandpawara/Real_Time_Image_Animation)  

---

