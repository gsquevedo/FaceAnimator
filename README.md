# Animação de Imagem em Tempo Real  

Este projeto é uma aplicação em tempo real utilizando OpenCV e o **First Order Model** para animação de imagens.  

---

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

