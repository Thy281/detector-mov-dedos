# Detector de movimentos dos dedos

Aplicativo em Python que usa a webcam e o MediaPipe para detectar até duas mãos, desenhar seus pontos de referência e contar quantos dedos estão estendidos.

## Requisitos

- Python 3.14 ou superior
- [uv](https://docs.astral.sh/uv/)
- Uma webcam
- O arquivo de modelo `hand_landmarker.task` na raiz do projeto

## Instalação

Na pasta do projeto, instale as dependências registradas no `pyproject.toml` e no `uv.lock`:

```bash
uv sync
```

O pacote `mediapipe` fornece o módulo `cv2` usado pelo programa.

## Execução

Inicie o detector pela pasta do projeto:

```bash
uv run python contadorDedos.py
```

A janela da webcam mostra o esqueleto das mãos detectadas e o total de dedos estendidos. Pressione `q` para fechar a janela.

O script procura `hand_landmarker.task` na mesma pasta de `contadorDedos.py`. Se o modelo estiver ausente, ele informa o caminho esperado.

## Configuração

Por padrão, o programa abre a câmera de índice `0`, definido por `WEB` em `contadorDedos.py`. Se houver mais de uma câmera, altere esse valor para o índice correspondente.
