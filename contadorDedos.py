import cv2
import mediapipe as mp
from pathlib import Path

def main():
    WEB = 0
    MODELO = Path(__file__).with_name("hand_landmarker.task")

    video = None
    try:
        if not MODELO.is_file():
            raise FileNotFoundError(f"Modelo não encontrado: {MODELO}")

        video = cv2.VideoCapture(WEB)
        if not video.isOpened():
            raise RuntimeError("Não foi possível abrir a webcam.")

        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 60

        opcoes = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELO)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with mp.tasks.vision.HandLandmarker.create_from_options(opcoes) as detector:
            numero_frame = 0
            while True:
                ret, frame = video.read()

                if not ret:
                    break

                videoRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=videoRGB)
                tempo_ms = int(numero_frame * 1000 / fps)
                resultado = detector.detect_for_video(imagem, tempo_ms)

                total_dedos = 0
                pontas = [8, 12, 16, 20]
                articulacoes = [6, 10, 14, 18]
                bases = [5, 9, 13, 17]

                for indice, pontos in enumerate(resultado.hand_landmarks):
                    mp.tasks.vision.drawing_utils.draw_landmarks(
                        frame,
                        pontos,
                        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    )
                    dedos_da_mao = 0

                    for ponta, articulacao, base in zip(pontas, articulacoes, bases):
                        if (
                            pontos[ponta].y < pontos[articulacao].y
                            and pontos[articulacao].y < pontos[base].y
                        ):
                            dedos_da_mao += 1

                    polegar_estendido = (
                        abs(pontos[4].x - pontos[2].x)
                        > abs(pontos[3].x - pontos[2].x)
                    )

                    if polegar_estendido:
                        dedos_da_mao += 1

                    total_dedos += dedos_da_mao

                cv2.rectangle(frame, (10, 10), (220, 55), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"Dedos: {total_dedos}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Video", frame)

                if cv2.waitKey(40) & 0xFF == ord('q'):
                    break

                numero_frame += 1
    except FileNotFoundError as erro:
        print(f"Erro: {erro}")
    except RuntimeError as erro:
        print(f"Erro ao iniciar o detector ou acessar a webcam: {erro}")
    except Exception as erro:
        print(f"Ocorreu um erro inesperado: {erro}")
    finally:
        if video is not None:
            video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
