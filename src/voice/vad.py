import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmobaseCNN:
    """
    Заготовка для VAD-модуля на базе openSMILE + CNN.
    Пока что неактивна, но структура готова под внедрение.
    """

    def __init__(self, model_path: str = None):
        # Здесь будет загрузка ONNX или Torch-модели
        self.model_path = model_path
        self.model = None  # TODO: модель не загружается пока

        logger.info("VAD инициализирован (заглушка)")

    def extract_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Обёртка под openSMILE (emobase + LLD).
        Возвращает фичи [time, features]
        """
        raise NotImplementedError("OpenSMILE фичи пока не подключены")

    def predict(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Основной метод VAD: принимает аудиосигнал,
        возвращает маску речи [0/1] по фреймам.
        """
        logger.warning("VAD не активен. Возвращаем пустую маску.")
        return np.zeros(len(signal) // 160)  # грубая оценка: 10ms фреймы

    def segment(self, signal: np.ndarray, sample_rate: int) -> list:
        """
        Делит аудио на сегменты, где есть голос.
        Возвращает список (start, end) в сэмплах.
        """
        vad_mask = self.predict(signal, sample_rate)
        # TODO: в будущем — нормальный порог и сглаживание
        segments = []
        in_speech = False
        start = 0
        for i, active in enumerate(vad_mask):
            if active and not in_speech:
                in_speech = True
                start = i * 160
            elif not active and in_speech:
                in_speech = False
                end = i * 160
                segments.append((start, end))
        return segments
