import numpy as np
import logging
from utils import sigmoid
import onnxruntime
from settings import Settings

logger = logging.getLogger(__name__)


class EmobaseCNN:
    """
    VAD-модель на ONNX. Принимает готовые признаки [frames, features].
    """

    def __init__(self, settings: Settings):
        self.thresholds = settings.voice_command_config.confidence_thresholds
        self.debug = settings.debug
        self.sample_rate = settings.audio_config.common.sample_rate
        self.merge_segments = settings.audio_config.vad_features.merge_segments
        self.merge_gap = settings.audio_config.vad_features.merge_gap

        model_path = settings.model_paths.vad
        try:
            self.session = onnxruntime.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"✅ VAD-модель загружена: {model_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки VAD-модели: {e}")

    def _preprocess(self, features: np.ndarray) -> np.ndarray:
        x = features.T  # [time, features]
        x = np.expand_dims(x, axis=0)      # [1, time, features]
        x = np.expand_dims(x, axis=0)      # [1, 1, time, features]
        return x.astype(np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess(features)
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        probs = sigmoid(logits[0])
        return np.squeeze(probs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Принимает признаки формы (frames, features), возвращает бинарную маску.
        """
        probs = self.predict_proba(features)
        threshold = self.thresholds.vad_default

        # Бинаризуем по порогу
        probs_bin = (probs >= threshold).astype(np.int32)

        return probs_bin


    def segment(self, probs_bin: np.ndarray, valid_frames: int) -> np.ndarray:
        """
        Находит интервалы, где речь (значение 1) начинается и заканчивается.
        
        Аргументы:
            probs_bin (np.ndarray): Бинарный массив (0 или 1).
            valid_frames (int): Число валидных отсчетов (например, длина исходного сигнала).
        
        Возвращает:
            numpy-массив с парами [начало, конец] для каждого сегмента речи в единицах отсчетов.
            Если сегмент начался в пределах valid_frames, но его завершение выходит за valid_frames,
            то конец сегмента заменяется на valid_frames.
        """
        # Если valid_frames не задан или больше длины probs_bin, используем всю длину
        if valid_frames is None or valid_frames > len(probs_bin):
            valid_frames = len(probs_bin)
        
        segments = []
        in_speech = False
        temp = []

        # Проходим по отсчетам (пробегаем только по valid_frames отсчетам)
        for i, m in enumerate(probs_bin[:valid_frames]):
            if m == 1 and not in_speech:
                temp = [i]  # сохраняем индекс начала сегмента
                in_speech = True
            elif m == 0 and in_speech:
                temp.append(i)  # сохраняем индекс окончания сегмента
                segments.append(temp)
                in_speech = False

        # Если сегмент остался активным до конца, завершаем его
        if in_speech:
            temp.append(valid_frames)
            segments.append(temp)

        # Объединение сегментов, если разрыв между ними меньше порога
        if self.merge_segments and len(segments) > 1:
            merged = [segments[0]]
            gap_threshold = self.merge_gap * self.sample_rate  # переводим merge_gap в отсчёты
            for current in segments[1:]:
                if (current[0] - merged[-1][1]) <= gap_threshold:
                    merged[-1][1] = current[1]
                else:
                    merged.append(current)
            segments = merged

        return np.array(segments)
