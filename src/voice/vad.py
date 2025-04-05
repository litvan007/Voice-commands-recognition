import numpy as np
import logging
import onnxruntime

logger = logging.getLogger(__name__)


class EmobaseCNN:
    """
    VAD-модель на ONNX. Принимает готовые признаки [frames, features].
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.threshold = threshold
        self.session = None

        if model_path:
            try:
                self.session = onnxruntime.InferenceSession(model_path)
                logger.info(f"✅ VAD-модель загружена: {model_path}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки VAD-модели: {e}")
        else:
            logger.warning("⚠️ Путь к VAD-модели не указан")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Принимает признаки формы (frames, features), возвращает бинарную маску.
        """
        input_tensor = features[np.newaxis, ...]  # (1, frames, features)

        if self.session:
            try:
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: input_tensor})[0]
                probs = np.squeeze(output)
                return (probs >= self.threshold).astype(np.uint8)
            except Exception as e:
                logger.error(f"Ошибка инференса VAD: {e}")
        return np.zeros(features.shape[0], dtype=np.uint8)

    def segment(self, mask: np.ndarray, signal_len: int, sample_rate: int) -> list[tuple[int, int]]:
        """
        Делит маску [0/1] на сегменты (start, end) в сэмплах.
        """
        frame_shift = int(0.01 * sample_rate)  # 10 мс шаг
        segments = []
        in_segment = False
        start = 0

        for i, flag in enumerate(mask):
            if flag and not in_segment:
                in_segment = True
                start = i * frame_shift
            elif not flag and in_segment:
                in_segment = False
                end = i * frame_shift
                segments.append((start, end))
        if in_segment:
            segments.append((start, signal_len))
        return segments
