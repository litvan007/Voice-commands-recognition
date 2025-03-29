import onnxruntime as ort
import numpy as np
from settings import Settings
from utils import softmax
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

class SpeechCommandModel:
    """
    ONNX-инференс-класс для голосовой модели команд с attention и BLSTM.

    Поддерживает стандартный ML-интерфейс:
    - predict: вернуть метку команды
    - predict_proba: вернуть распределение по классам
    - predict_top_k: вернуть top-k вероятных команд

    Архитектура: CNN → Residual CNN → Linear → BLSTM → Attention → Classifier
    """

    def __init__(self, settings: Settings):
        self.labels_map = settings.voice_command_config.labels_map
        self.thresholds = settings.voice_command_config.confidence_thresholds
        self.debug = settings.debug

        model_path = settings.model_paths.command_recognizer
        logger.info(f"Загрузка ONNX-модели команд: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, mfcc: np.ndarray) -> np.ndarray:
        x = mfcc.T  # [time, features]
        x = np.expand_dims(x, axis=0)      # [1, time, features]
        x = np.expand_dims(x, axis=0)      # [1, 1, time, features]
        return x.astype(np.float32)

    def predict_proba(self, mfcc: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess(mfcc)
        logits = self.session.run(None, {self.input_name: input_tensor})[0]
        probs = softmax(logits[0])
        return probs

    def predict(self, mfcc: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Возвращает: (метка команды, уверенность) или (None, p), если уверенность ниже порога.
        """
        probs = self.predict_proba(mfcc)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        threshold = self.thresholds.overrides.get(pred_idx, self.thresholds.default)

        if self.debug:
            logger.debug("Top-3 вероятности:")
            for idx in probs.argsort()[::-1][:3]:
                logger.debug(f"  {self.labels_map[idx]:<12} — {probs[idx]:.4f}")

        if confidence >= threshold:
            return self.labels_map[pred_idx], confidence
        else:
            logger.info(f"Команда отвергнута (p={confidence:.2f} < threshold={threshold})")
            return None, confidence

    def predict_top_k(self, mfcc: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Возвращает топ-k меток с их вероятностями.
        """
        probs = self.predict_proba(mfcc)
        top_indices = probs.argsort()[::-1][:k]
        return [(self.labels_map[i], float(probs[i])) for i in top_indices]
