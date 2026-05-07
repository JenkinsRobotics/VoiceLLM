from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, keywords):
        self.m = Model(wakeword_models=keywords)  # downloads on first use
    def score(self, audio_16k_int16):
        # expects float32 [-1,1]; convert
        x = audio_16k_int16.astype('float32') / 32768.0
        scores = self.m.predict(x)
        # return top keyword and score
        best_kw = max(scores, key=scores.get)
        return best_kw, scores[best_kw]