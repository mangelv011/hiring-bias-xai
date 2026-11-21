"""Fakes for tests: classifiers that satisfy `models.Classifier`, and a local LLM endpoint."""

import http.server
import json
import threading
import zlib

import numpy as np

from hiring_bias_xai import data


class FakeClassifier:
    """Records every `logits` call; subclasses define `score(text)` as the 28 logits."""

    labels = data.OCCUPATIONS

    def __init__(self):
        self.calls = []

    def logits(self, texts):
        self.calls.append(list(texts))
        return np.stack([self.score(text) for text in texts]).astype(np.float32)

    def predict(self, texts):
        return [self.labels[i] for i in self.logits(texts).argmax(axis=1)]


class BagOfWords(FakeClassifier):
    """Logit of an occupation = sum of the weights its table gives to the words of the text,
    with no normalisation; every other logit is 0. The log-odds of that occupation are then the
    sum of the word weights minus the constant log(27)."""

    def __init__(self, weights_by_occupation):
        super().__init__()
        self.weights_by_occupation = weights_by_occupation

    def score(self, text):
        logits = np.zeros(28)
        for occupation, weights in self.weights_by_occupation.items():
            logits[data.OCCUPATIONS.index(occupation)] = sum(
                weights.get(word, 0.0) for word in text.split()
            )
        return logits


class Hashed(FakeClassifier):
    """Arbitrary nonlinear logits that depend only on the text."""

    def score(self, text):
        return np.random.default_rng(zlib.crc32(text.encode())).normal(size=28) * 3


class FakeLLMServer:
    """An OpenAI-compatible chat endpoint on a free local port.

    `answer(payload)` returns the `message` dict for a request; every payload is kept in
    `requests`. Use it as a context manager and read the address from `base_url`.
    """

    def __init__(self, answer):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(
                    {"path": self.path, "payload": payload, "headers": dict(self.headers)}
                )
                body = json.dumps({"choices": [{"message": answer(payload)}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
