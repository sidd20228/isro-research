from __future__ import annotations

import random
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode

PARAMETER_BLOCK = re.compile(r"([?&])([^#\n]+)")


@dataclass
class RequestAugmenter:
    """Generate semantics-preserving request-text views."""

    probability: float = 0.5

    def augment(self, request_text: str) -> str:
        output = request_text
        if random.random() < self.probability:
            output = self._shuffle_parameters(output)
        if random.random() < self.probability:
            output = re.sub(r"[ \t]+", lambda _: " " * random.randint(1, 3), output)
        if random.random() < self.probability:
            output = output.replace("%20", " ")
        return output

    @staticmethod
    def _shuffle_parameters(value: str) -> str:
        lines = value.splitlines()
        shuffled: list[str] = []
        for line in lines:
            if "?" not in line:
                shuffled.append(line)
                continue
            prefix, query = line.split("?", maxsplit=1)
            pairs = parse_qsl(query, keep_blank_values=True)
            random.shuffle(pairs)
            shuffled.append(f"{prefix}?{urlencode(pairs)}")
        return "\n".join(shuffled)
