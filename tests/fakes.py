from types import SimpleNamespace


class FakeAsyncOpenAI:
    """Minimal AsyncOpenAI stand-in with a recording chat.completions.create."""

    def __init__(self, respond):
        self._respond = respond
        self.calls = []

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                return self._outer._respond(**kwargs)

        self.chat = SimpleNamespace(completions=_Completions(self))


def make_response(content, prompt_tokens=10, completion_tokens=5):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)
