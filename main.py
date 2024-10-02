"""
Для запуска скрипта потребуется установить библиотеку gradio_client.
Сделать это можно с помощью команды pip install gradio_client или pip install -r requirements.txt

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from gradio_client import Client


class GPTClient(ABC):
    instance = None
    client = None

    @abstractmethod
    def get_response(self, query) -> str:
        pass

    # singleton
    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance


class HuggingfaceClient(GPTClient):

    def __init__(self):
        super().__init__()
        if not self.instance.client:
            self.instance.client = Client("huggingface-projects/llama-3.2-3B-Instruct")

    def get_response(self, query) -> str:
        result = self.instance.client.predict(
            message=query,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            api_name="/chat"
        )
        return result


client = HuggingfaceClient()
while True:
    query = input("Запрос: ")
    if not query:
        continue

    response = client = client.get_response(query)
    print(f"Ответ: {response}")
