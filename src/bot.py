"""VK чат-бот для решения дифференциальных уравнений.

Пользователь пишет задание в чат → бот запускает Pipeline →
отвечает .md файлом с решением + PNG-графиком (если изоклины).

Запуск:
    set PYTHONPATH=src && python src/bot.py
"""

from __future__ import annotations

import logging
import random
import tempfile
import time
from pathlib import Path

import vk_api
from vk_api.longpoll import VkEventType, VkLongPoll
from vk_api.upload import VkUpload

from config import VK_KEY
from pipeline import Pipeline
from tools.markdown_tools import write_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("vk_bot")

HELP_TEXT = (
    "Привет! Я решаю дифференциальные уравнения.\n\n"
    "Просто напиши задание, например:\n"
    "• Реши уравнение y' = x*y\n"
    "• Реши разделением переменных y' = x*y\n"
    "• Построй изоклины для y' = x**2 + y\n"
    "• Классифицируй y'' + y = 0\n\n"
    "Команды:\n"
    "/help — эта справка\n"
)


def _make_temp_md(text: str) -> Path:
    """Создаёт временный .md файл с заданием в формате Obsidian."""
    tmp = Path(tempfile.mktemp(suffix=".md"))
    content = f"---\nstatus: pending\n---\n# Задание\n{text}\n"
    write_markdown(tmp, content)
    return tmp


class DiffEqBot:
    def __init__(self):
        self.vk_session = vk_api.VkApi(token=VK_KEY)
        self.vk = self.vk_session.get_api()
        self.upload = VkUpload(self.vk_session)
        self.longpoll = VkLongPoll(self.vk_session)
        self.pipeline = Pipeline(output_dir=Path(tempfile.gettempdir()) / "diffeq_output")

    def send_text(self, peer_id: int, text: str) -> None:
        self.vk.messages.send(
            peer_id=peer_id,
            message=text,
            random_id=random.randint(1, 2**31),
        )

    def _fresh_vk(self) -> tuple[vk_api.VkApi, VkUpload]:
        """Создаёт полностью новое VK-подключение (свежий HTTP-пул)."""
        session = vk_api.VkApi(token=VK_KEY)
        return session, VkUpload(session)

    def send_doc(self, peer_id: int, file_path: str, title: str | None = None, max_retries: int = 3) -> None:
        """Отправляет файл как документ. Каждая попытка — свежее соединение."""
        for attempt in range(max_retries):
            try:
                session, upload = self._fresh_vk()
                api = session.get_api()

                doc = upload.document_message(
                    doc=file_path,
                    title=title or Path(file_path).name,
                    peer_id=peer_id,
                )
                owner_id = doc["doc"]["owner_id"]
                doc_id = doc["doc"]["id"]
                api.messages.send(
                    peer_id=peer_id,
                    message="",
                    attachment=f"doc{owner_id}_{doc_id}",
                    random_id=random.randint(1, 2**31),
                )
                return
            except Exception as exc:
                log.warning("send_doc attempt %d failed: %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise

    def send_photo(self, peer_id: int, photo_path: str) -> None:
        """Отправляет фото."""
        response = self.upload.photo_messages(photos=photo_path, peer_id=peer_id)
        if response:
            photo = response[0]
            att = f"photo{photo['owner_id']}_{photo['id']}"
            self.vk.messages.send(
                peer_id=peer_id,
                message="",
                attachment=att,
                random_id=random.randint(1, 2**31),
            )

    def handle_message(self, peer_id: int, text: str) -> None:
        text_lower = text.strip().lower()

        if text_lower in ("/help", "помощь", "привет", "start", "/start"):
            self.send_text(peer_id, HELP_TEXT)
            return

        if len(text.strip()) < 5:
            self.send_text(peer_id, "Напиши задание подробнее. /help — справка.")
            return

        tmp_md = _make_temp_md(text.strip())
        try:
            result = self.pipeline.run(tmp_md)
        finally:
            tmp_md.unlink(missing_ok=True)

        if not result.success:
            self.send_text(
                peer_id,
                f"Не удалось решить.\n"
                f"Этап: {result.stage_failed}\n"
                f"Ошибка: {result.error}",
            )
            return

        if result.pdf_file and Path(result.pdf_file).exists():
            try:
                self.send_doc(peer_id, result.pdf_file, title="solution.pdf")
            except Exception as exc:
                log.warning("Failed to send PDF: %s", exc)
                self.send_text(peer_id, "Решение получено, но не удалось отправить файл.")

    def run(self) -> None:
        log.info("VK бот запущен, слушаю сообщения...")
        while True:
            try:
                self.longpoll = VkLongPoll(self.vk_session)
                for event in self.longpoll.listen():
                    if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.from_user:
                        peer_id = event.peer_id
                        text = event.text
                        log.info("Message from %s: %s", peer_id, text[:100])
                        try:
                            self.handle_message(peer_id, text)
                        except Exception as exc:
                            log.exception("Error handling message from %s", peer_id)
                            try:
                                self.send_text(peer_id, f"Внутренняя ошибка: {exc}")
                            except Exception:
                                pass
            except Exception as exc:
                log.warning("Long Poll disconnected: %s — reconnecting in 3s…", exc)
                time.sleep(3)
                self.vk_session = vk_api.VkApi(token=VK_KEY)
                self.vk = self.vk_session.get_api()
                self.upload = VkUpload(self.vk_session)


def main():
    if not VK_KEY:
        print("VK_KEY не задан в .env!")
        return
    bot = DiffEqBot()
    bot.run()


if __name__ == "__main__":
    main()
