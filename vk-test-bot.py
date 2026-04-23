import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import os
from dotenv import load_dotenv

load_dotenv()

def send_message(vk, user_id, text, random_id=0):
    vk.messages.send(
        user_id=user_id,
        message=text,
        random_id=random_id
    )

def main():
    token = os.getenv("VK_KEY")
    
    vk_session = vk_api.VkApi(token=token)
    vk = vk_session.get_api()
    
    longpoll = VkLongPoll(vk_session)

    print("Бот запущен и слушает сообщения...")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_id = event.user_id
            text = event.text.lower()
            
            if text == 'привет':
                send_message(vk, user_id, "Привет! Я простой бот ВК.")
            elif text == 'пока':
                send_message(vk, user_id, "До свидания!")
            else:
                send_message(vk, user_id, f"Вы написали: {event.text}")

if __name__ == '__main__':
    main()