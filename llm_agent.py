from groq import Groq

from dotenv import load_dotenv
import base64
import os


TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.8-27b"
MAX_IMAGES_PER_REQUEST = 3


class ConversationAgent:
    def __init__(self):
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ["GROQ_KEY"])
        self.conversation_history = [
            {"role": "system", "content": self.read_file(file_path="./context.txt")}
        ]

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()

    @staticmethod
    def encode_image(image_file):
        image_base64 = base64.b64encode(image_file.getvalue()).decode("utf-8")
        return f"data:{image_file.type};base64,{image_base64}"

    @staticmethod
    def has_image(message):
        return isinstance(message["content"], list)

    def get_conversation_history(self):
        return self.conversation_history[1:]

    def update_conversation_history(self, role, content):
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_messages_for_llm(self):
        # Le modele vision accepte au maximum 3 images par requete :
        # on garde les plus recentes et on remplace les autres par du texte.
        messages = []
        images_kept = 0

        for message in reversed(self.conversation_history):
            if self.has_image(message):
                if images_kept < MAX_IMAGES_PER_REQUEST:
                    images_kept += 1
                else:
                    text = message["content"][0]["text"]
                    message = {
                        "role": message["role"],
                        "content": f"{text} [image envoyée plus tôt]",
                    }
            messages.append(message)

        return list(reversed(messages))

    def ask_llm(self, text_interaction, image_file=None):

        if image_file:
            content = [
                {"type": "text", "text": text_interaction},
                {
                    "type": "image_url",
                    "image_url": {"url": self.encode_image(image_file)},
                },
            ]
        else:
            content = text_interaction

        self.update_conversation_history(role="user", content=content)

        conversation_has_image = any(
            self.has_image(message) for message in self.conversation_history
        )

        if conversation_has_image:
            response = self.groq_client.chat.completions.create(
                messages=self.get_messages_for_llm(),
                model=VISION_MODEL,
                reasoning_format="hidden",
            )
        else:
            response = self.groq_client.chat.completions.create(
                messages=self.conversation_history,
                model=TEXT_MODEL,
                include_reasoning=False,
            )

        response = response.choices[0].message.content

        self.update_conversation_history(role="assistant", content=response)

        return response

    def transcribe_audio(self, audio_file):
        transcription = self.groq_client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.getvalue()),
            model="whisper-large-v3-turbo",
            language="fr",
        )
        return transcription.text


if __name__ == "__main__":
    agent = ConversationAgent()
    question = input("Pose ta question :\n")
    print("\n" + agent.ask_llm(question))
