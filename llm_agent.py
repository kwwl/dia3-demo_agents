from groq import Groq

from dotenv import load_dotenv
import os


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

    def get_conversation_history(self):
        return self.conversation_history[1:]

    def update_conversation_history(self, role, content):
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    def ask_llm(self, text_interaction):

        self.update_conversation_history(role="user", content=text_interaction)

        response = (
            self.groq_client.chat.completions.create(
                messages=self.conversation_history,
                model="openai/gpt-oss-120b",
                include_reasoning=False,
            )
            .choices[0]
            .message.content
        )

        self.update_conversation_history(role="assistant", content=response)

    def transcribe_audio(self, audio_file):
        transcription = self.groq_client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.getvalue()),
            model="whisper-large-v3-turbo",
            language="fr",
        )
        return transcription.text


if __name__ == "__main__":
    question = input("Pose ta question :\n")
    response = ask_llm(question)
    print("\n")
    print(response)
