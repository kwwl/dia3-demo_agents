import base64

import streamlit as st
from llm_agent import ConversationAgent


class UserInterface:
    def __init__(self):
        st.set_page_config(page_title="Jarvis")
        st.title("Barbe-Quasar")
        self.conversation_agent_object = self.get_persistant_conversation_agent()

    def get_persistant_conversation_agent(self):

        if "conversation_agent_object" not in st.session_state:
            st.session_state.conversation_agent_object = ConversationAgent()

        return st.session_state.conversation_agent_object

    def show_discussion_history(self):
        for message in self.conversation_agent_object.get_conversation_history():
            with st.chat_message(message["role"]):
                if isinstance(message["content"], list):
                    text = message["content"][0]["text"]
                    image_url = message["content"][1]["image_url"]["url"]
                    st.markdown(text)
                    image_bytes = base64.b64decode(image_url.split(",")[1])
                    st.image(image_bytes, width=300)
                else:
                    st.markdown(message["content"])

    def get_text_interaction(self, user_input):
        if user_input.audio:
            return self.conversation_agent_object.transcribe_audio(user_input.audio)
        return user_input.text

    def get_image_file(self, user_input):
        if user_input.files:
            return user_input.files[0]
        return None

    def run(self):
        user_input = st.chat_input(
            "Que veux tu moussaillon ?",
            accept_file=True,
            file_type=["png", "jpg", "jpeg", "webp"],
            accept_audio=True,
        )

        if user_input:
            text_interaction = self.get_text_interaction(user_input)
            image_file = self.get_image_file(user_input)

            if image_file and not text_interaction:
                text_interaction = "Que vois-tu sur cette image ?"

            if text_interaction:
                self.conversation_agent_object.ask_llm(text_interaction, image_file)

        self.show_discussion_history()


if __name__ == "__main__":
    user_interface_object = UserInterface()
    user_interface_object.run()
