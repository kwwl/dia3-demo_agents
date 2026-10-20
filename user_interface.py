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
                st.markdown(message["content"])

    def get_text_interaction(self, user_input):
        if user_input.audio:
            return self.conversation_agent_object.transcribe_audio(user_input.audio)
        return user_input.text

    def run(self):
        user_input = st.chat_input(
            "Que veux tu moussaillon ?", accept_file=False, accept_audio=True
        )

        if user_input:
            text_interaction = self.get_text_interaction(user_input)
            if text_interaction:
                self.conversation_agent_object.ask_llm(text_interaction)

        self.show_discussion_history()


if __name__ == "__main__":
    user_interface_object = UserInterface()
    user_interface_object.run()
