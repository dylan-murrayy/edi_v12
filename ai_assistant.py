import io
import base64
import streamlit as st
import pandas as pd
import openai
from openai import AssistantEventHandler
from openai.types.beta.threads import Text, TextDelta
from PIL import Image

def ai_assistant_tab(df_filtered):
    # Custom CSS to make the input bar sticky
    st.markdown("""
        <style>
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 20px;
            width: 100%;
            background-color: #0F1117;
            padding: 10px;
            z-index: 100;
            box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
        }
        .main .block-container {
            padding-bottom: 150px;  /* Adjust this value if needed */
        }
        </style>
        """, unsafe_allow_html=True)

    st.header("AI Assistant")
    st.write("Ask questions about your data, and the assistant will analyze it using Python code.")

    # Initialize OpenAI client using Streamlit secrets
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]
    except KeyError as e:
        st.error(f"Missing secret: {e}")
        st.stop()

    client = openai.Client(api_key=openai_api_key)

    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
    except Exception as e:
        st.error(f"Failed to retrieve assistant: {e}")
        st.stop()

    # Convert dataframe to a CSV file using io.BytesIO
    csv_buffer = io.BytesIO()
    df_filtered.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Reset buffer position to the start

    # Upload the CSV file as binary data
    try:
        file = client.files.create(
            file=csv_buffer,
            purpose='assistants'
        )
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        st.stop()

    # Update the assistant to include the file
    try:
        client.beta.assistants.update(
            assistant_id,
            tool_resources={
                "code_interpreter": {
                    "file_ids": [file.id]
                }
            }
        )
    except Exception as e:
        st.error(f"Failed to update assistant with file resources: {e}")
        st.stop()

    # Initialize session state variables
    if 'thread_id' not in st.session_state:
        try:
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
        except Exception as e:
            st.error(f"Failed to create thread: {e}")
            st.stop()

    # Create a container for the chat messages
    chat_container = st.container()

    # Display chat history in the container
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    if 'content' in message:
                        st.markdown(message['content'], unsafe_allow_html=True)
                    if 'code' in message:
                        st.code(message['code'], language='python')
                    if 'output' in message:
                        st.code(message['output'])
                    if 'image' in message:
                        st.image(message['image'], use_column_width=True)

    # User input
    if prompt := st.chat_input("Enter your question about the data"):
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})

        # Display the user's message immediately
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # Create a new message in the thread
        try:
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )
        except Exception as e:
            st.error(f"Failed to create message in thread: {e}")
            st.stop()

        # Define event handler to capture assistant's response
        class MyEventHandler(AssistantEventHandler):
            def __init__(self, chat_container, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.chat_container = chat_container
                self.assistant_message = ""
                self.code_input = ""
                self.code_output = ""
                self.tool_call_active = False
                # Create placeholders
                self.content_placeholder = None
                self.code_placeholder = None
                self.output_placeholder = None

            def on_text_delta(self, delta: TextDelta, snapshot: Text, **kwargs):
                if delta and delta.value:
                    self.assistant_message += delta.value
                    # Update the assistant's message content
                    if not self.content_placeholder:
                        with self.chat_container:
                            with st.chat_message("assistant"):
                                self.content_placeholder = st.empty()
                    self.content_placeholder.markdown(self.assistant_message)

            def on_tool_call_created(self, tool_call):
                self.tool_call_active = True
                # Create a new message for code input
                if not self.code_placeholder:
                    with self.chat_container:
                        with st.chat_message("assistant"):
                            self.code_expander = st.expander("Assistant is coding...")
                            self.code_placeholder = st.empty()
                self.code_input = ""

            def on_tool_call_delta(self, delta, snapshot, **kwargs):
                if delta.type == "code_interpreter" and delta.code_interpreter:
                    if delta.code_interpreter.input:
                        self.code_input += delta.code_interpreter.input
                        # Update code placeholder
                        self.code_placeholder.code(self.code_input, language='python')

                    if delta.code_interpreter.outputs:
                        for output in delta.code_interpreter.outputs:
                            if output.type == "logs":
                                if not self.output_placeholder:
                                    with self.chat_container:
                                        with st.chat_message("assistant"):
                                            self.output_expander = st.expander("Code Output")
                                            self.output_placeholder = st.empty()
                                self.code_output += output.logs
                                self.output_placeholder.code(self.code_output)

            def on_tool_call_done(self, tool_call):
                self.tool_call_active = False
                self.code_input = ""
                self.code_output = ""
                # Optionally, close the expander
                if hasattr(self, 'code_expander'):
                    self.code_expander.expanded = False

            def on_image_file_done(self, image_file):
                """
                Handle image files generated by the assistant.
                """
                try:
                    # Download the image from OpenAI
                    image_data = client.files.content(image_file.file_id).read()
                    image = Image.open(io.BytesIO(image_data))

                    # Convert image to a format suitable for Streamlit
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue()

                    # Display the image in the chat
                    with self.chat_container:
                        with st.chat_message("assistant"):
                            st.image(img_bytes, use_column_width=True)

                except Exception as e:
                    st.error(f"Failed to process image file: {e}")

        # Instantiate the event handler
        event_handler = MyEventHandler(chat_container)

        # Run the assistant
        try:
            with client.beta.threads.runs.stream(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_id,
                event_handler=event_handler,
                temperature=0
            ) as stream:
                stream.until_done()
        except Exception as e:
            st.error(f"Failed to run assistant stream: {e}")
            st.stop()

        # Add assistant's message to chat history
        assistant_entry = {'role': 'assistant', 'content': event_handler.assistant_message}
        if event_handler.code_input:
            assistant_entry['code'] = event_handler.code_input
        if event_handler.code_output:
            assistant_entry['output'] = event_handler.code_output
        st.session_state.chat_history.append(assistant_entry)