# streamlit_app.py
import os
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
import json
import pickle
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import streamlit as st
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

from llama_parse import LlamaParse
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

import qdrant_client
from qdrant_client.http import models as rest

from groq import Groq as GroqClient
# import sounddevice as sd
from scipy.io.wavfile import write
# import simpleaudio as sa
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recorded_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten()
        self.recorded_frames.append(audio)
        return frame

def save_audio(frames, filename="query.wav", sample_rate=48000):
    if frames:
        audio_data = np.concatenate(frames)
        write(filename, sample_rate, audio_data)
        st.success(f"✅ Saved to {filename}")
    else:
        st.warning("⚠️ No audio data recorded.")

# Load API keys
llamaparse_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]
groq_og_api_key = st.secrets["GROQ_og_API_KEY"]

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("🧠 Say Hello! to EducAIt 📖")
st.subheader("The simplest AI-powered Accessibility Learning Assistant")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with open("uploaded_doc.pdf", "wb") as f:
        f.write(uploaded_file.read())

    reader = PDFReader()
    documents = reader.load_data(file=Path("uploaded_doc.pdf"))
    st.success(f"Loaded {len(documents)} document(s)")

    if documents:
        st.text_area("Preview", documents[0].text[:1000], height=200)

        # Setup embedding model and LLM
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
        Settings.embed_model = embed_model

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
        Settings.llm = llm

        # Qdrant setup
        client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url)
        client.recreate_collection(
            collection_name="qdrant_rag",
            vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE),
        )

        vector_store = QdrantVectorStore(
            client=client, collection_name='qdrant_rag', create_collection_if_missing=True
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents, storage_context=storage_context, show_progress=True
        )
        query_engine = index.as_query_engine(streaming=False, similarity_top_k=3, response_mode="compact")

        # Voice Interaction
        st.header("🎤 Ask EducAIt a question via voice or text")
        duration = st.slider("Recording Duration (seconds)", 2, 10, 5)
        col1, col2 = st.columns([1, 2])
        query_text = ""

        with col1:
            if st.button("🎙 Record Question"):
                st.info("Recording...")
                # recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='int16')
                # sd.wait()
                ctx = webrtc_streamer(
                    key="mic",
                    mode=WebRtcMode.SENDONLY,
                    audio_processor_factory=AudioProcessor,
                    media_stream_constraints={"audio": True, "video": False},
                    async_processing=True,
                )
                if ctx.audio_processor and st.button("🛑 Save Recording"):
                    save_audio(ctx.audio_processor.recorded_frames)
                    st.success("Audio saved as query.wav")
                    client_og = GroqClient(api_key=groq_og_api_key)

                def transcribe_audio(audio_path: str, prompt: str = "") -> str:
                    with open(audio_path, "rb") as file:
                        transcription = client_og.audio.transcriptions.create(
                            file=file,
                            model="whisper-large-v3-turbo",
                            prompt=prompt,
                            response_format="verbose_json",
                            timestamp_granularities=["segment"],
                            language="en",
                            temperature=0.0
                        )
                        return transcription.text

                query_text = transcribe_audio("query.wav")
                st.write(f"You asked (via voice): {query_text}")

        with col2:
            text_input = st.text_input("Or type your question")
            if text_input:
                query_text = text_input

        if query_text:
            response = query_engine.query(query_text)
            answer = str(response)

            # Show response and sources
            st.subheader("📘 Answer")
            st.write(answer)

            if hasattr(response, 'source_nodes'):
                st.markdown("### 🔍 Sources")
                for i, node in enumerate(response.source_nodes):
                    st.markdown(f"**Source {i+1}:** {node.node.text[:300]}...")

            # Add to chat history
            st.session_state.chat_history.append((query_text, answer))

            # TTS
            def synthesize_speech(text: str, output_path="response.wav", voice="Arista-PlayAI"):
                response = client_og.audio.speech.create(
                    model="playai-tts",
                    voice=voice,
                    input=text,
                    response_format="wav"
                )
                response.write_to_file(output_path)

            synthesize_speech(answer)
            st.audio("response.wav")

        if st.session_state.chat_history:
            st.markdown("## 🗂 Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
                with st.expander(f"Q{i+1}: {q}"):
                    st.write(a)
