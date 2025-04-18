import os
import nest_asyncio
nest_asyncio.apply()

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

##### LLAMAPARSE #####
from llama_parse import LlamaParse

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")


#llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/presentation.pptx")
#llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/uber_10q_march_2022.pdf")
#llama_parse_documents = LlamaParse(api_key=llamaparse_api_key, result_type="markdown").load_data("./data/state_of_union.txt")

import pickle
# Define a function to load parsed data if available, or parse if not

############################# WORKING CODE#################################
# from llama_index.core import SimpleDirectoryReader

# reader = SimpleDirectoryReader(input_dir="./llamaparser-example")
# documents = reader.load_data()

# print(f"Loaded {len(documents)} docs")
# print(documents[0].text[:500])

# llama_parse_documents = documents
################################################################################



from llama_index.readers.file import PDFReader
from pathlib import Path

pdf_path = Path("./llamaparser-example/Yokogawa REPORT FORMAT.pdf")
reader = PDFReader()
documents = reader.load_data(file=pdf_path)
print(f"Loaded {len(documents)} doc(s) from PDF")
if documents:
    print(documents[0].text[:100])  # preview the parsed text
llama_parse_documents = documents
######## QDRANT ###########

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

import qdrant_client

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

######### FastEmbedEmbeddings #############

# by default llamaindex uses OpenAI models
from llama_index.embeddings.fastembed import FastEmbedEmbedding
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

""" embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    #model_name="llama2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
) """

#### Setting embed_model other than openAI ( by default used openAI's model)
from llama_index.core import Settings

Settings.embed_model = embed_model

######### Groq API ###########

from llama_index.llms.groq import Groq
groq_api_key = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)
#llm = Groq(model="gemma-7b-it", api_key=groq_api_key)

######### Ollama ###########

#from llama_index.llms.ollama import Ollama  # noqa: E402
#llm = Ollama(model="llama2", request_timeout=30.0)

#### Setting llm other than openAI ( by default used openAI's model)
Settings.llm = llm

client = qdrant_client.QdrantClient(api_key=qdrant_api_key, url=qdrant_url,)

from qdrant_client.http import models as rest

client.recreate_collection(
    collection_name="qdrant_rag",
    vectors_config=rest.VectorParams(
        size=768,  # vector size of your embedding model
        distance=rest.Distance.COSINE,
    ),
)

vector_store = QdrantVectorStore(client=client, collection_name='qdrant_rag', create_collection_if_missing=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents=llama_parse_documents, storage_context=storage_context, show_progress=True)

#### PERSIST INDEX #####
#index.storage_context.persist()

#storage_context = StorageContext.from_defaults(persist_dir="./storage")
#index = load_index_from_storage(storage_context)

# create a query engine for the index
query_engine = index.as_query_engine()

# query the engine

# query = "what does a License mean?"
# query1 = "name one of the people the students must thank"
# response = query_engine.query(query1)
# print(f"Parsed documents count: {len(llama_parse_documents)}")
# print(llama_parse_documents[0].text[:500])  # Preview first few lines
# print(response)

# print(f"Parsed docs: {len(llama_parse_documents)}")
# if llama_parse_documents:
#     print(llama_parse_documents[0].text[:500])
# for i, doc in enumerate(documents):
#     print(f"[Doc {i}] Metadata: {doc.metadata}")


####################################### TTS - STT #########################################
from groq import Groq
import os, json

client_og = Groq(api_key=os.getenv("GROQ_og_API_KEY"))

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
        return transcription.text  # or return full object for timestamps

def synthesize_speech(text: str, output_path="response.wav", voice="Arista-PlayAI"):
    response = client_og.audio.speech.create(
        model="playai-tts",
        voice=voice,
        input=text,
        response_format="wav"
    )
    response.write_to_file(output_path)

import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="query.wav", duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"Saved to {filename}")

import simpleaudio as sa

def play_audio(filepath="response.wav"):
    try:
        wave_obj = sa.WaveObject.from_wave_file(filepath)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        print("✅ Playback complete.")
    except Exception as e:
        print("⚠️ Audio playback failed:", e)
        
# Record or load audio
audio_path = "query.wav"

# Step 0: Record audio
record_audio(filename=audio_path, duration=5)
# Step 1: Speech-to-Text
query_text = transcribe_audio(audio_path)
print("You said:", query_text)

# Step 2: Query Engine
response = query_engine.query(query_text)
print("LLM Response:", response)

# Step 3: Text-to-Speech
synthesize_speech(str(response), output_path="response.wav")

# Step 4: Play the result (optional)
import playsound
play_audio("response.wav")