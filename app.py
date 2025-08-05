import streamlit as st
import os
import tempfile
import uuid
import ffmpeg
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# App title
st.title("🎥 Movie Explorer with OpenCLIP + ChromaDB")

# ChromaDB setup
@st.cache_resource
def get_chroma():
    client = chromadb.Client()
    embedding_fn = OpenCLIPEmbeddingFunction()
    loader = ImageLoader()
    return client, embedding_fn, loader

client, embedding_function, data_loader = get_chroma()

# Upload video section
st.header("📤 Upload a Movie")
uploaded_video = st.file_uploader("Upload a movie file", type=["mp4", "mov", "avi", "mkv"])
movie_title = st.text_input("Movie title")

if uploaded_video and movie_title:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        frame_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        st.info("Extracting 1 frame per second...")

        (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=1)
            .output(os.path.join(frame_dir, "frame_%04d.jpg"), start_number=0)
            .run(quiet=True, overwrite_output=True)
        )

        # Register movie collection
        collection_name = f"movie_{movie_title.lower().replace(' ', '_')}"
        if collection_name in [c.name for c in client.list_collections()]:
            collection = client.get_collection(collection_name)
        else:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                data_loader=data_loader
            )

        # Add frames to collection
        frame_files = sorted(os.listdir(frame_dir))
        ids, filepaths, metadatas = [], [], []

        for i, frame in enumerate(frame_files):
            frame_path = os.path.join(frame_dir, frame)
            uid = str(uuid.uuid4())
            ids.append(uid)
            filepaths.append(frame_path)
            metadatas.append({
                "movie": movie_title,
                "frame": i
            })

        collection.add(
            ids=ids,
            images=filepaths,
            metadatas=metadatas
        )

        st.success(f"{len(filepaths)} frames added for movie '{movie_title}'")

# Query section
st.header("🔍 Query the Movie Embeddings")
query_text = st.text_input("Enter a text prompt (e.g. 'a man with a hat')")
num_results = st.slider("Number of results per movie", 1, 10, 3)
mode = st.radio("Search mode", ["Single Movie", "Compare All Movies"])

if query_text:
    if mode == "Single Movie":
        all_collections = [c.name for c in client.list_collections()]
        selected = st.selectbox("Choose a movie", all_collections)
        if selected:
            col = client.get_collection(selected)
            result = col.query(query_texts=[query_text], n_results=num_results)

            st.subheader(f"🎞️ Results for {selected.replace('movie_', '').replace('_', ' ').title()}")
            for i, metadata in enumerate(result['metadatas'][0]):
                st.markdown(f"**Frame {metadata['frame']}** from *{metadata['movie']}*")
    else:
        st.subheader("📊 Comparing results across all movies")
        for col_meta in client.list_collections():
            col = client.get_collection(col_meta.name)
            result = col.query(query_texts=[query_text], n_results=num_results)

            st.markdown(f"### 🎬 {col_meta.name.replace('movie_', '').replace('_', ' ').title()}")
            for i, metadata in enumerate(result['metadatas'][0]):
                st.markdown(f"- Frame {metadata['frame']} (Movie: {metadata['movie']})")
