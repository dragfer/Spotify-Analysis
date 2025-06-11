import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.cluster import KMeans
import numpy as np
import openai
import pandas as pd

# ---- CONFIGURATION ----
# Replace with your credentials
SPOTIPY_CLIENT_ID = '16109f89727a4d24be39a9e488746953'
SPOTIPY_CLIENT_SECRET = '1ffd776c89e649deba6617d8366ff76b'
SPOTIPY_REDIRECT_URI = 'https://localhost:8888/callback'
openai.api_key = os.getenv("OPENAI_API_KEY")

SCOPE = 'user-top-read'

# ---- HELPER FUNCTIONS ----
def authenticate():
    auth_manager = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def fetch_top_tracks(sp, limit=20):
    results = sp.current_user_top_tracks(limit=limit, time_range='medium_term')
    tracks = results['items']
    track_ids = [track['id'] for track in tracks]
    return track_ids, [f"{t['name']} - {t['artists'][0]['name']}" for t in tracks]

def analyze_tracks(sp, track_ids):
    features = sp.audio_features(track_ids)
    return features

def cluster_taste(features):
    data = np.array([[f['danceability'], f['energy'], f['valence']] for f in features if f])
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
    cluster = kmeans.predict([data.mean(axis=0)])
    labels = ['Melancholy Maestro', 'Chaos Goblin', 'Vibe Curator', 'Pop Princess']
    return labels[cluster[0]], data

def get_gpt_judgment(stats_summary):
    prompt = f"""
    My Spotify music stats: {stats_summary}.
    Give me a witty, sarcastic yet intelligent judgment about my taste in music.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Judge My Music Taste", layout="centered")
st.title("🎧 Judge My Music Taste - AI Style")

if st.button("Authenticate with Spotify"):
    try:
        sp = authenticate()
        st.success("Authenticated! Fetching your top tracks...")

        track_ids, track_names = fetch_top_tracks(sp)
        st.write("### Your Top Tracks")
        st.write(track_names)

        features = analyze_tracks(sp, track_ids)
        df = pd.DataFrame(features)
        st.write("### Audio Features of Top Tracks")
        st.dataframe(df[['danceability', 'energy', 'valence', 'tempo']])

        label, data = cluster_taste(features)
        st.write(f"### Your Taste Cluster: **{label}**")

        summary = {
            'Danceability': float(np.mean(df['danceability'])),
            'Energy': float(np.mean(df['energy'])),
            'Valence': float(np.mean(df['valence'])),
            'Tempo': float(np.mean(df['tempo'])),
        }

        st.write("### AI Judgment")
        gpt_output = get_gpt_judgment(summary)
        st.success(gpt_output)

    except Exception as e:
        st.error(f"Error: {e}")
