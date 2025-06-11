import streamlit as st
import spotipy
from sklearn.cluster import KMeans
import numpy as np
import openai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---- FUNCTIONS ----
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

# ---- UI ----
st.set_page_config(page_title="Judge My Music Taste", layout="centered")
st.title("🎧 Judge My Music Taste - AI Style")

st.write("### Step 1: Paste Your Spotify Access Token")
st.markdown("Get it here → [Spotify Console](https://developer.spotify.com/console/get-current-user-top-artists-and-tracks/?type=tracks)")

token = st.text_input("Paste your **Spotify OAuth token** (Bearer token)", type="password")

if token:
    try:
        sp = spotipy.Spotify(auth=token)
        st.success("Token accepted! Fetching your top tracks...")

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
        st.error(f"Failed to fetch data: {e}")
