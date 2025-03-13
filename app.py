import streamlit as st
from audiorecorder import audiorecorder
from io import BytesIO
from dotenv import dotenv_values
from openai import OpenAI

# Modele AI
TEXT_TO_EMBEDDINGS = "text-to-embeddings-3-large"
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
TEXT_TO_TEXT = "gpt-4o"

# Zakładki
SAY_OR_WRITE = "Napisz lub powiedz coś o swoich preferencjach"
FORM = "Uzupełnij formularz, gdzie chciałbyś polecieć"

env=dotenv_values(".env")
#Funkcja od klucza OPEN_AI
def get_openai_client():
   return OpenAI(api_key=env["OPENAI_API_KEY"])
# Funkcja do obsługi audio
def get_audio(audio):
    audio_file = BytesIO()
    audio.export(audio_file, format="mp3")
    return audio_file.getvalue()

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )

    return transcript.text
#
## st.session_state
# 

# Ustawienie ekranu startowego przy pierwszym uruchomieniu
if "page" not in st.session_state:
    st.session_state["page"] = "start"
if "user_text" not in st.session_state:
    st.session_state["user_text"]=""
# Ustawianie sesion state z audio
if "audio" not in st.session_state:
    st.session_state["audio"]=None

# Ustawianie sesion state textu z audio
if "text_from_audio" not in st.session_state:
    st.session_state["text_from_audio"]=""

#
## MAIN
#

# Obsługa ekranu startowego
if st.session_state["page"] == "start":
    st.title("🛫 Flight4u 🔍🌍")

    st.markdown(
        """
        ## Witaj w aplikacji Flight4u! 🌍✈️  
        Razem znajdziemy dla Ciebie idealne miejsce na wypoczynek!  
        """
    )

    if st.button("Kontynuuj"):
        st.session_state["page"] = "main"
        st.rerun()

# Obsługa głównej części aplikacji
elif st.session_state["page"] == "main":
    st.title("🛫 Flight4u – Znajdź swój wymarzony lot! 🌍")

    # Wybór sposobu wyszukiwania lotu
    tab_select = st.selectbox(
        "Wybierz opcję, jak chcesz znaleźć idealny lot",
        [SAY_OR_WRITE, FORM],
        index=0  # Ustawienie domyślnej wartości
    )

    # Opcja "Napisz lub powiedz coś o swoich preferencjach"
    if tab_select == SAY_OR_WRITE:
        st.markdown("### Powiedz lub napisz, gdzie chcesz polecieć ✈️")
        
        user_audio = audiorecorder(
            start_prompt="🎤 Powiedz",
            stop_prompt="⏹ Zakończ"
        )

        if user_audio:
            st.session_state["audio"]=get_audio(user_audio)
            if st.button("TRANSKRYUJ"):
                st.session_state["text_from_audio"]=transcribe_audio(st.session_state["audio"])
                st.session_state["user_text"]=st.session_state["text_from_audio"]

        st.text_area(
            label="",
            label_visibility="hidden",
            placeholder=(
                "Powiedz, gdzie lubisz podróżować, jaki klimat Cię interesuje, "
                "na ile dni chciałbyś wylecieć, o której chciałbyś mieć wylot w tamtą i powrotną stronę..."
            ),
            key="user_text",  # Dodajemy `key`, by Streamlit automatycznie śledził zmiany
        )

        st.markdown("###### Jeżeli wprowadzony przez ciebie tekst jest opowiedni kliknij: Zatwierdź")
        if st.button("Zatwierdź"):
            st.success("Zatwierdzono!")

    # Opcja "Uzupełnij formularz"
    elif tab_select == FORM:
        st.markdown("### Wypełnij formularz podróży 📝")
        st.info('''Pamiętaj że możesz pominąć niektóre pytania w formularzu, lecz wtedy
                    jest mnijesza szansa na znalezienie idealnego lotu dla ciebie''' )

        # Data lotu
        date_of_flight = st.date_input("📅 Wybierz datę wylotu")

        # Długość wyjazdu
        time_of_vacation = st.select_slider(
            "⏳ Wybierz liczbę dni podróży",
            options=list(range(1,14)),  # Zakres 1-50 dni
            value=(5, 7)  # Domyślnie 5-14 dni
        )

        # Wybór kraju i kontynentu
        country = st.selectbox("🌍 Wybierz kraj docelowy", ["Polska", "Hiszpania", "Włochy", "USA", "Japonia"])
        continent = st.selectbox("🗺️ Wybierz kontynent", ["Europa", "Azja", "Ameryka Północna", "Afryka", "Australia"])

        # Koszt biletu
        cost = st.slider(
            "💰 Wybierz maksymalną cenę biletu (za dwa loty łącznie)",
            min_value=0, max_value=1200, value=(500, 700)
        )

        # Godziny wylotów
        departure_from_home_time = st.time_input("🛫 Wybierz godzinę wylotu z Polski")
        departure_from_abroad_time = st.time_input("🛬 Wybierz godzinę lotu powrotnego")

        st.success("Formularz wypełniony!")

