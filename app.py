import streamlit as st
from audiorecorder import audiorecorder
from io import BytesIO
from dotenv import dotenv_values
from openai import OpenAI
from qdrant_client import QdrantClient
import re
import pandas as pd

# Modele AI
TEXT_TO_EMBEDDINGS = "text-to-embeddings-3-large"
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
TEXT_TO_TEXT = "gpt-4o"

# Zakładki
SAY_OR_WRITE = "Napisz lub powiedz coś o swoich preferencjach"
FORM = "Uzupełnij formularz, gdzie chciałbyś polecieć"

env=dotenv_values(".env")
# Funkcja od klucza OPEN_AI
def get_openai_client():
   return OpenAI(api_key=env["OPENAI_API_KEY"])

# Funkcje do komuniakcji z QDRANT
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"])

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

# FUNKCJE DO WYSZUKIWANIA LOTÓW
## FUNKCJE OPENAI

### FUNCKJA OPENAI TEXT TO Embedding
def user_text_to_embedings(text):
    openai_client=get_openai_client()
    response=openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=1536
    )
    return response.data[0].embedding
### FUNCKJA OPENAI TEXT TO TEXT
def user_text_to_text(prompt):
    openai_client=get_openai_client()
    response=openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
        {"role": "system", "content": "Jesteś asystentem pomagającym znaleźć idealny lot."},
        {"role": "user", "content": f'''
        Twoim zadaniem jest z tekstu, który poda użytkownik o tym, jaki lot chciałby znaleźć, 
        string z dopasowanymi poszczególnymi preferencjami użytkownika. 
        Weź pod uwagę temperaturę w danym miejscu, preferowany budżet, kontynent, konkretne miasto,
        ilość dni jaką chciałby spędzić, data wylotu z kraju itp.
         
        Tekst użytkownika: "{prompt}"
        
        Podpowiedzi, a propo co możesz dostać w prompcie i jak dopasować do jsona:

        <temperature_of_destination> Do tego co ci poda użytkownik wybierz z pośród dopasowując odpowiednio do danej destynacji: zimno,ciepło, bardzo ciepło, umiarkowanie,
        i do kazdej z kategori klimatu dobierz lub jeżeli poda tylko klimat lub temperaturę również odpowiednio dobierz:
        zimno:-10 - 10 ℃,
        umiarkowanie: 11- 20 ℃,
        ciepło: 20-30 ℃	,
        bardzo ciepło: 30 - 40℃
         1. Jeśli użytkownik poda nazwę miasta, sprawdź, do której kategorii należy:
         - "ciepłe_miejsca" → przypisz klimat ciepło: 20-30 ℃	,
         - "bardzo_cieple_miejsca" → przypisz klimat bardzo ciepło: 30 - 40℃.
         - "umiarkowane_miejsca" → przypisz klimat umiarkowanie: 11- 20 ℃.
         - "zimne_miejsca" → przypisz klimat zimno:-10 - 10 ℃.

         2. Jeśli użytkownik poda kraj, określ jego przynależność regionalną:
            - Morze Śródziemne (Hiszpania, Włochy, Grecja, Portugalia, Turcja) → ciepło (20-30℃).
            - Skandynawia (Norwegia, Szwecja, Finlandia, Dania, Islandia) → zimno (-10 - 10℃).
            - Europa Środkowa i Wschodnia (Polska, Niemcy, Czechy, Wielka Brytania, Ukraina) → umiarkowanie (11-20℃).
            - Afryka:
            - Północna Afryka (Maroko, Tunezja, Egipt) → ciepło (20-30℃).
            - Tropikalna Afryka (Zanzibar, Lagos, Dakar, Dar es Salaam) → bardzo ciepło (30-40℃).
            - Azja Północna (Rosja – Syberia, Mongolia) → zimno (-10 do 10℃).
            - Azja Środkowa (Kazachstan, Uzbekistan, Kirgistan) → umiarkowanie (11-20℃).
            - Azja Południowa (Indie, Pakistan, Sri Lanka, Bangladesz) → bardzo ciepło (30-40℃).
            - Azja Południowo-Wschodnia (Tajlandia, Wietnam, Indonezja, Filipiny) → bardzo ciepło (30-40℃).
            - Azja Wschodnia (Chiny, Japonia, Korea Południowa) → umiarkowanie (11-20℃).
            - Bliski Wschód (Arabia Saudyjska, Zjednoczone Emiraty Arabskie, Iran) → bardzo ciepło (30-40℃).

         3. Jeśli użytkownik poda tylko kraj, ale nie miasto, wybierz kilka pasujących miast z tego kraju.

         4. Jeśli miasto lub kraj nie znajduje się na liście, sprawdź jego szerokość geograficzną i pobliskie kraje, aby określić klimat.

        

         Jeszcze jeżli użytkownik powie ci że chce lecieć do któregoś ze Skandynawskich krajów np.Dania, Finlandia -> zimno:-10 - 10 ℃

        <cost_of_both_flights>:
         - Jeśli użytkownik poda konkretną kwotę,wyższą niż 300, zwróć ją doklładnie, lecz jeśli niższą zwróć 300  
         - Jeśli opisuje budżet ogólnie lub opisowo, przypisz go do jednej z kategorii:  
         - 300 zł → (gdy  UZYTKOWNIK powie "nie mam kasy"," mam mały budżet", "jestem spłukany" itp)
         - 700 zł → (gdy UZYTKOWNIK powie "mam średnia budżet" lub "mam nie za duży budżet" itp)
         - 1300 zł → (gdy UZYTKOWNIK powie "ma bardzo dużo pieniędzy czy ma bardzo duzy budżet" itp)

        
        4. **Jeśli użytkownik poda porę roku, przypisz dokładną porę roku oraz oszacuj liczbę dni do wylotu (zakładając dzisiejszą datę jako 5 marca 2025):**
            - Wiosna (marzec, kwiecień, maj).np. Lot w maju
            - Lato (czerwiec, lipiec, sierpień). np.
            - Jesień (wrzesień, październik, listopad).
            - Zima (grudzień, styczeń, luty).
            - Jeśli użytkownik powie „chcę lecieć na święta” → przypisz porę roku „zima”.

         5. **Jeśli użytkownik poda miesiąc, określ jego porę roku oraz liczbę dni do wylotu** (zakładając, że dziś jest 5 marca 2025):
            - Marzec → Wiosna, lot za mniej niż 14 dni (jeśli data jest bliska).
            - Kwiecień → Wiosna, lot za około miesiąc.
            - Maj → Wiosna, lot za więcej niż miesiąc.
            - Czerwiec → Lato, lot za około pół roku.
            - Lipiec → Lato, lot za około pół roku.
            - Sierpień → Lato, lot za więcej niż pół roku.
            - Wrzesień → Jesień, lot za więcej niż pół roku.
            - Październik → Jesień, lot za więcej niż pół roku.
            - Listopad → Jesień, lot za więcej niż pół roku.
            - Grudzień → Zima, lot za więcej niż pół roku.
            - Styczeń → Zima, lot za więcej niż pół roku.
            - Luty → Zima, lot za więcej niż pół roku.

         6. **Jeśli użytkownik poda, że chce lot za konkretną liczbę dni, określ kategorię wylotu (przyjmując datę 5 marca 2025 jako punkt startowy):**
            - Mniej niż 14 dni → wylot przed 19 marca.
            - Około miesiąc → wylot około 5 kwietnia.
            - Więcej niż miesiąc → wylot między 6 kwietnia a 5 września.
            - Około pół roku → wylot około września 2025.
            - Więcej niż pół roku → wylot po wrześniu 2025.

        <days_of_vacation>: Jeżeli użytkownik poda konkretnie to podaj konkretnie, zawsze pisz liczbowo nie oposiowo czyli np."3 dni" lub "6 dni "8 dni " itp.
        - Jeżeli użytkowinik nie powie konkretnie MUSISZ być wyczulony na zwroty jak:"city break", "mało dni", "kilka dni","parę dni","na krótko" to wtedy piszesz 2 dni
        - Pisz tylko wartości liczbowe czyli jak Uzytkownik poda np. 1-3 dni to napisz -> 2 lub jak poda 6-9 -> napisz 8 czyli pisz średnią, ale zaokrąglona
        - Jeżeli użytkownik nie powie nic o ilości dni wakacji zwracaj domyśłnie 7
        
        <wchich_part_of_day_departure_from_poland>: Jeżeli użytkownik poda ci po połundiu to napisz "po południu", natomiast jak powie, że lubi wracać w nocy 
        pasowałoby "wieczorem", natomiast jeśli powie, że chce wracać rano podaj "rano", jak powie że nie lubi w nocy lub nie lubi rano to napisz po południu.
       
        <wchich_part_of_day_departure_from_abroad>: Jeżeli użytkownik poda po południu to napisz "po południu", natomiast jak powie, że lubi wracać w nocy 
        pasowałoby "wieczorem", natomiast jeśli powie, że chce wracać rano podaj "rano", jak powie że nie lubi w nocy lub nie lubi rano to napisz po południu.

        
        <destination>: 
        -Jeżeli użytkownik powie dokładnie gdzie chce polecieć to napisz to konkretnie musisz napisać.
        -Powinienieś także wykrywać zabytki jakie chce odwiedzić, czy np. mecz jaki chciałby zobaczyć wtedy z tego co powie musisz dać dokładnie destynację miasta gdzie to się znajduje,
        -Bardzo ważne jest również to, że jak poda że był w jakimś miejscu i chce do podobnego miejsca lub że chce coś w tym stylu i podobne wyrażenia 
        np. Był Barcelonie i chce coś podobnego to nie pisz dokładnie tego miejsca tylko napisz np." Miasto w Europie z Morzem Śródziemnym np. Kreta,",
        albo poda np. podobny do Tokio to napisz "Rozwinięty kraj dalekiego wschodu", z kolei jak napisz np. Oslo napisz kraj Skandynawski. Jednym słowem jeżeli nie jest zdecydowany pisz opisowo pasujące do 
        danego miejsca sformułowanie
        - Jeżeli napisze Stany Zjednoczone pisz USA,
        - Jeżeli napisze miasto albo Europa zachodnia to napisz Europa zachodnia, a jak napisze Bliski wschód napisz Bliski wschód itp.
        - Także jak poda np. ,że lubi Konkretne Morze to napisz basen, któregoś morza, np. "Basen Morza Bałtyckeigo","Basen Morza Czarnego" itp.
        - Jeżeli napisze tylko kontynenty np. Europa-> Europa, Azja, Afryka-> Afryka, chyba że użytkownik poda dokładnie destynację wtedy nie
        - Jeżeli użytkownik powie ci że nie chce do jakiegoś kontynenty np. "Nie chce do Europejskiego miasta" lub "poza Europą"-> napisz :Azja, Afryka, Ameryka Północna
        - jeżeli stwierdzisz, że powiedział o bardzo bardzo ciepłym klimacie -> wybierz z "Egipt, Zanzibar, Dubaj itp" 
        - Jeżeli nie powie destynacji, a powie ciepłe miejsce itp. -> wybierz z "Hiszpania, Włochy itp."
        - Jeżeli nie powie destynacji, a powie zimne miejsce itp. -> wybierz z "Tallin, Skandywnawia, Kopenhaga itp."
        - jeżeli stwierdzisz, że powiedział o umiarkowanym klimacie -> wybierz z "Paryż, Londyn, Dublin" 
        

        <continent_of_destination>: Tutaj, jak użytkownik poda ci destynację to podaj kontynent na których się znajduje, lub jak
        poda konkretny kontynent to napisz ten kontynent lub kontynenty,
        - Jeżeli nie będziesz umiał określić kontynentu nie pisz nic 

        Pamiętaj do zwracanego stringa zawieraj tylko składowe, które poda ci użytkownik i zawsze dni wakacji nawet jak nie poda
        Odpowiedz w podaj w formacie String po polsku np.: 
           
         - Jeżli użytkownik poda wszytsko:
         "Klimat: ciepło , Temperatura: 20-30 ℃	, 
         Dokładny budżet na loty: 500 zł,
         Wylot z Polski: mniej niż 14 dni, Kategoria wylotu: bardzo mało, Miesiąc wylotu z Polski: marzec, Pora roku wylotu z Polski:wiosna "
         Czas trwania wakacji: 10 dni, 
         Kategoria czasu wylotu z Polski: rano, 
         Kategoria godziny wylotu z zagranicy: po południu, 
         Destynacja: Rzym, Kontynent destynacji: Europa."
          
        - Jeżli poda np. ilość dni wakacji, klimat i budżet:  
         "Klimat: bardzo ciepło , Temperatura: 30-40 ℃	, 
         Dokładny budżet na loty: 350 zł, 
         Czas trwania wakacji: 12 dni,
         Destynacja: USA, Kontynent destynacji: Ameryka Północna." " 
            
         - Jeżli poda np. ilość o której chce mieć wylot powrotny i destynacje ,budżet:
         
         "Dokładny budżet na loty: 150 zł,
         Czas trwania wakacji: 7 dni
         Kategoria godziny wylotu: wieczorem,
         Destynacja: Skandynawia, Kontynent destynacji: Europa.

      
         - Jeżli poda np. budżet niedokładną ilość dni wakacji, miesiąc kiedy chce lecieć i w jakiej porze roku: 
         "Dokładny budżet na loty: 700 zł, 
         Pora roku wylotu z Polski:lato,  Miesiąc wylotu z Polski: lipiec,  
         Czas trwania wakacji: 2 dni, 
         Kategoria godziny wylotu: wieczorem, "
           
        '''}
    ]
    )
    return response.choices[0].message.content 


## FUNKCE DO ZMIANY z TEXT TO TEXT NA DICT
def convert_string_user_to_dict(s):
    # Tworzymy słownik na podstawie wzorców
    dict_user_conv = {}

    # Wyszukiwanie wartości w stringu
    match_klimat = re.search(r"Klimat:\s*([^,]+)", s)
    match_temperatura = re.search(r"Temperatura:\s*(\d+\s*-\s*\d+\s*℃)", s)
    match_budzet = re.search(r"Dokładny budżet na loty:\s*(\d+)", s)
    match_kat_budzet = re.search(r"Budżet \(kategoria\):\s*([\w\s]+)", s)
    match_wylot = re.search(r"Wylot z Polski:\s*([\w\s]+)", s)
    match_kat_wylotu = re.search(r"Kategoria wylotu:\s*([\w\s]+)", s)
    match_miesiac_wylotu = re.search(r"Miesiąc wylotu z Polski:\s*([\w]+)", s)
    match_pora_roku = re.search(r"Pora roku wylotu z Polski:\s*([\w]+)", s)
    match_czas_wakacji = re.search(r"Czas trwania wakacji:\s*(\d+)", s)
    match_kat_czasu_wylotu = re.search(r"Kategoria czasu wylotu z Polski:\s*([\w\s]+)", s)
    match_destynacja = re.search(r"Destynacja:\s*([\wÀ-ÿĄąĆćĘęŁłŃńÓóŚśŹźŻż\s-]+)", s)
    match_kat_godz_wylotu = re.search(r"Kategoria godziny wylotu z zagranicy:\s*([\w\s]+)", s)
    match_kontynent = re.search(r"Kontynent destynacji:\s*([\wÀ-ÿĄąĆćĘęŁłŃńÓóŚśŹźŻż\s-]+)", s)


   

    # Dodawanie wartości do słownika
    if match_klimat:
        dict_user_conv["climate"] = match_klimat.group(1)

    if match_temperatura:
        dict_user_conv["temperature"] = match_temperatura.group(1)

    if match_budzet:
        dict_user_conv["budget"] = int(match_budzet.group(1))

    if match_kat_budzet:
        dict_user_conv["budget_kategory"] = match_kat_budzet.group(1).strip()

    if match_wylot:
        dict_user_conv["when_flight"] = match_wylot.group(1).strip()

    if match_kat_wylotu:
        dict_user_conv["category_of_flight"] = match_kat_wylotu.group(1).strip()

    if match_miesiac_wylotu:
        dict_user_conv["month_of_flight"] = match_miesiac_wylotu.group(1)

    if match_pora_roku:
        dict_user_conv["season_of_flight"] = match_pora_roku.group(1)

    if match_czas_wakacji:
        dict_user_conv["days_of_vacation"] = int(match_czas_wakacji.group(1))

    if match_kat_czasu_wylotu:
        dict_user_conv["part_of_day_dep_poland"] = match_kat_czasu_wylotu.group(1).strip()

    if match_kat_godz_wylotu:
        dict_user_conv["part_of_day_dep_abroad"] = match_kat_godz_wylotu.group(1).strip()

    if match_destynacja:
        dict_user_conv["destination"] = match_destynacja.group(1)

    if match_kontynent:
        temp_cont = match_kontynent.group(1)
        if temp_cont=='Ameryka':
            dict_user_conv["continent"]="Ameryka Północna"
        else:
            dict_user_conv["continent"]=temp_cont

    # ✅ Wyświetlenie słownika
    return dict_user_conv

## FUNKCE DO ZMIANY z TEXT TO TEXT NA String do embeddingów
def embedding_to_compare(dict_compare):
    if ("destination" in dict_compare) | ("continent" in dict_compare):
        if ("destination" not in dict_compare):
            destination="-"
        else:
            destination=dict_compare["destination"]
        if ("continent" not in dict_compare):
            continent="-"
        else:
            continent=dict_compare["continent"]
        text_to_comp_embedding=""
        text_to_comp_embedding=f"Destynacja: {destination}, Kontynent destynacji: {continent}"
        return text_to_comp_embedding
    else: 
        return 0

## FUNKCJE QDRANT i DANE DO BAZY QDRANT
QDRANT_COLLECTION_NAME="loty_test_2"
EMBEDDING_DIM=1536
def flights_from_db(query):
    qdrant_client=get_qdrant_client()
    flights = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=user_text_to_embedings(query),
            limit=500,
        )
    result = []
    for flight in flights:
        result.append({
            "destination": flight.payload["destination"], 
            "continent_of_destination":flight.payload["continent_of_destination"],
            "classify_continent_number":flight.payload["classify_continent_number"],
            "temperature_of_destination":flight.payload["temperature_of_destination"],
            "classify_climat_number":flight.payload["classify_climat_number"],
            "cost_of_flight_from_poland":flight.payload["cost_of_flight_from_poland"],
            "cost_of_flight_to_poland":flight.payload["cost_of_flight_to_poland"],
            "cost_of_both_flights":flight.payload["cost_of_both_flights"], 
            "amount_of_days_vacation":int(flight.payload["amount_of_days_vacation"]), 
            "date_of_departure":flight.payload["date_of_departure"],
            "hour_of_departure_from_poland":flight.payload["hour_of_departure_from_poland"],
            "categorized_dep_time_poland":flight.payload["categorized_dep_time_poland"],
            "categorized_dep_time_poland_number":flight.payload["categorized_dep_time_poland_number"],
            "hour_of_departure_from_abroad":flight.payload["hour_of_departure_from_abroad"],
            "date_of_arrival":flight.payload["date_of_arrival"],
            "categorized_dep_time_abroad":flight.payload["categorized_dep_time_abroad"], 
            "categorized_dep_time_abroad_number":flight.payload["categorized_dep_time_abroad_number"], 
            "when_flight":flight.payload["time_of_flight_and_arrival"],
            "when_flight_number":flight.payload["amount_of_days_when_flight_number"],
            "season_of_flight":flight.payload["season_of_departure"],
            "season_of_flight_number":flight.payload["season_of_departure_number"],
            "month_of_flight":flight.payload["month_of_flight"],
            "month_of_flight_number":flight.payload["month_of_flight_number"],
            "score": flight.score,
        })
        
    result_df=pd.DataFrame(result)
    return result_df
## Funkcje do podobieństwa preferencji użytkownika do lotów

### Funkcja do podobieństwa kontynentu
def classify_continent_number(time_of_flight_and_arrival):
    if time_of_flight_and_arrival == "Europa":
        return 0
    elif time_of_flight_and_arrival =="Afryka":
        return 5
    elif time_of_flight_and_arrival == "Azja":
        return 10
    elif time_of_flight_and_arrival == 'Ameryka Północna':
        return 15
    else:
        return 20
    
### Funkcja do podobieństwa klimatu
def classify_climat_number(time_of_flight_and_arrival):
    if time_of_flight_and_arrival == "zimno":
        return 0
    elif time_of_flight_and_arrival =="umiarkowanie":
        return 2
    elif time_of_flight_and_arrival == "ciepło":
        return 6
    elif time_of_flight_and_arrival == "bardzo ciepło":
        return 8
    
### Funkcja do podobieństwa pory roku
def get_season_number(month):
    if month == "wiosna":
        return 0
    elif month == "lato" :
        return 4
    elif month == "zima":
        return 13
    else:
        return 10

### Funckja do podobieństwa pory dnia wylotu
def categorize_time_number(hour):
    
    if hour=="rano":
        return 0
    elif hour == "po południu":
        return 3
    elif hour== "wieczorem":
        return 6
    else:
        return 9

### Funkcja do podobieństwa miesiąca
def get_month_specfyic_number(month_name):
    month_values = {
        "styczeń": 0,
        "luty": 2,
        "marzec": 4,
        "kwiecień": 6,
        "maj": 8,
        "czerwiec": 10,
        "lipiec": 12,
        "sierpień": 14,
        "wrzesień": 16,
        "październik": 18,
        "listopad": 20,
        "grudzień": 22
    }

    return month_values.get(month_name.lower(), "Niepoprawna nazwa miesiąca")  # Obsługa błędu

### klasyfikacja jak długo do lotu
def classify_days_number(time_of_flight_and_arrival):
    if time_of_flight_and_arrival == "mniej niż 14 dni":
        return 0
    elif time_of_flight_and_arrival =="około miesiąc":
        return 3
    elif time_of_flight_and_arrival == "około 3 miesiące":
        return 6
    elif time_of_flight_and_arrival == "około pół roku":
        return 8
    else:
        return 10

### Funkcja do zliczania wszystkich prefernecji
def filter_dataframe_from_db(df_future, dict_data):
    df_future = df_future.copy()  # Kopia DataFrame, aby nie modyfikować oryginału

    # Filtrowanie po budżecie
    if "budget" in dict_data:
        df_future = df_future[df_future["cost_of_both_flights"] <= (dict_data["budget"] + 100)]
    
    if df_future.empty:  # Jeśli po filtracji jest pusty, zwróć od razu
        return df_future  

    df_future["preference_score"] = 0  # Inicjalizacja kolumny po filtracji

    # Lista warunków do obliczenia wartości preferencji
    conditions = []

    if "climate" in dict_data:
        dict_data["classify_climat_number"] = classify_climat_number(dict_data["climate"])
        conditions.append(abs(df_future["classify_climat_number"] - dict_data["classify_climat_number"]))

    if "continent" in dict_data:
        if dict_data["continent"] in ["Europa", "Azja", "Afryka", "Ameryka Północna", "Australia", "Ameryka Południowa"]:
            dict_data["classify_continent_number"] = classify_continent_number(dict_data["continent"])
            conditions.append(abs(df_future["classify_continent_number"] - dict_data["classify_continent_number"]))

    if "days_of_vacation" in dict_data:
        conditions.append(abs(df_future["amount_of_days_vacation"] - dict_data["days_of_vacation"]))

    if "part_of_day_dep_poland" in dict_data:
        dict_data["part_of_day_dep_poland_number"] = categorize_time_number(dict_data["part_of_day_dep_poland"])
        conditions.append(abs(df_future["categorized_dep_time_poland_number"] - dict_data["part_of_day_dep_poland_number"]) / 2)

    if "part_of_day_dep_abroad" in dict_data:
        dict_data["part_of_day_dep_abroad_number"] = categorize_time_number(dict_data["part_of_day_dep_abroad"])
        conditions.append(abs(df_future["categorized_dep_time_abroad_number"] - dict_data["part_of_day_dep_abroad_number"]) / 2)

    if "season_of_flight" in dict_data:
        dict_data["season_of_flight_number"] = get_season_number(dict_data["season_of_flight"])
        conditions.append(abs(df_future["season_of_flight_number"] - dict_data["season_of_flight_number"]))

    if "month_of_flight" in dict_data:
        dict_data["month_of_flight_number"] = get_month_specfyic_number(dict_data["month_of_flight"])
        conditions.append(abs(df_future["month_of_flight_number"] - dict_data["month_of_flight_number"]))

    if "when_flight" in dict_data:
        dict_data["when_flight_number"] = classify_days_number(dict_data["when_flight"])
        conditions.append(abs(df_future["when_flight_number"] - dict_data["when_flight_number"]))

    # Sumujemy wartości bezwzględne i zapisujemy do nowej kolumny 'preference_score'
    if conditions:
        df_future["preference_score"] = sum(conditions)  # Sumowanie wartości dla każdego wiersza
        df_future = df_future.sort_values("preference_score")
    return df_future

    

### Funkcja do dawania lotu możliwie najpeszego pod destynację
def check_scores(df):
    top_10 = df.sort_values("score", ascending=False).head(70)
    min_preference = top_10["preference_score"].min()
    selected_rows = []

    for i in range(len(top_10)):
        for j in range(len(top_10)):
            if i != j and (
                top_10.iloc[i]["score"] >= top_10.iloc[j]["score"] + 0.04 and
                top_10.iloc[i]["preference_score"] <= min_preference + 10
            ):
                selected_rows.append(top_10.index[i])  # Użyj indeksu zamiast i

    result = top_10.loc[list(set(selected_rows))] if selected_rows else pd.DataFrame()

    if not result.empty:
        result = result.sort_values("preference_score")
    return result

#
## st.session_state
# 

# Ustawienie ekranu startowego przy pierwszym uruchomieniu
if "page" not in st.session_state:
    st.session_state["page"] = "start"
# Ustawianie sesion state tekstu jaki podaje użytkownik
if "user_text" not in st.session_state:
    st.session_state["user_text"]=""
# Ustawianie sesion state z audio
if "audio" not in st.session_state:
    st.session_state["audio"]=None
# Ustawianie sesion state textu z audio
if "text_from_audio" not in st.session_state:
    st.session_state["text_from_audio"]=""

# Ustawianie session state na loty najlepsze pd destynację
if "best_flight_dest" not in st.session_state:
    st.session_state["best_flight_dest"]={}

# Ustawianie session state na loty najlepsze pod preferencje
if "best_flights" not in st.session_state:
    st.session_state["best_flights"]={}

# Ustawainie session state na najlepszy lot
if "best_flight" not in st.session_state:
    st.session_state["best_flight"] = {}  # Inicjalizacja pustego słownika, jeśli go nie ma

# Ustawainie session state na najlepszy lot
if "second_best_flight" not in st.session_state:
    st.session_state["second_best_flight"]={}

# Ustawianie session state do ładowania ekranu
if "loading" not in st.session_state:
    st.session_state["loading"] = False



# Ustawianie session state dla opcji klimatu
if "climat_list" not in st.session_state:
    st.session_state["climat_list"]=["❄️zimno", "🌤️umiarkowanie","☀️ciepło","🔥bardzo ciepło"]

if "continent_list" not in st.session_state:
    st.session_state["continent_list"]=["Europa", "Azja", "Ameryka Północna", "Afryka"]
if "destination_list" not in st.session_state:
    st.session_state["destination_list"]=["Polska", "Hiszpania", "Włochy", "USA", "Japonia"]
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
    # tab_select = st.selectbox(
    #     "Wybierz opcję, jak chcesz znaleźć idealny lot",
    #     [SAY_OR_WRITE, FORM],
    #     index=0  # Ustawienie domyślnej wartości
    # )

    # Opcja "Napisz lub powiedz coś o swoich preferencjach"
    #if tab_select == SAY_OR_WRITE:
    st.markdown("### Powiedz lub napisz, gdzie chcesz polecieć ✈️")
    
    st.info("🎤 **Jak podać cel podróży?** Kliknij **Powiedz**, następnie mów, a gdy zakończysz, naciśnij **Zakończ**.")

    user_audio = audiorecorder(
        start_prompt="🎤 Powiedz",
        stop_prompt="⏹ Zakończ"
    )

    if user_audio:
        st.session_state["audio"]=get_audio(user_audio)  
        st.session_state["text_from_audio"]=transcribe_audio(st.session_state["audio"])
        st.session_state["user_text"]=st.session_state["text_from_audio"]

    st.info("⌨️ Możesz również napisać w **polu pod spodem** o swoich preferencjach co do lotu.")
    st.text_area(
        label="",
        label_visibility="hidden",
        placeholder=(
            "Powiedz, gdzie lubisz podróżować, jaki klimat Cię interesuje, "
            "na ile dni chciałbyś wylecieć, o której chciałbyś mieć wylot w tamtą i powrotną stronę..."
        ),
        key="user_text",  # Dodajemy `key`, by Streamlit automatycznie śledził zmiany
    )

    st.info("✅ **Sprawdź swój tekst!** Jeśli wszystko się zgadza, kliknij **Zatwierdź**.")
    if st.button("Zatwierdź"):
        st.success("Zatwierdzono!")
        st.session_state["page"]="flights_for_user"
        st.rerun()

elif st.session_state["page"]== "flights_for_user":
    with st.spinner("🔍 Szukam najlepszych lotów dla Ciebie... ✈️"):
        st.markdown(
        """
        ## Oto idealne loty dla Ciebie! 🌍✈️  
        """
        )
        text_from_user_to_model=st.session_state["user_text"]
        if st.button("Powrót"):
            st.session_state["page"]="main"
            st.rerun()
        
        if "user_text" in st.session_state and st.session_state["user_text"]:
            #text="Klimat: ciepło, Temperatura: 20-30 ℃, Czas trwania wakacji: 10 dni, Destynacja: Australia, Kontynent destynacji: Australia."
            text=user_text_to_text(text_from_user_to_model)
            if text=="0":
                st.error("Podaj informacje jeszcze raz")
                st.stop()
            else:
                try:
                    dcit_of_pref=convert_string_user_to_dict(text)
                    string_to_embeddings=embedding_to_compare(dcit_of_pref)
                    flights_from_DB=flights_from_db(string_to_embeddings)
                    best_flights=st.session_state["best_flights"]
                    flights_with_filter=filter_dataframe_from_db(flights_from_DB, dcit_of_pref)
                    best_flights=check_scores(flights_with_filter)
                    best_flights_len=len(best_flights)
                except Exception as e:
                    st.error(f"Wystąpił błąd kurwaa: {e}")
                    st.stop()
                best_flight = st.session_state["best_flight"]
                if best_flights_len >= 2:
                    try:
                        best_flight = best_flights.iloc[0]
                        second_best_flight_df = best_flights.iloc[1]
                        third_best_flight=flights_with_filter.iloc[0]
                        fourth_best_flight=flights_with_filter.iloc[1]
                    except Exception as e:
                        st.st.error(f"Wystąpił nieoczekiwany błąd w >=2: {e}")
                        st.stop()
                    
                elif best_flights_len == 1 :
                    try:
                        best_flight = best_flights.iloc[0]
                        second_best_flight_df=flights_with_filter.iloc[0]
                        third_best_flight=flights_with_filter.iloc[1]
                        fourth_best_flight=flights_with_filter.iloc[2]
                    except Exception as e:
                        st.st.error(f"Wystąpił nieoczekiwany błąd w ==1: {e}")
                        st.stop()
                else:
                    try:
                        best_flight=flights_with_filter.iloc[0]
                        second_best_flight_df=flights_with_filter.iloc[1]
                        third_best_flight=flights_with_filter.iloc[2]
                        fourth_best_flight=flights_with_filter.iloc[3]
                    except Exception as e:
                        st.st.error(f"Wystąpił nieoczekiwany błąd w else: {e}")
                        st.stop()  

                # Pierwszy lot (widoczny)
                departure_date_1 = best_flight["date_of_departure"] if "date_of_departure" in best_flight else "Brak danych"
                departure_time_1 = best_flight["hour_of_departure_from_abroad"] if "hour_of_departure_from_abroad" in best_flight else "Brak danych"
                departure_airport_1 = "🛫 Warszawa (WAW)"
                destination_airport_1 = f"🛫 {best_flight['destination']}" if "destination" in best_flight else "🛫 Brak danych"

                return_date_1 = best_flight["date_of_arrival"] if "date_of_arrival" in best_flight else "Brak danych"
                return_time_1 = best_flight["hour_of_departure_from_abroad"] if "hour_of_departure_from_abroad" in best_flight else "Brak danych"
                return_airport_1 = f"🛫 {best_flight['destination']}" if "destination" in best_flight else "🛫 Brak danych"
                home_airport_1 = "🏠 Warszawa (WAW)"

                def flight_box(departure_date, departure_time, departure_airport, destination_airport,
                    return_date, return_time, return_airport, home_airport, price_1, price_2, total_cost):
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #A0C4FF, #BDB2FF);
                            padding: 15px; 
                            border-radius: 10px; 
                            color: black; 
                            text-align: center;
                            margin-bottom: 10px;
                            box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);">
                            <h3>📅 {departure_date} | ⏰ {departure_time}</h3>
                            <p style="font-size: 18px; font-weight: bold;">{departure_airport} ➝ {destination_airport}</p>
                            <p style="font-size: 20px; font-weight: bold;">💰 {price_1}</p>
                            <hr style="border: 0.5px solid black;">      
                            <h3>🔄 {return_date} | ⏳ {return_time}</h3>
                            <p style="font-size: 18px; font-weight: bold;">{return_airport} ➝ {home_airport}</p>
                            <p style="font-size: 20px; font-weight: bold;">💰 {price_2}</p>  
                            <hr style="border: 0.5px solid black;">      
                            <h3>💰 Łączna cena: {total_cost}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True

                        
                    )
                    st.markdown(
                        """

                        """
                        )

                # Definiowanie zmiennych dla czterech najlepszych lotów
                flights = [
                    (best_flight, "🛫 Warszawa (WAW)", "🏠 Warszawa (WAW)"),
                    (second_best_flight_df, "🛫 Kraków (KRK)", "🏠 Kraków (KRK)"),
                    (third_best_flight, "🛫 Gdańsk (GDN)", "🏠 Gdańsk (GDN)"),
                    (fourth_best_flight, "🛫 Wrocław (WRO)", "🏠 Wrocław (WRO)")
                ]

                for flight, dep_airport, home_airport in flights:
                    departure_date = flight.get("date_of_departure", "Brak danych")
                    departure_time = flight.get("hour_of_departure_from_abroad", "Brak danych")
                    destination_airport = f"🛫 {flight.get('destination', 'Brak danych')}"
                    return_date = flight.get("date_of_arrival", "Brak danych")
                    return_time = flight.get("hour_of_departure_from_abroad", "Brak danych")
                    return_airport = f"🛫 {flight.get('destination', 'Brak danych')}"
                    price_1 = f"{flight.get('cost_of_flight_from_poland', 'Brak danych')} zł"
                    price_2 = f"{flight.get('cost_of_flight_to_poland', 'Brak danych')} zł"
                    total_cost = f"{flight.get('cost_of_both_flights', 'Brak danych')} zł"
                    
                    flight_box(departure_date, departure_time, dep_airport, destination_airport,
                            return_date, return_time, return_airport, home_airport, price_1, price_2, total_cost)
        # else:
        #     st.warning("Proszę wpisać lub nagrać tekst przed zatwierdzeniem.")
    
    

    

