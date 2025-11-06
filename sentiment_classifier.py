# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --- KROK 1: DANE WEJŚCIOWE I ZAKŁADANE ODPOWIEDZI ---

# Zestaw 20 obiektów (opinii) do klasyfikacji
X_train_text = [
    # Pozytywne (Klasa 1)
    "Ten kurs to prawdziwe złoto! Wiedza podana w pigułce.",
    "Świetny instruktor, wszystko tłumaczy w prosty i zrozumiały sposób.",
    "Bardzo polecam, materiał jest wartościowy i praktyczny.",
    "Najlepsza inwestycja w mój rozwój zawodowy.",
    "Jestem zachwycona poziomem tego szkolenia.",
    "Praktyczne przykłady i zadania domowe bardzo pomagają w nauce.",
    "Warto każdych pieniędzy. Wiedza na najwyższym poziomie.",
    "Doskonały materiał, z pewnością wrócę po więcej kursów.",
    "Struktura kursu jest logiczna i dobrze przemyślana.",
    "Nareszcie kurs, który spełnił wszystkie moje oczekiwania.",
    # Negatywne (Klasa 0)
    "Kompletna strata czasu, niczego nowego się nie nauczyłem.",
    "Materiał przedstawiony chaotycznie i bez ładu.",
    "Instruktor mówi monotonnym głosem, ciężko się skupić.",
    "Nie polecam, informacje są przestarzałe i nieaktualne.",
    "Pieniądze wyrzucone w błoto. Jestem bardzo zawiedziony.",
    "Zbyt dużo teorii, za mało praktycznych przykładów.",
    "Słaba jakość dźwięku i wideo utrudniała naukę.",
    "Spodziewałem się czegoś znacznie lepszego.",
    "Ten kurs to nieporozumienie, omijać szerokim łukiem.",
    "Kontakt z autorem kursu był praktycznie niemożliwy."
]

# Zakładane odpowiedzi (etykiety): 1 - opinia pozytywna, 0 - opinia negatywna
y_train = np.array([1]*10 + [0]*10)


# --- KROK 2: WEKTORYZACJA TEKSTU ---
# Przekształcenie tekstu na wektory liczbowe metodą TF-IDF

print(">>> Rozpoczynam wektoryzację tekstu metodą TF-IDF...")
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train_text)
print(f">>> Wektoryzacja zakończona. Powstała macierz o wymiarach: {X_train_vectors.shape}\n")


# --- KROK 3: TRENING MODELU KLASYFIKACYJNEGO ---
# Używamy Regresji Logistycznej do zadania klasyfikacji binarnej

print(">>> Rozpoczynam trening modelu (Regresja Logistyczna)...")
model = LogisticRegression()
model.fit(X_train_vectors, y_train)
print(">>> Trening modelu zakończony.\n")


# --- KROK 4: OCENA EFEKTYWNOŚCI MODELU ---
# Predykcja na danych treningowych i obliczenie metryk

print(">>> Dokonuję predykcji na danych treningowych...")
predictions = model.predict(X_train_vectors)

# 1. Dokładność (Accuracy)
accuracy = accuracy_score(y_train, predictions)
print("\n--- OKREŚLONY PROCENT POPRAWNYCH KLASYFIKACJI ---")
print(f"Dokładność (Accuracy): {accuracy * 100:.2f}%")
print("Oznacza to, że model poprawnie sklasyfikował", int(accuracy * len(y_train)), "z", len(y_train), "opinii.")

# 2. Macierz Pomyłek (Confusion Matrix)
#   [TN, FP]
#   [FN, TP]
cm = confusion_matrix(y_train, predictions)
tn, fp, fn, tp = cm.ravel()


# --- KROK 5: EKSPLORACJA DANYCH I OPCJONALNY EKSPORT WYNIKÓW ---

print("\n--- EKSPLORACJA WEKTORÓW I WAG MODELU ---")

# Задаем вопрос пользователю
export_choice = input("Czy chcesz wyeksportować wektory TF-IDF oraz wagi modelu do plików CSV? (t/n): ").lower()

if export_choice == 't':
    try:
        print("\n>>> Przygotowuję dane do eksportu...")

        # --- 1. Przygotowanie tabeli wektorów TF-IDF ---

        # Pobieramy nazwy wszystkich słów (cech) z naszego słownika
        feature_names = vectorizer.get_feature_names_out()

        # Konwertujemy rzadką macierz TF-IDF na gęstą tablicę NumPy
        dense_matrix = X_train_vectors.toarray()

        # Tworzymy DataFrame z biblioteki Pandas
        tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)

        # Dodajemy na początku kolumny z oryginalnymi danymi dla lepszej czytelności
        tfidf_df.insert(0, 'SENTYMENT_RZECZYWISTY', y_train)
        tfidf_df.insert(0, 'ORYGINALNA_OPINIA', X_train_text)

        # --- 2. Przygotowanie tabeli współczynników (wag) modelu ---

        # Pobieramy współczynniki nauczone przez model
        # model.coef_[0] zawiera wagi dla klasy "1" (pozytywnej)
        coefficients = model.coef_[0]

        # Tworzymy DataFrame, łącząc słowa z ich wagami
        coef_df = pd.DataFrame({'SŁOWO': feature_names, 'WAGA_WSPÓŁCZYNNIKA': coefficients})

        # Sortujemy, aby zobaczyć, które słowa mają największy wpływ
        coef_df = coef_df.sort_values(by='WAGA_WSPÓŁCZYNNIKA', ascending=False)

        # --- 3. Zapis do plików ---

        # Zapisujemy do plików CSV. Używamy kodowania 'utf-8-sig' aby polskie znaki
        # poprawnie otwierały się w programie Excel.
        tfidf_filename = 'tfidf_vectors.csv'
        coef_filename = 'word_coefficients.csv'

        tfidf_df.to_csv(tfidf_filename, index=False, encoding='utf-8-sig')
        coef_df.to_csv(coef_filename, index=False, encoding='utf-8-sig')

        print(f">>> Pomyślnie zapisano dane do plików:")
        print(f"  - Wektory TF-IDF: {tfidf_filename}")
        print(f"  - Wagi modelu: {coef_filename}")

        # WSKAZÓWKA: Aby zapisać do formatu Excel (.xlsx), odkomentuj poniższe linie
        # i upewnij się, że masz zainstalowaną bibliotekę 'openpyxl' (pip install openpyxl)
        # tfidf_df.to_excel('tfidf_vectors.xlsx', index=False)
        # coef_df.to_excel('word_coefficients.xlsx', index=False)

    except ImportError:
        print("\nBŁĄD: Biblioteka 'pandas' nie jest zainstalowana.")
        print("Proszę ją zainstalować komendą: pip install pandas")
    except Exception as e:
        print(f"\nWystąpił nieoczekiwany błąd podczas eksportu: {e}")

else:
    print("\n>>> Pominięto eksport danych.")

# --- KROK 6: DRUKOWANIE WYNIKÓW ---

HEADER_WIDTH = 18
CELL_WIDTH = 24

row_separator = f"{'-' * HEADER_WIDTH}-+-{'-' * CELL_WIDTH}-+-{'-' * CELL_WIDTH}-|"
header_line = f"{'':<{HEADER_WIDTH}} | {'Predykcja: Negatywna':^{CELL_WIDTH}} | {'Predykcja: Pozytywna':^{CELL_WIDTH}} |"

tn_cell = f"TN = {tn}"
fp_cell = f"FP = {fp}"
fn_cell = f"FN = {fn}"
tp_cell = f"TP = {tp}"

row_neg = f"{'Rzeczywista: Neg.':<{HEADER_WIDTH}} | {tn_cell:^{CELL_WIDTH}} | {fp_cell:^{CELL_WIDTH}} |"
row_pos = f"{'Rzeczywista: Poz.':<{HEADER_WIDTH}} | {fn_cell:^{CELL_WIDTH}} | {tp_cell:^{CELL_WIDTH}} |"

print("\n--- SZCZEGÓŁOWA ANALIZA METRYK ---")
print("Macierz pomyłek (Confusion Matrix):\n")
print(header_line)
print(row_separator)
print(row_neg)
print(row_pos)

print("\nObjaśnienie terminów:")
print(f"  [TP] True Positive (Prawdziwie Pozytywne): {tp} - Poprawnie zidentyfikowane pozytywne opinie.")
print(f"  [TN] True Negative (Prawdziwie Negatywne): {tn} - Poprawnie zidentyfikowane negatywne opinie.")
print(f"  [FP] False Positive (Fałszywie Pozytywne): {fp} - Błąd typu I. Negatywne opinie błędnie oznaczone jako pozytywne.")
print(f"  [FN] False Negative (Fałszywie Negatywne): {fn} - Błąd typu II. Pozytywne opinie błędnie oznaczone jako negatywne.")
