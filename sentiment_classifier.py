# -*- coding: utf-8 -*-
import numpy as np
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

print("\n--- SZCZEGÓŁOWA ANALIZA METRYK ---\n")
print("Macierz pomyłek (Confusion Matrix):")
print(f"                  | Predykcja: Negatywna | Predykcja: Pozytywna |")
print(f"------------------|----------------------|----------------------|")
print(f"Rzeczywista: Neg. | TN = {tn:<15} | FP = {fp:<15} |")
print(f"Rzeczywista: Poz. | FN = {fn:<15} | TP = {tp:<15} |")
print("\nObjaśnienie terminów:")
print(f"  [TP] True Positive (Prawdziwie Pozytywne): {tp} - Poprawnie zidentyfikowane pozytywne opinie.")
print(f"  [TN] True Negative (Prawdziwie Negatywne): {tn} - Poprawnie zidentyfikowane negatywne opinie.")
print(f"  [FP] False Positive (Fałszywie Pozytywne): {fp} - Błąd typu I. Negatywne opinie błędnie oznaczone jako pozytywne.")
print(f"  [FN] False Negative (Fałszywie Negatywne): {fn} - Błąd typu II. Pozytywne opinie błędnie oznaczone jako negatywne.")
