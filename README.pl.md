# Klasyfikator Tonalności Opinii w Języku Polskim

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Licencja](https://img.shields.io/badge/Licencja-MIT-green.svg)

Prosty model uczenia maszynowego do klasyfikacji sentymentu (tonalności) recenzji kursów edukacyjnych napisanych w języku polskim. Projekt stanowi praktyczną demonstrację podstaw przetwarzania języka naturalnego (NLP) na potrzeby zadania akademickiego.

## Opis Projektu

Głównym celem projektu jest automatyczna klasyfikacja recenzji tekstowych jako **Pozytywne (1)** lub **Negatywne (0)**. Model jest trenowany na małym, zrównoważonym zbiorze 20 ręcznie przygotowanych opinii dotyczących fikcyjnych kursów edukacyjnych.

### Główne Funkcjonalności
-   Klasyfikacja sentymentu tekstu w języku polskim.
-   Wykorzystanie techniki wektoryzacji TF-IDF do konwersji tekstu na dane numeryczne.
-   Zastosowanie modelu Regresji Logistycznej do klasyfikacji binarnej.
-   Szczegółowa ocena wydajności modelu przy użyciu standardowych metryk, takich jak Dokładność (Accuracy) i Macierz Pomyłek (Confusion Matrix).

## Wykorzystane Technologie
-   **Python 3**
-   **Scikit-learn:** Do implementacji modeli uczenia maszynowego, wektoryzacji i metryk.
-   **NumPy:** Do wydajnych operacji numerycznych.

## Instalacja i Uruchomienie

Postępuj zgodnie z poniższymi krokami, aby uruchomić projekt lokalnie.

1.  **Sklonuj repozytorium:**
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/Polish-Sentiment-Analysis-Classifier.git
    cd Polish-Sentiment-Analysis-Classifier
    ```

2.  **Stwórz i aktywuj środowisko wirtualne (zalecane):**
    ```bash
    # Dla systemu Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Dla systemów macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Zainstaluj wymagane zależności:**
    ```bash
    pip install -r requirements.txt
    ```

## Sposób Użycia
Aby uruchomić skrypt klasyfikacyjny i zobaczyć wyniki, wykonaj następujące polecenie w terminalu:

```bash
python sentiment_classifier.py
```

Skrypt wyświetli w konsoli dokładność modelu oraz szczegółową macierz pomyłek.

## Wyniki i Ocena Modelu

Model został wytrenowany i oceniony na tym samym zbiorze 20 recenzji. Na tych danych osiągnął **100% dokładności**.

### Macierz Pomyłek (Confusion Matrix)
Macierz pomyłek pozwala na bardziej szczegółową analizę wydajności modelu.

|                       | Predykcja: Negatywna | Predykcja: Pozytywna |
|-----------------------|----------------------|----------------------|
| **Rzeczywista: Neg.** | **TN = 10**          | FP = 0               |
| **Rzeczywista: Poz.** | FN = 0               | **TP = 10**          |

-   **True Positives (TP):** 10 - Poprawnie zidentyfikowane recenzje pozytywne.
-   **True Negatives (TN):** 10 - Poprawnie zidentyfikowane recenzje negatywne.
-   **False Positives (FP):** 0 - Żadna negatywna recenzja nie została błędnie sklasyfikowana jako pozytywna.
-   **False Negatives (FN):** 0 - Żadna pozytywna recenzja nie została błędnie sklasyfikowana jako negatywna.

Doskonały wynik wskazuje, że model był w stanie znaleźć wyraźne, rozróżnialne wzorce w dostarczonych danych tekstowych.

## Licencja
Ten projekt jest udostępniany na licencji MIT. Zobacz plik `LICENSE`, aby uzyskać więcej informacji.