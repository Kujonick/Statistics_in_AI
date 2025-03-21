### Laboratoria: Statystyka w Sztucznej Inteligencji i Analizie Danych

### 1. Przygotowanie środowiska

#### 1.1 Wymagania:
 - python==3.13: https://www.python.org
 - R==4.4: https://cran.r-project.org
 - IDE (zalecane RStudio do pracy z kodem w R)

#### 1.2 Uruchomienie projektu:

Aby zsynchronizować zainstalowane pakiety i uruchomić pliki .rmd, należy wykonać poniższe kroki:

 1. Sklonowanie repozytorium do wybranego folderu.
 2. Otworzenie projektu w RStudio (File -> Open Project -> Wybranie pliku SwSI.Rproj). Spowoduje to automatyczne pobranie pakietu kontroli wersji renv.
 3. Zainstalowanie zależności R (renv::restore()). 
 4. Utworzenie środowiska wirtualnego python (python -m venv venv), aktywowanie go oraz zainstalowanie zależności (pip install -r requirements.txt).
 5. Wskazanie w Rstudio interpretera python (Tools -> Global Options -> Python) na plik wykonwalny ze środowiska wirtualnego

Po wykonaniu powyższych należy zweryfikować działanie wykonując komórki w notebooku lab0.rmd
