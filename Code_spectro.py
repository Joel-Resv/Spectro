import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
data = pd.read_csv('donnees.csv')

# Extraire les colonnes x et y
x = data['Colonne_X']
y = data['Colonne_Y']

# Tracer le graphique
plt.plot(x, y, label='calibration')

# Ajouter des titres et une légende
plt.title('Graphique à partir de données CSV')
plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.legend()

# Afficher le graphique
plt.show()
