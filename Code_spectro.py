import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
#data1 = pd.read_csv('calibration', sep = ' ', header = None)

# Extraire les colonnes x et y

# Tracer le graphique
#plt.plot(x, y, label='Données')

# Ajouter des titres et une légende
#plt.title('Graphique à partir de données CSV')
#plt.xlabel('Axe X')
#plt.ylabel('Axe Y')
#plt.legend()

# Afficher le graphique
#plt.show()

# Ouvrir le fichier en mode lecture
with open('spectre_99%', 'r') as fichier:
    # Lire toutes les lignes du fichier et les stocker dans une liste
    lignes = fichier.readlines()

# Traiter la première ligne pour obtenir la dernière valeur
premiere_ligne_elements = lignes[0].strip().split()
derniere_valeur_premiere_ligne = float(premiere_ligne_elements[-1])
print("Dernière valeur de la première ligne :", derniere_valeur_premiere_ligne)

# Traiter la deuxième ligne pour obtenir la première valeur
deuxieme_ligne_elements = lignes[1].strip().split()
premiere_valeur_deuxieme_ligne = float(deuxieme_ligne_elements[0])
print("Première valeur de la deuxième ligne :", premiere_valeur_deuxieme_ligne)

# Imprimer la première ligne si la dernière valeur est inférieure à la première valeur de la deuxième ligne
if derniere_valeur_premiere_ligne < premiere_valeur_deuxieme_ligne:
    print("Première ligne :", premiere_ligne_elements)

# Imprimer la deuxième ligne si la première valeur est inférieure à la dernière valeur de la première ligne
if premiere_valeur_deuxieme_ligne > derniere_valeur_premiere_ligne:
    print("Deuxième ligne :", deuxieme_ligne_elements)
    
# Tracer le graphique
plt.plot(deuxieme_ligne_elements, premiere_ligne_elements, label='Données')

# Ajouter des titres et une légende
plt.title('Spectre de calibration du mercure')
plt.xlabel('Intensité')
plt.ylabel('Longueur onde (nm)')
plt.legend()

# Afficher le graphique
plt.show()



