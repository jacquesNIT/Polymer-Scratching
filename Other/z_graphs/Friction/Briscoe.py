import matplotlib.pyplot as plt

# 1. Définition des données du tableau
# Les valeurs de l'axe X correspondent aux modifications/tailles de maillage
x_mesh = [0.04, 0.03, 0.02, 0.015]

# Données SCOF pour chaque famille (sans et avec Briscoe)
data = {
    'Crys': {
        'without': [0.463, 0.471, 0.491, 0.495],
        'with': [0.312, 0.32, 0.332, 0.335]
    },
    'PMMA': {
        'without': [0.413, 0.422, 0.449, 0.468],
        'with': [0.314, 0.325, 0.35, 0.359]
    },
    'PC': {
        'without': [0.438, 0.446, 0.472, 0.485],
        'with': [0.348, 0.354, 0.375, 0.387]
    },
    'DP': {
        'without': [0.426, 0.438, 0.484, 0.492],
        'with': [0.322, 0.335, 0.364, 0.378]
    }
}

# Attribution d'une couleur unique par famille de matériau
colors = {
    'Crys': '#1f77b4',  # Bleu
    'PMMA': '#ff7f0e',  # Orange
    'PC': '#2ca02c',    # Vert
    'DP': '#d62728'     # Rouge
}

# 2. Création du graphique
plt.figure(figsize=(10, 6))

for family, cases in data.items():
    # Courbe "sans Briscoe" : Ligne continue (-) et marqueurs ronds (o)
    plt.plot(x_mesh, cases['without'], 
             marker='o', linestyle='-', color=colors[family], 
             label=f'{family} (without Briscoe)')
    
    # Courbe "avec Briscoe" : Ligne pointillée (--) et marqueurs carrés (s)
    plt.plot(x_mesh, cases['with'], 
             marker='s', linestyle='--', color=colors[family], 
             label=f'{family} (with Briscoe)')

# 3. Personnalisation et habillage du graphique
plt.title("Évolution du SCOF en fonction de la convergence en maillage", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Paramètre de maillage (Friction changes)", fontsize=12)
plt.ylabel("SCOF", fontsize=12)

# Inversion de l'axe X pour suivre la convergence (du maillage le plus grossier au plus fin)
plt.gca().invert_xaxis()

# Ajout d'une grille pour faciliter la lecture des valeurs
plt.grid(True, linestyle=':', alpha=0.6)

# Positionnement de la légende à l'extérieur du graphique pour éviter les superpositions
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Ajustement automatique des marges
plt.tight_layout()

# 4. Affichage du graphique
plt.show()