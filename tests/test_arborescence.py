



import os

def afficher_arborescence(dossier, prefixe=""):
    if not os.path.exists(dossier):
        print(f"Le dossier '{dossier}' n'existe pas.")
        return

    fichiers_et_dossiers = sorted(os.listdir(dossier))
    for index, nom in enumerate(fichiers_et_dossiers):
        chemin_complet = os.path.join(dossier, nom)
        est_dernier = (index == len(fichiers_et_dossiers) - 1)
        branche = "└── " if est_dernier else "├── "
        print(prefixe + branche + nom)
        if os.path.isdir(chemin_complet):
            nouveau_prefixe = prefixe + ("    " if est_dernier else "│   ")
            afficher_arborescence(chemin_complet, nouveau_prefixe)

chemin_racine = r"C:\Users\steph\Downloads\Projet_IA"
afficher_arborescence(chemin_racine)
