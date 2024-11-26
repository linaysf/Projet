#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:08:50 2024

@author: yousfilina
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spsopt
import sympy as sp
from scipy.linalg import eigh

#%% Initialisation des paramètres

L = 1
l = 30*10**(-2)
M = 10
rho = 2300
E = 35*10**10
cL = np.sqrt(E/rho)
I = (M*L**2)/3
S = L*l
cT = np.sqrt((E*I)/(rho*S))

#%% les matrices K et M pour la traction/compression

def Matrix_K(p):
    # Déclaration de la matrice K
    K = np.zeros((p+1, p+1))
    
    # Déclaration du symbole pour l'intégration
    x = sp.Symbol('x')
    
    # Boucles sur les indices i et j
    for i in range(p+1):
        for j in range(p+1):
            # Définir P_i et Q_j
            P_i = x**(i+1) 
            dP_i = sp.diff(P_i, x)  # Dérivée par rapport à x
            
            Q_j = Q_j = x**(j+1) - x**(j+2) # Exemple de Q_j
            dQ_j = sp.diff(Q_j, x)  # Dérivée par rapport à x
            
            # Intégration symbolique
            K[i, j] = sp.integrate(dP_i * dQ_j, (x, 0, 1))
            # Symétrisation explicite de la matrice K
    K = (K + K.T) / 2  # Moyenne entre K et sa transposée pour forcer la symétrie
    
    # Conversion de la matrice en type float
    return np.array(K).astype(float)

def Matrix_M(p):
    # Déclaration de la matrice K
    M = np.zeros((p+1, p+1))
    
    # Déclaration du symbole pour l'intégration
    x = sp.Symbol('x')
    
    # Boucles sur les indices i et j
    for i in range(p+1):
        for j in range(p+1):
            # Définir P_i et Q_j
            P_i = x**(i+1)  # Exemple de P_i
            dP_i = sp.diff(P_i, x)  # Dérivée par rapport à x
            ddP_i = sp.diff(dP_i,x) #Dérivé seconde
            
            Q_j = x**(j+1) - x**(j+2) # Exemple de Q_j
            dQ_j = sp.diff(Q_j, x)  # Dérivée par rapport à x
            ddQ_j = sp.diff(dQ_j,x) #derivé seconde
            # Intégration symbolique
            M[i, j] = sp.integrate(P_i * Q_j, (x, 0, 1))
    M = (M + M.T) / 2
    # Conversion de la matrice en type float
    return np.array(M).astype(float)
#%% Valeurs propres et vecteurs propres pour la traction/compression


#%% Valeurs propres et vecteurs propres pour la traction/compression

def Q(p):
    # Résolution des valeurs propres généralisées
    K = Matrix_K(p)
    M = Matrix_M(p)
    # Résolution du problème aux valeurs propres généralisé
    eigenvalues, eigenvectors = eigh(K, M)
    _, M_eigenvalues = eigh(M)
    return eigenvalues, eigenvectors, M_eigenvalues # Résout KQ

eigenvalues, eigenvectors, M_eigenvalues = Q(12)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
print("Eigenvalues of M:\n", M_eigenvalues)

#%% Vecteur/pulsation propres analytiques et u_analytique

n_modes = 13
n = np.arange(len(eigenvalues))  # Index des modes
w_analytique = (((2 * n + 1) * np.pi) / (2 * L))  # Fréquences analytiques

x = np.linspace(0, L, 1000)
t = 0  # Moment pour lequel afficher les déplacements

# Calcul des fréquences analytiques (pulsations) et des vecteurs propres
n = np.arange(n_modes)  # Modes indexés de 0 à 11
w_analytique = ((2 * n + 1) * np.pi) / (2 * L)  # Fréquences analytiques
analytical_eigenvectors = np.sin(((2 * n[:, None] + 1) * np.pi * x) / (2 * L))  # Vecteurs propres #[:,None] permet de convertir un tableau unidimensionnel en une matrice colonne

# Affichage du déplacement analytique pour chaque mode n
plt.figure(figsize=(10, 6))
for i in range(n_modes):
    u_analytique = analytical_eigenvectors[i] * np.cos(w_analytique[i] * t)  # Déplacement pour chaque mode
    plt.plot(x, u_analytique, label=f'Mode {i}')
plt.plot(x,np.zeros_like(x),linewidth=5, label='Poutre')
plt.xlabel("Position (x)")
plt.ylabel("Déplacement analytique (u)")
plt.title("Déplacement analytique pour chaque mode de vibration")
plt.legend()
plt.grid(True)
plt.show()

#%% Valeurs propres et vecteurs propres pour la traction/compressioin

w_fem = np.sqrt(np.maximum(eigenvalues, 0))

print("FEM Eigenvalues:", w_fem)
print("Theoretical eigenvalues:", w_analytique)

plt.plot(n, w_fem, label='FEM Eigenvalues', marker='o')
plt.plot(n, w_analytique, label='Theoretical Eigenvalues', linestyle='--')
plt.xlabel("Mode index (n)")
plt.ylabel("Frequency")
plt.legend()
plt.title("Comparison of FEM and Analytical Eigenvalues")
plt.grid(True)
plt.show()

# Calcul de l'erreur
error = np.abs((w_analytique - w_fem) / w_analytique ) # Erreur relative (absolue si nécessaire)

# Tracé de l'erreur en échelle log-log
plt.semilogy(n[1:], error[1:], marker='o', label='Relative Error')  # Ignorer le mode 0 si nécessaire
plt.xlabel("Mode index (n)")
plt.ylabel("Relative Error")
plt.title("Error between FEM and Analytical Eigenvalues")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()


#%% Déplacement p-FEM pour la traction/compression

def phi(x, p):
    #phi1 = fi(x)
    phi1 = x**(p+1)
    #phi2 = gi(x)
    phi2 = x**(p+1) - x**(p+2)
    return phi1, phi2

def displacement(x_values, eigenvectors, p):
    u_x = np.zeros_like(x_values)
    for indice, x in enumerate(x_values):
        for i in range(p + 1):
            #recuperer les phi de la fonction
            phi_1, phi_2 = phi(x,i)
            #les eigenvectors c'est les ui et vi
            u_x[indice] += eigenvectors[i, 0] * phi_1  # Contribution longitudinale
    return u_x

x_values = np.linspace(0, L, 100)
u_x = displacement(x_values, eigenvectors, 12)

plt.figure(figsize=(10, 6))
plt.plot(x_values, u_x, label="Déplacement longitudinal $u(x)$")
plt.xlabel("x")
plt.ylabel("Déplacement")
plt.title("Déplacements longitudinaux et transversaux (p-FEM)")
plt.legend()
plt.grid()
plt.show()

