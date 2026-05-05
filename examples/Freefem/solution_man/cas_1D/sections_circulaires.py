import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================================
# 1D : REPRÉSENTATION D'UNE BARRE (modèle linéique)
# ============================================================================

def visualize_1D_barre():
    fig, ax = plt.subplots(figsize=(10, 2))
    
    L = 10
    ax.plot([0, L], [0, 0], 'b-', linewidth=8, label='Barre (modèle 1D)')
    ax.plot(0, 0, 'ro', markersize=10, label='Encastrement')
    ax.annotate('F', xy=(L, 0.1), xytext=(L+0.5, 0.1), 
                arrowprops=dict(arrowstyle='->', lw=2), fontsize=14)
    
    ax.set_xlim(-1, L+2)
    ax.set_ylim(-1, 1)
    ax.set_title('1D : Modèle barre (section = aire A uniquement)', fontsize=12)
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.text(L/2, -0.5, f'Aire de section = π × R² (mais non visible)', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('section_1D_barre.png', dpi=150)
    plt.show()
    print("✓ Figure 1D générée : section_1D_barre.png")
    print("  → En 1D, on ne voit que la ligne. La section circulaire est 'cachée' dans le paramètre A.\n")

# ============================================================================
# 2D : SECTION CIRCULAIRE (vue en coupe)
# ============================================================================

def visualize_2D_section_circulaire(rayon=1.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    theta = np.linspace(0, 2*np.pi, 100)
    x_cercle = rayon * np.cos(theta)
    y_cercle = rayon * np.sin(theta)
    ax.fill(x_cercle, y_cercle, 'lightblue', edgecolor='blue', linewidth=2, alpha=0.5)
    
    ax.plot([0, rayon], [0, 0], 'r-', linewidth=2, label=f'Rayon R = {rayon}')
    ax.plot(0, 0, 'ro', markersize=5)
    
    ax.annotate('R', xy=(rayon/2, 0.05), xytext=(rayon/2, 0.2), fontsize=12)
    ax.set_xlim(-rayon*1.2, rayon*1.2)
    ax.set_ylim(-rayon*1.2, rayon*1.2)
    ax.set_aspect('equal')
    ax.set_title(f'2D : Section circulaire (Aire = π × R² = {np.pi*rayon**2:.3f})', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('section_2D_cercle.png', dpi=150)
    plt.show()
    print("✓ Figure 2D générée : section_2D_cercle.png")
    print(f"  → Aire de la section = π × {rayon}² = {np.pi*rayon**2:.3f}\n")

# ============================================================================
# 3D : POUTRE À SECTION CIRCULAIRE (version CORRIGÉE)
# ============================================================================

def create_cylinder_mesh(radius=0.5, height=5.0, n_theta=20, n_z=10):
    """
    Crée un maillage de cylindre pour visualisation 3D.
    Retourne les points (vertices) et les faces (triangles).
    """
    # Utiliser des listes Python standard (pas numpy au début)
    vertices = []
    faces = []
    
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Anneau du bas (z = 0)
    bottom_indices = []
    for i, t in enumerate(theta):
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        vertices.append([x, y, 0])
        bottom_indices.append(i)
    
    # Anneau du haut (z = height)
    top_indices = []
    for i, t in enumerate(theta):
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        vertices.append([x, y, height])
        top_indices.append(len(vertices) - 1)
    
    # Faces latérales (quadrangles divisés en triangles)
    n = n_theta
    for i in range(n):
        i_next = (i + 1) % n
        # Premier triangle du quadrangle
        faces.append([bottom_indices[i], bottom_indices[i_next], top_indices[i]])
        # Deuxième triangle du quadrangle
        faces.append([bottom_indices[i_next], top_indices[i_next], top_indices[i]])
    
    # Disque du bas (centre + triangles)
    center_bottom = len(vertices)
    vertices.append([0, 0, 0])
    for i in range(n):
        i_next = (i + 1) % n
        faces.append([center_bottom, bottom_indices[i], bottom_indices[i_next]])
    
    # Disque du haut (centre + triangles)
    center_top = len(vertices)
    vertices.append([0, 0, height])
    for i in range(n):
        i_next = (i + 1) % n
        faces.append([center_top, top_indices[i], top_indices[i_next]])
    
    return np.array(vertices), faces

def visualize_3D_poutre_circulaire(radius=0.5, height=5.0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Générer le maillage du cylindre
    vertices, faces = create_cylinder_mesh(radius, height, n_theta=30, n_z=15)
    
    # Créer la collection de faces
    mesh = Poly3DCollection(vertices[faces], alpha=0.6, edgecolor='gray', linewidth=0.1)
    mesh.set_facecolor('lightblue')
    ax.add_collection3d(mesh)
    
    # Ajouter une flèche pour la traction
    ax.quiver(0, 0, height, 0, 0, 1, length=1.5, color='red', linewidth=3, 
              arrow_length_ratio=0.3, label='Force de traction F')
    
    # Ajouter un plan de coupe pour montrer la section circulaire
    z_coupe = height / 2
    theta_coupe = np.linspace(0, 2*np.pi, 100)
    x_coupe = radius * np.cos(theta_coupe)
    y_coupe = radius * np.sin(theta_coupe)
    z_coupe_arr = np.ones_like(x_coupe) * z_coupe
    ax.plot(x_coupe, y_coupe, z_coupe_arr, 'r-', linewidth=2, label='Section circulaire (coupe)')
    
    # Annotations
    ax.text(0, 0, -0.5, 'Encastrement', ha='center', fontsize=10, style='italic')
    ax.text(radius+0.3, 0, height/2, f'Rayon R = {radius}', fontsize=10)
    ax.text(0, 0, height+0.8, 'Traction', ha='center', fontsize=10)
    
    # Configuration des axes
    ax.set_xlim([-radius*1.5, radius*1.5])
    ax.set_ylim([-radius*1.5, radius*1.5])
    ax.set_zlim([-0.5, height+1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D : Poutre à section circulaire (cylindre)\nAire = π × R² = {np.pi*radius**2:.3f}', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('section_3D_poutre_cylindre.png', dpi=150)
    plt.show()
    print("✓ Figure 3D générée : section_3D_poutre_cylindre.png")
    print(f"  → Géométrie : cylindre de rayon {radius} et hauteur {height}")
    print(f"  → Aire de la section = π × {radius}² = {np.pi*radius**2:.3f}")
    print("  → La section circulaire est visible sur le plan de coupe (ligne rouge).\n")

# ============================================================================
# COMPARAISON DES TROIS REPRÉSENTATIONS (version CORRIGÉE)
# ============================================================================

def visualize_comparison():
    """Crée une figure de synthèse comparant 1D, 2D et 3D."""
    fig = plt.figure(figsize=(15, 5))
    
    # 1D : Ligne
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot([0, 10], [0, 0], 'b-', linewidth=8)
    ax1.plot(0, 0, 'ro', markersize=10)
    ax1.annotate('F', xy=(10, 0.1), xytext=(10.5, 0.1), 
                arrowprops=dict(arrowstyle='->', lw=2))
    ax1.set_xlim(-1, 12)
    ax1.set_ylim(-1, 1)
    ax1.set_title('1D : Modèle barre\n(la section est "cachée" dans A)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2D : Cercle (section)
    ax2 = fig.add_subplot(1, 3, 2)
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.fill(np.cos(theta), np.sin(theta), 'lightblue', edgecolor='blue', alpha=0.5)
    ax2.plot([0, 1], [0, 0], 'r-', linewidth=2)
    ax2.plot(0, 0, 'ro', markersize=5)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('2D : Section transversale\n(le cercle = forme de la coupe)')
    ax2.grid(True, alpha=0.3)
    
    # 3D : Cylindre
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    radius, height = 0.5, 3
    vertices, faces = create_cylinder_mesh(radius, height, n_theta=20, n_z=8)
    mesh = Poly3DCollection(vertices[faces], alpha=0.5, edgecolor='gray', linewidth=0.1)
    mesh.set_facecolor('lightblue')
    ax3.add_collection3d(mesh)
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([0, 4])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D : Poutre complète\n(cylindre = barre réelle)')
    
    plt.suptitle('Comparaison 1D / 2D / 3D : la "section circulaire"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparaison_1D_2D_3D.png', dpi=150)
    plt.show()
    print("✓ Figure de comparaison générée : comparaison_1D_2D_3D.png")

# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALISATION DE LA SECTION CIRCULAIRE EN 1D, 2D ET 3D")
    print("=" * 60)
    print()
    
    # 1D : Modèle barre
    visualize_1D_barre()
    
    # 2D : Section transversale (cercle)
    visualize_2D_section_circulaire(rayon=1.0)
    
    # 3D : Poutre cylindrique
    visualize_3D_poutre_circulaire(radius=0.8, height=5.0)
    
    # Comparaison
    visualize_comparison()
    
