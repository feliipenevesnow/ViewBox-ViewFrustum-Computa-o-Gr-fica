import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def matriz_translacao(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])


def matriz_rotacao(theta_x, theta_y, theta_z):
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return np.dot(np.dot(Rz, Ry), Rx)


def matriz_escala(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def transformar_vertice(vertice, matriz_transformacao):
    vertice_homogeneo = np.append(vertice, 1)
    vertice_resultante = np.dot(matriz_transformacao, vertice_homogeneo)
    return vertice_resultante[:3]

vertices = np.array([
    [-5, -5, -5],
    [5,-5,-5],
    [5, 5, -5],
    [-5, 5, -5],
    [-100, -100, -100],
    [100, -100, -100],
    [100, 100, -100],
    [-100, 100, -100]
])

left = 5
bottom = -5

right = -5
top = 5

near = -5
far = -100

matriz_projecao = np.array([
    [2*near / (right - left), 0,  (right + left) / (right - left), 0],
    [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
    [0, 0,  -((far + near) / (far - near)), -(2*far*near/far-near)],
    [0, 0, -1, 0]
])

arestas = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

triangulo3d_vertices = np.array([
    [2, -2, 0],
    [2, -2, -2],
    [4, -2, 0],
    [4, -2, -2],
    [3, 1, -1]
])

triangulo3d_arestas = [
    [0,  1], [2,  0],
    [1,  3], [2,  3],
    [0,  4], [1,  4],
    [2,  4], [3,  4]
]


translacao = matriz_translacao(-3, 0, -5)

transtalado = []

for vertice in triangulo3d_vertices:
    vertice = transformar_vertice(vertice, translacao)
    transtalado.append(vertice)

triangulo3d_vertices = transtalado


figura = plt.figure(figsize=(16, 10))


eixo1 = figura.add_subplot(121, projection='3d')


for aresta in arestas:
    eixo1.plot(vertices[aresta, 0], vertices[aresta, 2], vertices[aresta, 1], color='black')

for aresta in triangulo3d_arestas:
    v1_idx, v2_idx = aresta
    v1 = triangulo3d_vertices[v1_idx]
    v2 = triangulo3d_vertices[v2_idx]
    eixo1.plot(
        [v1[0], v2[0]],
        [v1[2], v2[2]],
        [v1[1], v2[1]],
        color='blue', label='Triângulo 3D'
    )

eixo1.set_xlabel('X')
eixo1.set_ylabel('Z')
eixo1.set_zlabel('Y')
eixo1.set_title('Triângulo 3D (Modificado)')

def projetar_para_2d(vertice, matriz_projecao):
    vertice_homogeneo = np.append(vertice, 1)
    vertice_resultante = np.dot(matriz_projecao, vertice_homogeneo)
    x_2d = vertice_resultante[0] / vertice_resultante[3]
    y_2d = vertice_resultante[1] / -vertice_resultante[3]
    return x_2d, y_2d


triangulo2d_vertices = []

triangulo2d_arestas = []

for aresta in triangulo3d_arestas:
    v1_idx, v2_idx = aresta
    v1 = triangulo3d_vertices[v1_idx]
    v2 = triangulo3d_vertices[v2_idx]

    if v1[2] >= -100 and v2[2] >= -100:
        x1_2d, y1_2d = projetar_para_2d(v1, matriz_projecao)
        x2_2d, y2_2d = projetar_para_2d(v2, matriz_projecao)

        triangulo2d_vertices.append([x1_2d, y1_2d])
        triangulo2d_vertices.append([x2_2d, y2_2d])

        triangulo2d_arestas.append([len(triangulo2d_vertices) - 2, len(triangulo2d_vertices) - 1])

triangulo2d_vertices = np.array(triangulo2d_vertices)

eixo2 = figura.add_subplot(122)


for aresta in triangulo2d_arestas:
    v1_idx, v2_idx = aresta
    v1 = triangulo2d_vertices[v1_idx]
    v2 = triangulo2d_vertices[v2_idx]
    eixo2.plot(
        [v1[0], v2[0]],
        [v1[1], v2[1]],
        color='blue', label='Triângulo 2D'
    )

eixo2.set_xlim(-1, 1)
eixo2.set_ylim(-1, 1)
eixo2.set_xlabel('X')
eixo2.set_ylabel('Y')
eixo2.set_title('Triângulo 3D Projetado em 2D')
eixo1.invert_xaxis()

plt.show()