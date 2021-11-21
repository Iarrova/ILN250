import time

import numpy as np
import sympy as sp

"""
---------------------------------
1.- Definicion del Problema
---------------------------------

-------------------------------------
    1.1.- Definicion de Parametros
-------------------------------------
"""
M = 100       # Tamano de la matriz de parametros (filas) (i)
N = 2         # Tamano de la matriz de parametros (columnas) (j)
MIN = -10     # Valor minimo que puede obtener la distribucion uniforme
MAX = 10      # Valor maximo que puede obtener la distribucion uniforme

A = np.random.uniform(MIN, MAX, (M, N))
B = np.random.uniform(MIN, MAX, M)

A = np.round(A, 2)
B = np.round(B, 2)


"""
-----------------------------------------------------------------
    1.2.- Definicion de Funcion Objetivo, Derivadas y Hessiano
-----------------------------------------------------------------
"""
# Definimos la funcion objetivo
i, j, m, n = sp.symbols('i j m n', integer=True)    # Indexes para las sumatorias
x = sp.IndexedBase('x')                             # Variable 
a = sp.MatrixSymbol('a', M, N)                      # Matriz de parametros 
b = sp.IndexedBase('b')                             # Vector de parametros

f = sp.ln(sp.Sum(sp.exp(sp.Sum(a[i, j]*x[j]+b[i], (j, 0, n-1))), (i, 0, m-1)))

def evaluate_f(point):
    # Nos permite evaluar el valor de la funcion en un punto
    # Retorna un valor flotante
    return float(f.subs(x, sp.Matrix(point)).subs(a, sp.Matrix(A)).subs(b, sp.Matrix(B)).subs(m, M).subs(n, N).doit())

def get_gradient_f(point):
    # Nos permite obtener el gradiente de la funcion en un punto
    # Retorna una lista de flotantes como arreglo NumPy
    gradient = []
    for i in range(len(point)):
        gradient.append(sp.diff(f, x[i]).subs(x, sp.Matrix(point)).subs(a, sp.Matrix(A)).subs(b, sp.Matrix(B)).subs(m, M).subs(n, N).doit())
    return np.array(gradient).astype('float')

def get_hessian_f(point):
    # Nos permite obtener el Hessiano de la funcion en un punto
    # Retorna una matriz de flotantes como arreglo NumPy
    hessian = []
    for i in range(len(point)):
        tmp = []
        for j in range(len(point)):
            tmp.append(sp.diff(f, x[i], x[j]).subs(x, sp.Matrix(point)).subs(a, sp.Matrix(A)).subs(b, sp.Matrix(B)).subs(m, M).subs(n, N).doit())
        hessian.append(tmp)
    
    return np.array(hessian).astype('float')


"""
---------------------------------
2.- Desarrollo
---------------------------------

------------------------------------------------
    2.1.- Metodo de Gradiente con Backtracking
------------------------------------------------
"""
# Definimos el metodo de calculo de direccion para el metodo de la gradiente
def calculate_direction_gradient(point):
    # Calcula el gradiente negativo (ya que es minimizacion) evaluado en un punto
    # Retorna un arreglo Numpy de flotantes
    direction = get_gradient_f(point)
    return -np.array(direction)


def calculate_step_backtracking(point, direction, alpha, beta):
    # Calcula el paso utilizando el metodo de backtracking
    # Retorna un valor flotante

    # Transformar a np.array para poder realizar la suma y multiplicacion por componentes.
    vector_point = np.array(point)
    vector_direction = np.array(direction)

    t = 1       # Inicializar el paso como 1
    while(evaluate_f(vector_point + t*vector_direction) > (evaluate_f(vector_point) + alpha * t * np.dot(np.transpose(get_gradient_f(vector_point)), vector_direction))):
        t = beta*t
    return t


def optimize_gradient_backtracking(point, epsilon, alpha, beta):
    iteration = 0
    start = time.time()

    print('Iteration: {}: {} - {}'.format(iteration, point, evaluate_f(point)))

    # Mientras la norma del gradiente sea mayor a epsilon, debemos seguir iterando
    # Este es el criterio de parada para el metodo de la gradiente
    while(np.linalg.norm(get_gradient_f(point), 2) > epsilon):
        # Calculamos la direccion y el paso
        direction = calculate_direction_gradient(point)
        step = calculate_step_backtracking(point, direction, alpha, beta)

        # Transformamos a np.array para poder hacer la suma por elementos
        vector_point = np.array(point)
        vector_direction = np.array(direction)

        # Calculamos el nuevo punto x(k+1)
        point = vector_point + (step*vector_direction)

        # Actualizamos el contador de iteraciones
        iteration = iteration + 1

    end = time.time()

    print('Iteration: {}: {} - {}'.format(iteration, point, evaluate_f(point)))
    print('Time [s]: {}'.format(end - start))


"""
------------------------------------------------
    2.2.- Metodo de Newton con Backtracking
------------------------------------------------
"""

# Definimos el metodo de calculo de direccion para el metodo de Newton
def calculate_direction_newton(point):
    # Calcula el paso de Newton junto con el valor Lambda^2
    # Retorna un arreglo Numpy de flotantes y un flotante

    # Calculamos el paso de Newton
    inv_hessian = np.linalg.inv(get_hessian_f(point))
    gradient = get_gradient_f(point)
    direction = -np.dot(inv_hessian, gradient)

    # Calcular el decremento de Newton
    lambda_squared = np.dot(np.transpose(get_gradient_f(point)), -direction)
    return direction, lambda_squared


def optimize_newton_backtracking(point, epsilon, alpha, beta):
    iteration = 0
    start = time.time()
    lambda_squared = np.inf

    print('Iteration: {}: {} - {}'.format(iteration, point, evaluate_f(point)))

    # Mientras Lambda^2/2 sea mayor a epsilon debemos seguir iterando
    # Este es el criterio de parada para el metodo de Newton
    while(lambda_squared/2 > epsilon):
        # Calculamos la direccion y el paso
        direction, lambda_squared = calculate_direction_newton(point)
        step = calculate_step_backtracking(point, direction, alpha, beta)

        # Transformamos a np.array para poder hacer la suma por elementos
        vector_point = np.array(point)
        vector_direction = np.array(direction)

        # Calculamos el nuevo punto x(k+1)
        point = vector_point + (step*vector_direction)

        # Actualizamos el contador de iteraciones
        iteration = iteration + 1

    end = time.time()

    print('Iteration: {}: {} - {}'.format(iteration, point, evaluate_f(point)))
    print('Time [s]: {}'.format(end - start))


"""
------------------------------------------------
3.- Ejecucion
------------------------------------------------
"""
# Ya que tenemos todas las funciones y parametros definidas, podemos ejecutar las respectivas
# funciones para hacer el calculo del optimo

POINT = (1,5)       # Definicion del punto inicial
EPSILON = 10e-6     # Definicion del epsilon para los criterios de parada
ALPHA = 0.4         # Definicion de alpha para backtracking (0 < alpha < 0.5)
BETA = 0.9          # Definicion de beta para backtracking (0 < beta < 1)

print('Optimizing with Gradient and Backtracking Method')
optimize_gradient_backtracking(POINT, EPSILON, ALPHA, BETA)
print()
print('Optimizing with Newton and Backtracking method')
optimize_newton_backtracking(POINT, EPSILON, ALPHA, BETA)