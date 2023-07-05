import tkinter
from collections import defaultdict, deque
import copy

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)
    def set_V(self,a):
        self.V=a
    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
    def add_edge_directed(self, u, v, weight):
        self.graph[u].append((v, weight))
    def reset(self):
        self.graph = defaultdict(list)
    def bfs(self, rGraph, s, t, parent):
        visited = [False] * self.V
        queue = []
        queue.append(s)
        visited[s] = True

        while queue:
            u = queue.pop(0)

            for v, weight in rGraph[u]:
                if not visited[v] and weight > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u

        return visited[t]

    def ford_fulkerson(self, source, sink):
        rGraph = copy.deepcopy(self.graph)
        parent = [-1] * self.V
        max_flow = 0

        while self.bfs(rGraph, source, sink, parent):
            path_flow = float('inf')
            v = sink

            while v != source:
                u = parent[v]
                for i, (neighbor, weight) in enumerate(rGraph[u]):
                    if neighbor == v:
                        path_flow = min(path_flow, weight)
                        rGraph[u][i] = (neighbor, weight - path_flow)
                        break

                for i, (neighbor, weight) in enumerate(rGraph[v]):
                    if neighbor == u:
                        rGraph[v][i] = (neighbor, weight + path_flow)
                        break

                v = u

            max_flow += path_flow

        return max_flow
    def welch_powell_algorithm(self):
        sorted_vertices = sorted(self.graph.keys(), key=lambda x: len(self.graph[x]), reverse=True)
        colors = {}
        
        for vertex in sorted_vertices:
            used_colors = []
            for neighbor, _ in self.graph[vertex]:
                if neighbor in colors:
                    used_colors.append(colors[neighbor])
            
            available_colors = [color for color in range(len(self.graph)) if color not in used_colors]
            
            if not available_colors:
                color = len(self.graph)
            else:
                color = min(available_colors)
            
            colors[vertex] = color
        
        return colors
    def prim(self):
        visited = [False] * self.V
        parent = [-1] * self.V
        key = [float('inf')] * self.V
        key[0] = 0

        for _ in range(self.V):
            min_key = float('inf')
            min_vertex = -1

            for v in range(self.V):
                if not visited[v] and key[v] < min_key:
                    min_key = key[v]
                    min_vertex = v

            visited[min_vertex] = True

            for neighbor, weight in self.graph[min_vertex]:
                if not visited[neighbor] and weight < key[neighbor]:
                    parent[neighbor] = min_vertex
                    key[neighbor] = weight

        mst = []
        for v in range(1, self.V):
            mst.append((parent[v], v, key[v]))

        return mst

    def dijkstra(self, source):
        visited = [False] * self.V
        dist = [float('inf')] * self.V
        dist[source] = 0
        parent = [-1] * self.V

        for _ in range(self.V):
            min_dist = float('inf')
            min_vertex = -1

            for v in range(self.V):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    min_vertex = v

            visited[min_vertex] = True

            for neighbor, weight in self.graph[min_vertex]:
                if not visited[neighbor] and dist[min_vertex] + weight < dist[neighbor]:
                    dist[neighbor] = dist[min_vertex] + weight
                    parent[neighbor] = min_vertex

        shortest_paths = []
        for v in range(self.V):
            path = []
            current = v
            while current != -1:
                path.append(current)
                current = parent[current]
            path.reverse()
            shortest_paths.append((v, dist[v], path))

        return shortest_paths

    def kruskal(self):
        edges = []

        for u in range(self.V):
            for v, weight in self.graph[u]:
                edges.append((u, v, weight))

        edges.sort(key=lambda x: x[2])  # Sort edges by weight
        parent = [-1] * self.V
        mst = []

        def find(parent, vertex):
            if parent[vertex] == -1:
                return vertex
            return find(parent, parent[vertex])

        def union(parent, x, y):
            parent[x] = y

        for u, v, weight in edges:
            parent_u = find(parent, u)
            parent_v = find(parent, v)

            if parent_u != parent_v:
                mst.append((u, v, weight))
                union(parent, parent_u, parent_v)

        return mst
    def bellman_ford(self, source):
        dist = [float('inf')] * self.V
        dist[source] = 0
        intermediates = [[] for _ in range(self.V)]

        for _ in range(self.V - 1):
            for u in range(self.V):
                for v, weight in self.graph[u]:
                    if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        intermediates[v] = intermediates[u] + [u]

        negative_cycle = False
        for u in range(self.V):
            for v, weight in self.graph[u]:
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    negative_cycle = True
                    break

        return dist, negative_cycle, intermediates
    def dfs(self, start):
        visited = [False] * self.V
        traversal = []

        def dfs_recursive(node):
            nonlocal visited
            visited[node] = True
            traversal.append(node)

            for neighbor, _ in self.graph[node]:
                if not visited[neighbor]:
                    dfs_recursive(neighbor)

        dfs_recursive(start)

        return traversal
    def larg(self, start):
        visited = [False] * self.V
        traversal = []

        queue = deque()
        queue.append(start)
        visited[start] = True

        while queue:
            node = queue.popleft()
            traversal.append(node)

            for neighbor, _ in self.graph[node]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True

        return traversal
g = Graph(4)  # Create a graph with 4 vertices
g.add_edge_directed(0, 1, 4)
g.add_edge_directed(0, 3, 1)
g.add_edge_directed(0, 2, 6)
g.add_edge_directed(1, 3, 1)
g.add_edge_directed(2, 1, 1)
g.add_edge_directed(3, 1, 3)
g.add_edge_directed(3, 2, 1)
def create_identity_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return matrix
def concatenate_matrices(matrix1, matrix2):
    # Vérifier les dimensions des matrices
    if len(matrix1) != len(matrix2):
        raise ValueError("Les matrices doivent avoir le même nombre de lignes.")

    result = []
    for i in range(len(matrix1)):
        row = matrix1[i] + matrix2[i]
        result.append(row)

    return result
def find_max_positive(nums):
    positive_nums = [num for num in nums if num > 0]
    if positive_nums:
        max_positive = max(positive_nums)
        max_index = nums.index(max_positive)
        return  max_index
    else:
        return  None
def divide_lists(list1, list2):
    result = []
    for num1, num2 in zip(list1, list2):
        division = num1 / num2
        result.append(division)
    return result
def divide_list_by_number(lst, number):
    divided_list = [num / number for num in lst]
    return divided_list
def multiply_list_by_number(lst, number):
    divided_list = [num * number for num in lst]
    return divided_list
def subtract_and_update(list1, list2):
    for i in range(len(list1)):
        list1[i] -= list2[i]
    return list1
def add_zeros(list1, num_zeros):
    zeros_list = [0] * num_zeros
    combined_list = list1 + zeros_list
    return combined_list
def get_min_index(nums):
    positive_nums = [num for num in nums if num > 0]
    if positive_nums:
        min_positive = min(positive_nums)
        min_index = nums.index(min_positive)
        return  min_index
    else:
        return  None
def get_column(matrix, column_index):
    column = [row[column_index] for row in matrix]
    return column
def remove_zeros(lst):
    while 0 in lst:
        lst.remove(0)
    return lst
def simplex(x,C,b):
    
    Matrice=concatenate_matrices(C,create_identity_matrix(len(b)))

    coefficient=add_zeros(x, len(b))
    variables = ['x{}'.format(i) for i in range(1, len(coefficient) + 1)]
    base = ['x{}'.format(i) for i in range(len(x) + 1, len(x) + len(b)+1)]
    value=0
    colone=0
    while (colone!=None) :
        colone=find_max_positive(coefficient)
        if colone == None:
            break
        ligne=get_min_index(divide_lists(b,get_column(Matrice,colone)))
        base[ligne]=variables[colone]

        pivot=Matrice[ligne][colone]
        b[ligne]=b[ligne]/pivot
        Matrice[ligne]=divide_list_by_number(Matrice[ligne],pivot)
        value=value-coefficient[colone]*b[ligne]
        subtract_and_update(coefficient,multiply_list_by_number(Matrice[ligne],coefficient[colone]))
        for i in range(len(b)):
            if i!=ligne:
                b[i]=b[i]-b[ligne]*Matrice[i][colone]
                subtract_and_update(Matrice[i],multiply_list_by_number(Matrice[ligne],Matrice[i][colone]))
                

        

    return base,b , -value
x=[]
C=[]
b=[]
def reset_simplexe(x,C,b):
    x=[]
    C=[]
    b=[]


"""""
optimal_value, optimal_solution = simplex_method(c, A, b)
print("Valeur optimale:", optimal_value)
print("Solution optimale:", optimal_solution)

(0, 1, 16)
(0, 2, 13)
(1, 2, 10)
(1, 3, 12)
(2, 1, 4)
(2, 4, 14)
(3, 2, 9)
(3, 5, 20)
(4, 3, 7)
(4, 5, 4)
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 3)
g.add_edge(1, 2, -2)
g.add_edge(2, 3, 1)
g.add_edge(3, 1, -4)
"""""
def algorithme():
    def ford_fulkerson_():
        try:     
            max_flow = g.ford_fulkerson(int(start.get()), int(final.get()))
            screen.insert('end',"le flaux maximal entre le  noeud "+start.get()+"vers le noeud "+final.get()+": "+str(max_flow)+"\n")
            final.delete(0, tkinter.END)
            start.delete(0, tkinter.END)
        except  Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n")
    def prim_():
        try:
            mst_prim = g.prim()
            screen.insert('end',"l`arbre couvrantes au poid minimum (prim) : \n")
            for u, v, weight in mst_prim:
                screen.insert('end',"Arc: "+str(u)+" -- "+str(v)+" , poid : "+str(weight)+"\n")
        except  Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n")
    def krustal_():
        try:
            mst_kruskal = g.kruskal()
            screen.insert('end',"l`arbre couvrantes au poid minimum (Kruskal):\n")
            for u, v, weight in mst_kruskal:
                screen.insert('end',"Arc: "+str(u)+" -- "+str(v)+" , Poid: "+str(weight)+"\n")
        except  Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n") 

    def dijkstra_():
        try:
            shortest_paths = g.dijkstra(int(start.get()))
            screen.insert('end',"le plus court chemin (Dijkstra):\n")
            for v, dist, path in shortest_paths:
                screen.insert('end',"le plus court chemin vers le noeud "+str(v)+":\n")
                screen.insert('end'," -> ".join(str(node) for node in path))
                screen.insert('end',"\nDistance: "+str(dist)+"\n")
            final.delete(0, tkinter.END)
            start.delete(0, tkinter.END)
        except  Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n") 
    def ford_bellman_():
        try:
            distances, has_negative_cycle,intermediates= g.bellman_ford(int(start.get()))
            if has_negative_cycle:
                screen.insert('end',"il existe un cycle negatif.\n")
            else:
                screen.insert('end',"le plus court chemin (Bellman-Ford):\n")
                for v in range(g.V):
                    screen.insert('end',"Noeud "+str(v)+" :"+str(distances[v])+"\n")
                    screen.insert('end',"les noeuds intermidiares:"+str(intermediates[v])+"\n") 
            final.delete(0, tkinter.END)
            start.delete(0, tkinter.END) 
        except  Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n") 
    def prf_():
        try:
            dfs_traversal = g.dfs(int(start.get()))
            screen.insert('end',"le parcours par profendeur:\n")
            for n in dfs_traversal:
                screen.insert('end'," "+str(n))
            final.delete(0, tkinter.END)
            start.delete(0, tkinter.END)
        except Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n")
    def largeur_():
        try:
            bfs_traversal = g.larg(int(start.get()))
            screen.insert('end',"le parcours par largeur:\n")
            for n in bfs_traversal:
                screen.insert('end'," "+str(n))
            final.delete(0, tkinter.END)
            start.delete(0, tkinter.END) 
        except Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n")
    def color_():
        welch_powell_result = g.welch_powell_algorithm()
        screen.insert('end',"Welch-Powell Algorithm result:\n")
        for vertex, color in welch_powell_result.items():
            screen.insert('end',"Vertex:"+str(vertex)+", Color: "+str(color)+"\n")
    A=tkinter.Toplevel()
    A.geometry('520x400')
    arbre=tkinter.Label(A,text="l`arbre couvrantes:")
    krus=tkinter.Button(A,text="krustal",width='10',command=krustal_)
    court=tkinter.Button(A,text="prim",width='10',command=prim_)
    initialisation=tkinter.Label(A,text="les initialisations pour les algorithmes de parcours et de plus cours chemin:")
    fin=tkinter.Label(A,text="point d`arrivee:")
    final=tkinter.Entry(A)
    depart=tkinter.Label(A,text="point de de depart:")
    start=tkinter.Entry(A)
    plus_court_chemin=tkinter.Label(A,text="le plus court chemin:")
    dijks=tkinter.Button(A,text="dijkstra",width='10',command=dijkstra_)
    ford_bell=tkinter.Button(A,text="ford-bellman",width='10',command=ford_bellman_)
    parcours=tkinter.Label(A,text="les algorithmes de parcours:")
    prf=tkinter.Button(A,text="profendeur",width='10',command=prf_)
    largeur=tkinter.Button(A,text="largeur",width='10',command=largeur_)
    flow=tkinter.Label(A,text="le flaux maximal:")
    ford_ful=tkinter.Button(A,text="ford_fulkston",width='10',command=ford_fulkerson_)
    color=tkinter.Label(A,text="coloration")
    colorisation=tkinter.Button(A,text="welch-powell",width='10',command=color_)
    arbre.place(x=10,y=10)
    krus.place(x=50,y=30)
    court.place(x=150,y=30)
    initialisation.place(x=10,y=60)
    depart.place(x=10,y=90)
    start.place(x=150,y=90)
    fin.place(x=300,y=90)
    final.place(x=400,y=90)
    plus_court_chemin.place(x=10,y=120)
    ford_bell.place(x=50,y=150)
    dijks.place(x=150,y=150)
    parcours.place(x=10,y=180)
    largeur.place(x=50,y=210)
    prf.place(x=150,y=210)
    flow.place(x=10,y=250)
    ford_ful.place(x=10,y=280)
    color.place(x=10,y=320)
    colorisation.place(x=10,y=360)
def implimentation():
    def get_entry_value():
        pr = int(pred.get())
        sc = int(succe.get())
        poi = int(poid.get())
        try:
            g.add_edge(pr,sc,poi)
            pred.delete(0, tkinter.END)
            succe.delete(0, tkinter.END)
            poid.delete(0, tkinter.END)
        except Exception as e:
            screen.insert('end',"An error occurred:"+str(e)) 
    def valider():
        im.destroy() 
    def orienter():
        pr = int(pred.get())
        sc = int(succe.get())
        poi = int(poid.get())
        try:
            g.add_edge_directed(pr,sc,poi)
            pred.delete(0, tkinter.END)
            succe.delete(0, tkinter.END)
            poid.delete(0, tkinter.END)
        except Exception as e:
            screen.insert('end',"An error occurred:"+str(e)) 
    im=tkinter.Toplevel()
    im.geometry('350x250')
    predecesseur = tkinter.Label(im, text="predecesseur:")
    pred=tkinter.Entry(im)
    successeur = tkinter.Label(im, text="successeur:")
    succe=tkinter.Entry(im)
    weight=tkinter.Label(im, text="poid:")
    oriente=tkinter.Button(im,text="ajouter arc oriente",command=orienter)
    poid=tkinter.Entry(im)
    button = tkinter.Button(im, text="ajouter", command=get_entry_value)
    confirm = tkinter.Button(im, text="build graph", command=valider)
    predecesseur.place(x=10,y=10)
    pred.place(x=90,y=10)
    successeur.place(x=10,y=50)
    succe.place(x=90,y=50)
    weight.place(x=10,y=100)
    poid.place(x=90,y=100)
    button.place(x=10,y=150)
    confirm.place(x=90,y=150) 
    oriente.place(x=200,y=150)  
def graphe():
    def cree():
        try:
            g.set_V(int(nbr_node.get()))
            nbr_node.delete(0, tkinter.END)
        except  Exception as e:
            screen.insert('end',"An error occurred:"+str(e)+"\n")
    screen.insert('end',"vous avez choisi le traitement des graphes veuiller implimenter votre graphe est appliquer les algo\n")
    GA=tkinter.Toplevel()
    GA.geometry('400x200')
    node = tkinter.Label(GA, text="nbr de node :")
    nbr_node=tkinter.Entry(GA)
    valid=tkinter.Button(GA,text='enregistrer',command=cree)
    algo=tkinter.Button(GA,text='algorithme',command=algorithme)
    impli=tkinter.Button(GA,text='implimenter',command=implimentation)
    node.place(x=0,y=50)
    nbr_node.place(x=100,y=50)
    valid.place(x=50,y=100)
    impli.place(x=150,y=100)
    algo.place(x=250,y=100)
def pro():
    def valid1_():
        entry_text = var1.get()
        entry_list = entry_text.split(',')
        entry_list = [int(item) for item in entry_list]
        x.extend(entry_list)
        var1.delete(0, tkinter.END)
    def valid2_():
        entry_text = var2.get()
        entry_list = entry_text.split(',')
        entry_list = [int(item) for item in entry_list]
        C.append(entry_list)
        var2.delete(0, tkinter.END)
        
    def valid3_():
        entry_text = var3.get()
        entry_list = entry_text.split(',')
        entry_list = [int(item) for item in entry_list]
        b.extend(entry_list)
        var3.delete(0, tkinter.END)
    def simplex_():
        base,valeurs , result =simplex(x,C,b)
        screen.insert('end',"Valeur optimale:"+str(result) +"\n")
        screen.insert('end',"Solution optimale: \n")
        for i in range(len(base)):
            screen.insert('end',str(base[i])+" :"+str(valeurs[i])+"\n")

    screen.insert('end',"vous avez choisi le traitement des ppl veuiller implimenter votre probleme est appliquer les algo\n")
    PO=tkinter.Toplevel()
    PO.geometry('400x300')
    titre1=tkinter.Label(PO,text="donner les coefficient des variables de dicision dans en les separant par ',' :")
    var1=tkinter.Entry(PO)
    valid1=tkinter.Button(PO,text='valider',command=valid1_)
    titre2=tkinter.Label(PO,text="donner les coefficient des contraintes en les separant par ',' :")
    var2=tkinter.Entry(PO)
    valid2=tkinter.Button(PO,text='valider',command=valid2_)
    titre3=tkinter.Label(PO,text="donner les valeurs de b en les separant par ',' :")
    var3=tkinter.Entry(PO)
    valid3=tkinter.Button(PO,text='valider',command=valid3_)
    simplexe=tkinter.Button(PO,text='appliquer simplexe',command=simplex_)
    titre1.place(x=10,y=10)
    var1.place(x=10,y=60)
    valid1.place(x=100,y=60)
    titre2.place(x=10,y=90)
    var2.place(x=10,y=120)
    valid2.place(x=100,y=120)
    titre3.place(x=10,y=150)
    var3.place(x=10,y=180)
    valid3.place(x=100,y=180)
    simplexe.place(x=10,y=250)
def reset_():
    reset_simplexe(x,C,b)
    g.reset()
def sup_():
    screen.delete("1.0", tkinter.END)
mainapp=tkinter.Tk ()
mainapp.title("ppl & graphe")
mainapp.geometry("800x400")
screen=tkinter.Text(mainapp)
poo=tkinter.Button(mainapp,text='ppl',width='10',command=pro)
graph=tkinter.Button(mainapp,text='graphes',width='10',command=graphe)
reini=tkinter.Button(mainapp,text='reset',width='10',command=reset_)
effacer=tkinter.Button(mainapp,text='effacer',width='10',command=sup_)
screen.place(x=150,y=0)
poo.place(x=30,y=70) 
graph.place(x=30,y=20)
reini.place(x=30,y=120)
effacer.place(x=30,y=170)


mainapp.mainloop()