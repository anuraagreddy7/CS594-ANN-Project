import time
import math
import random
import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt

class Quadtree:
   def __init__(self, boundary, max_points=4, depth=0):
       self.boundary = boundary  # (center, half-widths)
       self.points = []
       self.children = []
       self.depth = depth
       self.max_points = max_points
       self.n = 0

   def subdivide(self):
       center, half_widths = self.boundary
       d = len(center)
       offsets = list(product([-1, 1], repeat=d))
       new_half_widths = [hw/2 for hw in half_widths]
       for offset in offsets:
           new_center = [center[i] + offset[i]*new_half_widths[i] for i in range(d)]
           self.children.append(Quadtree((new_center, new_half_widths), self.max_points, self.depth + 1))

   def insert(self, point):
       if not self.contains(point):
           return

       if len(self.points) < self.max_points and not self.children:
           self.points.append(point)
           self.n += 1
       else:
           if not self.children:
               self.subdivide()
               for p in self.points:
                   self._insert_to_children(p)
               self.points = []
           self._insert_to_children(point)
           self.n += 1

   def _insert_to_children(self, point):
       for child in self.children:
           if child.contains(point):
               child.insert(point)
               return

   def contains(self, point):
       center, half_widths = self.boundary
       return all(abs(point[i] - center[i]) <= half_widths[i] for i in range(len(center)))

   def height(self):
       if not self.children:
           return 1
       return 1 + max(child.height() for child in self.children)

   def query_ann(self, query_point, epsilon):
       def distance(p1, p2):
           return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

       def get_box_distance(node, point):
           center, half_widths = node.boundary
           d = len(center)
           dist = 0
           for i in range(d):
               dim_dist = abs(point[i] - center[i]) - half_widths[i]
               if dim_dist > 0:
                   dist += dim_dist ** 2
           return math.sqrt(dist)

       def traverse(node):
           if not node.children and node.points:
               points_with_dist = [(p, distance(p, query_point)) for p in node.points]
               min_dist = min(d for _, d in points_with_dist)
               acceptable_points = [p for p, d in points_with_dist 
                                 if d <= (1 + epsilon) * min_dist]
               return random.choice(acceptable_points) if acceptable_points else None

           if not node.children:
               return None

           children_dist = [(child, get_box_distance(child, query_point)) 
                         for child in node.children]
           children_dist.sort(key=lambda x: x[1])

           best_point = None
           best_dist = float('inf')

           for child, min_possible_dist in children_dist:
               if best_dist < float('inf') and min_possible_dist > (1 + epsilon) * best_dist:
                   continue
                   
               point = traverse(child)
               if point:
                   dist = distance(point, query_point)
                   if (best_point is None or 
                       dist < best_dist or 
                       (dist <= (1 + epsilon) * best_dist and random.random() < 0.5)):
                       best_point = point
                       best_dist = dist

           return best_point

       return traverse(self)
   
   def extend_to_4d(self, point):
        """Extend 2D point to 4D space"""
        if len(point) == 2:
            return point + (0.0, 0.0)
        return point

   def delete(self, point):
       def find_and_delete(node):
           if not node.children:  # Leaf node
               if point in node.points:
                   node.points.remove(point)
                   node.n -= 1
                   return True
               return False
               
           for child in node.children:
               if child.contains(point):
                   if find_and_delete(child):
                       node.n -= 1
                       return True
           return False
           
       if self.n <= 0:
           return False
           
       deleted = find_and_delete(self)
       
       if deleted and self.n <= len(self.get_all_points()) / 2:
           self.reconstruct()
       
       return deleted

   def get_all_points(self):
       points = []
       if not self.children:
           return self.points.copy()
       for child in self.children:
           points.extend(child.get_all_points())
       return points

   def reconstruct(self):
       points = self.get_all_points()
       self.__init__(self.boundary, self.max_points)
       for point in points:
           self.insert(point)

   def delete_points_in_box(self, min_coords, max_coords):
       points_to_delete = []
       deletion_times = []
       
       def points_in_box(node):
           if not node.children:
               return [p for p in node.points 
                      if all(min_coords[i] <= p[i] <= max_coords[i] 
                      for i in range(len(p)))]
           
           box_points = []
           for child in node.children:
               box_points.extend(points_in_box(child))
           return box_points
       
       points_to_delete = points_in_box(self)
       
       for point in points_to_delete:
           start_time = time.time()
           self.delete(point)
           deletion_times.append(time.time() - start_time)
       
       return np.mean(deletion_times) if deletion_times else 0
   
   def static_to_dynamic_insert(self, point):
    """Implements static-to-dynamic transformation as per the article"""
    if self.n == 0:
        self.insert(point)
        return

    # If n is a power of 2, rebuild completely
    if (self.n & (self.n - 1)) == 0:  # Check if n is power of 2
        points = self.get_all_points()
        points.append(point)
        
        # Create new tree with expanded boundary if needed
        min_coords = [min(p[i] for p in points) for i in range(len(point))]
        max_coords = [max(p[i] for p in points) for i in range(len(point))]
        center = [(min_coords[i] + max_coords[i])/2 for i in range(len(min_coords))]
        half_widths = [(max_coords[i] - min_coords[i])/2 for i in range(len(min_coords))]
        
        self.__init__((center, half_widths), self.max_points)
        for p in points:
            self.insert(p)
    else:
        # Just insert the point
        self.insert(point)

def read_dataset(file_path):
   with open(file_path, 'r') as f:
       lines = f.readlines()
   d = int(lines[0].strip())
   points = [tuple(map(float, line.strip().split(','))) for line in lines[1:]]
   return d, points

def compute_spread(points):
   distances = [math.dist(p, q) for p, q in combinations(points, 2)]
   return max(distances) / min(d for d in distances if d > 0)

def compute_bounding_box(points):
   min_coords = [min(p[i] for p in points) for i in range(len(points[0]))]
   max_coords = [max(p[i] for p in points) for i in range(len(points[0]))]
   center = [(min_coords[i] + max_coords[i])/2 for i in range(len(min_coords))]
   half_widths = [(max_coords[i] - min_coords[i])/2 for i in range(len(min_coords))]
   return center, half_widths

def run_random_queries(quadtree, box_min, box_max, num_queries=1000, epsilon=0.1):
   queries = [(random.uniform(box_min[0], box_max[0]), 
              random.uniform(box_min[1], box_max[1])) 
             for _ in range(num_queries)]
   times = []
   distances = []
   
   def euclidean_distance(p1, p2):
       return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
   
   for query in queries:
       start_time = time.time()
       result = quadtree.query_ann(query, epsilon)
       query_time = time.time() - start_time
       
       if result:
           dist = euclidean_distance(query, result)
           distances.append(dist)
           times.append(query_time)
           
   return np.mean(times) if times else 0, np.mean(distances) if distances else float('inf')

def run_all_experiments():
   print("=== Problem 1: Quadtree Construction ===")
   file_path = "dataset.txt"
   d, points = read_dataset(file_path)
   boundary = compute_bounding_box(points)
   
   start_time = time.time()
   quadtree = Quadtree(boundary)
   for point in points:
       quadtree.insert(point)
   build_time = time.time() - start_time
   
   spread = compute_spread(points)
   tree_height = quadtree.height()
   
   print(f"Construction Time: {build_time:.6f} seconds")
   print(f"Spread: {spread:.6f}")
   print(f"Tree Height: {tree_height}")
   print("Relation between spread and height: The height of the tree is approximately O(log Δ), where Δ is the spread.")
   print(f"Observed height / log2(spread): {tree_height/math.log2(spread):.6f}")

   print("\n=== Problem 2: ANN Queries ===")
   queries = [(500, 500), (1000, 1000), (30, 950), (0, 1020)]
   epsilons = [0.05, 0.1, 0.15, 0.2, 0.25]
   
   def euclidean_distance(p1, p2):
       return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

   for query in queries:
       distances = []
       for epsilon in epsilons:
           trial_distances = []
           for _ in range(10):  # 10 trials per epsilon
               result = quadtree.query_ann(query, epsilon)
               if result:
                   dist = euclidean_distance(query, result)
                   trial_distances.append(dist)
           avg_dist = np.mean(trial_distances) if trial_distances else float('inf')
           distances.append(avg_dist)
       
       plt.figure(figsize=(10, 6))
       plt.plot(epsilons, distances, 'bo-', linewidth=2)
       plt.xlabel('ε')
       plt.ylabel('Average Distance')
       plt.title(f'Query point {query}')
       plt.grid(True)
       plt.savefig(f'query_{query[0]}_{query[1]}.png')
       plt.close()
       print(f"Query {query}: {distances}")

   avg_time_1, avg_dist_1 = run_random_queries(quadtree, (0,0), (1000,1000))
   print(f"\nBox [0,1000] x [0,1000]:")
   print(f"Average query time: {avg_time_1:.6f} seconds")
   print(f"Average distance: {avg_dist_1:.6f}")

   avg_time_2, avg_dist_2 = run_random_queries(quadtree, (1000,1000), (1500,1500))
   print(f"\nBox [1000,1500] x [1000,1500]:")
   print(f"Average query time: {avg_time_2:.6f} seconds")
   print(f"Average distance: {avg_dist_2:.6f}")

   print("\n=== Problem 3: Deletion Experiments ===")
   print("\nDeleting points in [450,550] × [450,550]")
   avg_time_1 = quadtree.delete_points_in_box([450,450], [550,550])
   print(f"Average deletion time: {avg_time_1:.6f} seconds")
   
   q0 = (500, 500)
   distances_q0 = []
   for epsilon in epsilons:
       trial_distances = []
       for _ in range(10):
           result = quadtree.query_ann(q0, epsilon)
           if result:
               dist = euclidean_distance(q0, result)
               trial_distances.append(dist)
       avg_dist = np.mean(trial_distances) if trial_distances else float('inf')
       distances_q0.append(avg_dist)
   
   plt.figure(figsize=(10, 6))
   plt.plot(epsilons, distances_q0, 'bo-', linewidth=2)
   plt.xlabel('ε')
   plt.ylabel('Average Distance')
   plt.title('Query point (500,500) after first deletion')
   plt.grid(True)
   plt.savefig('q0_after_deletion.png')
   plt.close()
   
   print("\nDeleting points in [900,1000] × [900,1000]")
   avg_time_2 = quadtree.delete_points_in_box([900,900], [1000,1000])
   print(f"Average deletion time: {avg_time_2:.6f} seconds")
   
   q1 = (1000, 1000)
   distances_q1 = []
   for epsilon in epsilons:
       trial_distances = []
       for _ in range(10):
           result = quadtree.query_ann(q1, epsilon)
           if result:
               dist = euclidean_distance(q1, result)
               trial_distances.append(dist)
       avg_dist = np.mean(trial_distances) if trial_distances else float('inf')
       distances_q1.append(avg_dist)
   
   plt.figure(figsize=(10, 6))
   plt.plot(epsilons, distances_q1, 'bo-', linewidth=2)
   plt.xlabel('ε')
   plt.ylabel('Average Distance')
   plt.title('Query point (1000,1000) after second deletion')
   plt.grid(True)
   plt.savefig('q1_after_deletion.png')
   plt.close()
   
   initial_points = len(quadtree.get_all_points())
   while quadtree.n > initial_points / 2:
       points = quadtree.get_all_points()
       if not points:
           break
       point = random.choice(points)
       quadtree.delete(point)
   
   deletion_times = []
   for _ in range(1000):
       points = quadtree.get_all_points()
       if not points:
           break
       point = random.choice(points)
       start_time = time.time()
       quadtree.delete(point)
       deletion_times.append(time.time() - start_time)
   
   avg_time_3 = np.mean(deletion_times) if deletion_times else 0
   print(f"\nAverage deletion time after reconstruction: {avg_time_3:.6f} seconds")
   
   epsilon = 0.1
   avg_time_4, avg_dist_4 = run_random_queries(quadtree, (0,0), (1000,1000))
   print(f"\nRandom queries [0,1000] × [0,1000] after deletions:")
   print(f"Average query time: {avg_time_4:.6f} seconds")
   print(f"Average distance: {avg_dist_4:.6f}")
   
   avg_time_5, avg_dist_5 = run_random_queries(quadtree, (1000,1000), (1500,1500))
   print(f"\nRandom queries [1000,1500] × [1000,1500] after deletions:")
   print(f"Average query time: {avg_time_5:.6f} seconds")
   print(f"Average distance: {avg_dist_5:.6f}")
   
   print(f"\nPoints remaining in tree: {quadtree.n}")

def run_insertion_experiments():
    print("\n=== Problem 4: Insertion Experiments ===")
    
    # Initial setup
    boundary = ((500, 500), (500, 500))
    quadtree = Quadtree(boundary)
    
    # Part 1: Insert 6000 points and test
    print("\nInserting 6000 points in [0,1000]×[0,1000]")
    initial_time = time.time()
    for _ in range(6000):
        point = (random.uniform(0, 1000), random.uniform(0, 1000))
        quadtree.static_to_dynamic_insert(point)
    print(f"Insertion time: {time.time() - initial_time:.6f} seconds")

    # Test with different epsilons
    queries = [(500, 500), (1000, 1000), (30, 950), (0, 1020)]
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    # Run queries and create plots
    for query in queries:
        distances = []
        for epsilon in epsilons:
            trial_distances = []
            for _ in range(10):
                result = quadtree.query_ann(query, epsilon)
                if result:
                    dist = euclidean_distance(query, result)
                    trial_distances.append(dist)
            avg_dist = np.mean(trial_distances) if trial_distances else float('inf')
            distances.append(avg_dist)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, distances, 'bo-', linewidth=2)
        plt.xlabel('ε')
        plt.ylabel('Average Distance')
        plt.title(f'Query point {query} after insertions')
        plt.grid(True)
        plt.savefig(f'query_{query[0]}_{query[1]}_p4.png')
        plt.close()
        print(f"Query {query}: {distances}")

    # Random queries in [0,1000]×[0,1000]
    print("\nRunning random queries in [0,1000]×[0,1000]")
    avg_time_1, avg_dist_1 = run_random_queries(quadtree, (0,0), (1000,1000))
    print(f"Average query time: {avg_time_1:.6f} seconds")
    print(f"Average distance: {avg_dist_1:.6f}")

    # Insert 2000 points in larger space
    print("\nInserting 2000 points in [1000,2000]×[1000,2000]")
    for _ in range(2000):
        point = (random.uniform(1000, 2000), random.uniform(1000, 2000))
        quadtree.static_to_dynamic_insert(point)

    # 4D queries
    print("\nRunning 4D queries in [1000,2000]⁴")
    times_4d = []
    distances_4d = []
    epsilon = 0.1
    
    for _ in range(1000):
        # Generate 4D query point
        query_4d = tuple(random.uniform(1000, 2000) for _ in range(4))
        
        start_time = time.time()
        result = quadtree.query_ann(query_4d[:2], epsilon)  # Use first 2 dimensions
        query_time = time.time() - start_time
        
        if result:
            # Extend result to 4D for comparison
            result_4d = quadtree.extend_to_4d(result)
            dist = euclidean_distance(query_4d, result_4d)
            distances_4d.append(dist)
            times_4d.append(query_time)
    
    avg_time_4d = np.mean(times_4d) if times_4d else 0
    avg_dist_4d = np.mean(distances_4d) if distances_4d else float('inf')
    
    print("\n4D Query Results:")
    print(f"Average query time: {avg_time_4d:.6f} seconds")
    print(f"Average distance: {avg_dist_4d:.6f}")

    # Compare with original space
    print("\nComparison with original space:")
    avg_time_final, avg_dist_final = run_random_queries(quadtree, (0,0), (1000,1000))
    print(f"[0,1000]×[0,1000] after all insertions:")
    print(f"Average query time: {avg_time_final:.6f} seconds")
    print(f"Average distance: {avg_dist_final:.6f}")

# Main execution block
if __name__ == "__main__":
    run_all_experiments()
    run_insertion_experiments()
    print("\n==== All Experiments Complete ====")
