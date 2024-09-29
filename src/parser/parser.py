from openbabel import openbabel
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import pandas as pd


class Molecule:
    def __init__(self, file_path='', mapping=None):
        self.file_path=file_path
        self.content = ""
        self.labels = []
        self.coordinates = []
        self.distances = None  # Attribute to store the distance matrix or the custom distances
        self.sequence_A = []
        self.sequence_B = []
        self.mapping = mapping

        
        # Read and populate self.content
        self._read_file()

        # Parse the content to extract relevant data
        self._parse_file()


        self._compute_distances()
        # self._compute_sequence_distances()

    def _read_file(self, output_file=None):
        # Create an instance of the OBConversion class
        
        if self.file_path[-3:]=="cml":
        
            obConversion = openbabel.OBConversion()

            out="pdb"

            # Set the input format to CML and output format to PDB
            obConversion.SetInAndOutFormats("cml", f"{out}")

            # Create an OBMol object to hold the molecule data
            mol = openbabel.OBMol()

            # Read the CML file
            obConversion.ReadFile(mol, f"{self.file_path}")

            if output_file is not None:
                # obConversion.WriteFile(mol, "output.pdb")
                obConversion.WriteFile(mol, f"{output_file}")

            # Write the molecule data to a PDB file
            self.content = obConversion.WriteString(mol)
        elif self.file_path[-3:]=="pdb":
            with open(self.file_path, 'r') as file:
               self.content = file.read()
    
        else:
            print(f'File ({self.file_path[-3:]}) must be cml or pdb. If error persists, check file contents.')



    def _parse_file(self):
            if self.file_path[-3:]=="cml":
                for index, line in enumerate(self.content.splitlines()):
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        parts = line.split()                    
                        sequence = parts[4]
                        x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
                        
                        self.coordinates.append([x, y, z])

                        
                        if self.mapping is not None:
                            label = parts[2]
                            label = self.mapping.get(label, label)
                            self.labels.append(label)
                        else:
                            label = parts[-1]
                            self.labels.append(label)
                        
                        # Store coordinates and indices for sequence A and sequence B
                        if sequence == 'A':
                            self.sequence_A.append((label, [x, y, z]))
                            # self.indices_A.append(index)
                        elif sequence == 'B':
                            self.sequence_B.append((label, [x, y, z]))
                            # self.indices_B.append(index)
            elif self.file_path[-3:]=="pdb":
                for index, line in enumerate(self.content.splitlines()):
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        parts = line.split()                    

                        x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                        
                        self.coordinates.append([x, y, z])

                        if self.mapping is not None:
                            label = parts[2]
                            label = self.mapping.get(label, label)
                            self.labels.append(label)
                        else:
                            label = parts[-1]
                            self.labels.append(label)
                        
                if 'TER' in self.content:
                    strands = self.content.split('TER')

                    # Ensure there are exactly two strands after the split
                    if len(strands) > 2:
                        # print(f"{len(strands)} TER found")
                        # Add back the 'TER' line to the first strand and save
                        strand_A = strands[0].strip() + '\nTER\nEND\n'
                        for line in strand_A.splitlines():
                            if line.startswith("ATOM") or line.startswith("HETATM"):
                                parts = line.split()                    

                                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                                
                                self.coordinates.append([x, y, z])

                                if self.mapping is not None:
                                    label = parts[2]
                                    label = self.mapping.get(label, label)
                                    self.labels.append(label)
                                else:
                                    label = parts[-1]
                                    self.labels.append(label)
                                
                                self.sequence_A.append((label, [x, y, z]))
                            
                        # with open(output_file_A, 'w') as file_A:
                        #     file_A.write(strand_A)
                        
                        # The second strand already contains the END line
                        strand_B = strands[1].strip() + '\nEND\n'
                        for line in strand_B.splitlines():
                            if line.startswith("ATOM") or line.startswith("HETATM"):
                                parts = line.split()                    

                                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                                
                                self.coordinates.append([x, y, z])

                                if self.mapping is not None:
                                    label = parts[2]
                                    label = self.mapping.get(label, label)
                                    self.labels.append(label)
                                else:
                                    label = parts[-1]
                                    self.labels.append(label)
                                
                                self.sequence_B.append((label, [x, y, z]))
                        # with open(output_file_B, 'w') as file_B:
                        #     file_B.write(strand_B)
                else:
                    print("No TER line found in the PDB file.")

            else:
                print(f'File ({self.file_path[-3:]}) must be cml or pdb. If error persists, check file contents.')
                

            ### Temporary for Development, substitute with a dictionary of atom labels and maping instrucitons
            self.unique_atom_labels =  set(self.labels)
            
            # Convert coordinates to a numpy array
            self.coordinates = np.array(self.coordinates)

    # def _parse_file(self, file_path):
    #     with open(file_path, 'r') as file:
    #         for index, line in enumerate(file):
    #             if line.startswith("ATOM") or line.startswith("HETATM"):
    #                 parts = line.split()
    #                 label = parts[-1]
    #                 sequence = parts[4]
    #                 x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
    #                 self.labels.append(label)
    #                 self.coordinates.append([x, y, z])
    #                 self.indices.append(index)
    #                 self.duplex.append((label, [x, y, z]))
    #                 # Store coordinates and indices for sequence A and sequence B
    #                 if sequence == 'A':
    #                     self.sequence_A.append((label, [x, y, z]))
    #                     self.indices_A.append(index)
    #                 elif sequence == 'B':
    #                     self.sequence_B.append((label, [x, y, z]))
    #                     self.indices_B.append(index)

    #     self.unique_atom_labels =  set(self.labels)
        
    #     # Convert coordinates to a numpy array
    #     self.coordinates = np.array(self.coordinates)

    def _compute_distances(self):
        # Compute pairwise Euclidean distances and store them in the distances attribute
        distances = pdist(self.coordinates, metric='euclidean')
        self.distances = squareform(distances)

    def plot_heatmap(self, custom_distances=None, lower_cutoff=None, upper_cutoff=None):
        # Apply cutoff criteria if specified
        if custom_distances == None:
            filtered_distances = np.copy(self.distances)
        else:
            filtered_distances = self.sequence_distances
        
        if lower_cutoff is not None:
            filtered_distances[filtered_distances < lower_cutoff] = np.nan
        
        if upper_cutoff is not None:
            filtered_distances[filtered_distances > upper_cutoff] = np.nan
        
        # Plot the heatmap using the filtered distance matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(filtered_distances, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Distance')
        if custom_distances == None:
            plt.xticks(ticks=np.arange(len(self.labels)), labels=self.labels, rotation=90)
            plt.yticks(ticks=np.arange(len(self.labels)), labels=self.labels)
        else:
            plt.xticks(ticks=np.arange(len(self.sequence_A)), labels=[label for label, _ in self.sequence_A], rotation=90)
            plt.yticks(ticks=np.arange(len(self.sequence_B)), labels= [label for label, _ in self.sequence_B])
        plt.title('Pairwise Distance Heatmap with Cutoff')
        plt.show()

    def count_occurrences(self, atoms = None, cutoff_distance=None, crit=None):
        # Initialize a dictionary to store the occurrence counts for each atom pair
        occurrence_counts = {}
        if atoms == None:
            atoms = self.unique_atom_labels
            atoms = self.mapping.values()
      
        # Iterate over all unique pairs of atom labels
        for label_A in atoms:
            for label_B in atoms:

                if crit == None:
                    
                    # Extract coordinates for atoms in Sequence A and Sequence B with the specified labels
                    coords_A = np.array([coord for label, coord in self.sequence_A if label == label_A])
                    coords_B = np.array([coord for label, coord in self.sequence_B if label == label_B])

                else:
                    duplex = self.sequence_A + self.sequence_B

                    # Extract coordinates for atoms in Sequence A and Sequence B with the specified labels
                    coords_A = np.array([coord for label, coord in duplex if label == label_A])
                    coords_B = np.array([coord for label, coord in duplex if label == label_B])

                # Compute pairwise distances between sequence A and sequence B for the given atom pair
                # print(len(coords_A),len(coords_B))
                if len(coords_A) !=0 and len(coords_B) !=0:
                    sequence_distances = cdist(coords_A, coords_B, metric='euclidean')

                    # Count occurrences where distance is less than or equal to the cutoff distance
                    if cutoff_distance == None:
                        count = np.sum(sequence_distances <= sequence_distances.max())
                    else:
                        count = np.sum(sequence_distances <= cutoff_distance)
                else:
                    count = 0

                # Store the count in the dictionary with the pair of labels as the key
                occurrence_counts[(label_A, label_B)] = count

        return occurrence_counts


if "__name__" == "__main__" :
    print('hi')