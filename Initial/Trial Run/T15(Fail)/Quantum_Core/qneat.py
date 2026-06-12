import random
import numpy as np
import copy
import pickle
import os
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
from tensorboardX import SummaryWriter
@dataclass
class QNEATOptions:
    """A dataclass to hold all hyperparameters for the QNEAT algorithm."""
    population_size: int = 200  # INCREASED: More genetic diversity
    add_rot_prob: float = 0.3   # INCREASED: Encourage more parametric gates early
    add_cnot_prob: float = 0.1  # INCREASED: Encourage more entanglement
    weight_mutate_prob: float = 0.8
    new_weight_prob: float = 0.1
    weight_mutate_power: float = 0.5
    crossover_rate: float = 0.75
    compatibility_threshold: float = 2.0 # DECREASED: Creates more species, protecting novel mutations
    c1: float = 1.0 # Disjoint coefficient
    c2: float = 1.0 # Excess coefficient
    c3: float = 0.4 # Weight difference coefficient
    
    # Elitism & Survival
    species_elitism: int = 1
    survival_rate: float = 0.2

class Gene:
    """Represents a single quantum gate with a unique innovation number."""
    def __init__(self,
                 innovation_number: int,
                 gate_type: str,
                 target_qubits: Union[int, List[int]],
                 parameters: Optional[List[float]] = None) -> None:
        self.innovation_number = innovation_number
        self.gate_type = gate_type
        self.target_qubits = [target_qubits] if isinstance(target_qubits, int) else target_qubits
        self.parameters = parameters if parameters is not None else []

    def __repr__(self) -> str:
        param_str = f", params={self.parameters}" if self.parameters else ""
        return (f"Gene(in_num={self.innovation_number}, gate={self.gate_type}, "
                f"targets={self.target_qubits}{param_str})")

class Genome:
    """Represents the complete genetic encoding for a Variational Quantum Circuit."""
    def __init__(self,
                 num_qubits: int,
                 options: QNEATOptions,
                 global_innovation_counter: Dict[str, int],
                 global_innovation_dict: Dict[Any, int],
                 genes: Optional[List[Gene]] = None) -> None:
        self.num_qubits = num_qubits
        self.options = options
        self.genes = genes if genes is not None else []
        self.fitness: float = 0.0
        self._innovation_counter = global_innovation_counter
        self._global_innovation_dict = global_innovation_dict

    def add_gene(self, gene: Gene) -> None:
        """Appends a gene to the genome."""
        # In a full NEAT implementation, new genes would be added
        # via a mutation function that handles innovation numbers.
        self.genes.append(gene)

    def __len__(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        return f"Genome(num_qubits={self.num_qubits}, fitness={self.fitness}, genes={self.genes})"

    def _get_new_innovation_number(self, gene_topology_key: Any) -> int:
        """Gets a unique innovation number for a new gene topology."""
        if gene_topology_key in self._global_innovation_dict:
            return self._global_innovation_dict[gene_topology_key]
        else:
            self._innovation_counter['count'] += 1
            new_in_num = self._innovation_counter['count']
            self._global_innovation_dict[gene_topology_key] = new_in_num
            return new_in_num

    def mutate(self) -> None:
        """Applies mutations to the genome."""
        if random.random() < self.options.add_rot_prob:
            self._mutate_add_rot_gate()
        if random.random() < self.options.add_cnot_prob:
            self._mutate_add_cnot_gate()
        self._mutate_parameters()
        
    def _mutate_add_rot_gate(self) -> None:
        """Adds a new rotation gate gene to the genome."""
        target_qubit = random.randint(0, self.num_qubits - 1)
        topology_key = ('rx', target_qubit)
        in_num = self._get_new_innovation_number(topology_key)
        new_gene = Gene(
            innovation_number=in_num,
            gate_type='rx',
            target_qubits=[target_qubit],
            parameters=[random.uniform(-np.pi, np.pi)]
        )
        self.genes.append(new_gene)
        self.genes.sort(key=lambda g: g.innovation_number)

    def _mutate_add_cnot_gate(self) -> None:
        """Adds a new CNOT gate gene to the genome."""
        if self.num_qubits < 2: return
        control, target = random.sample(range(self.num_qubits), 2)
        topology_key = ('cnot', control, target)
        in_num = self._get_new_innovation_number(topology_key)
        new_gene = Gene(
            innovation_number=in_num,
            gate_type='cnot',
            target_qubits=[control, target]
        )
        self.genes.append(new_gene)
        self.genes.sort(key=lambda g: g.innovation_number)

    def _mutate_parameters(self) -> None:
        """Modifies the parameters of existing rotation gates."""
        for gene in self.genes:
            if gene.gate_type in ['rx', 'ry', 'rz']:
                if random.random() < self.options.weight_mutate_prob:
                    if random.random() < self.options.new_weight_prob:
                        gene.parameters[0] = random.uniform(-np.pi, np.pi)
                    else:
                        perturbation = random.uniform(-1, 1) * self.options.weight_mutate_power
                        gene.parameters[0] += perturbation

    @staticmethod
    def crossover(parent1: 'Genome', parent2: 'Genome') -> 'Genome':
        """
        Performs crossover between two parent genomes to create a child.
        Follows the NEAT methodology for aligning and inheriting genes.

        Args:
            parent1: The first parent Genome.
            parent2: The second parent Genome.

        Returns:
            A new child Genome.
        """
        # Designate the more fit parent
        if parent1.fitness > parent2.fitness:
            fitter_parent, less_fit_parent = parent1, parent2
        elif parent2.fitness > parent1.fitness:
            fitter_parent, less_fit_parent = parent2, parent1
        else:
            # If fitness is equal, coin flip
            fitter_parent, less_fit_parent = random.sample([parent1, parent2], 2)

        child_genes: List[Gene] = []
        p1_idx, p2_idx = 0, 0

        while p1_idx < len(fitter_parent.genes) or p2_idx < len(less_fit_parent.genes):
            gene1 = fitter_parent.genes[p1_idx] if p1_idx < len(fitter_parent.genes) else None
            gene2 = less_fit_parent.genes[p2_idx] if p2_idx < len(less_fit_parent.genes) else None

            if gene1 and gene2:
                # --- Both parents have genes to consider ---
                if gene1.innovation_number == gene2.innovation_number:
                    # Matching genes: Inherit randomly from either parent
                    chosen_gene = random.choice([gene1, gene2])
                    child_genes.append(copy.deepcopy(chosen_gene))
                    p1_idx += 1
                    p2_idx += 1
                elif gene1.innovation_number < gene2.innovation_number:
                    # Disjoint gene from the fitter parent: Inherit it
                    child_genes.append(copy.deepcopy(gene1))
                    p1_idx += 1
                else: # gene2.innovation_number < gene1.innovation_number
                    # Disjoint gene from the less fit parent: Do not inherit
                    p2_idx += 1
            
            elif gene1 and not gene2:
                # --- Fitter parent has excess genes ---
                child_genes.append(copy.deepcopy(gene1))
                p1_idx += 1
            
            else:
                # --- Less fit parent has excess genes (or both are done) ---
                break # Stop the loop

        # Create the child genome with the inherited genes
        child_genome = Genome(
            num_qubits=fitter_parent.num_qubits,
            options=fitter_parent.options,
            global_innovation_counter=fitter_parent._innovation_counter,
            global_innovation_dict=fitter_parent._global_innovation_dict,
            genes=child_genes
        )
        return child_genome

    @staticmethod
    def compatibility_distance(genome1: 'Genome', genome2: 'Genome', options: QNEATOptions) -> float:
        """
        Calculates the compatibility distance between two genomes.
        This is the core of the speciation mechanism.
        """
        g1_idx, g2_idx = 0, 0
        disjoint, excess, weight_diff, matching = 0, 0, 0, 0
        
        while g1_idx < len(genome1.genes) or g2_idx < len(genome2.genes):
            gene1 = genome1.genes[g1_idx] if g1_idx < len(genome1.genes) else None
            gene2 = genome2.genes[g2_idx] if g2_idx < len(genome2.genes) else None
            
            if not gene1: # Genome 1 is exhausted
                excess += 1
                g2_idx += 1
                continue
            if not gene2: # Genome 2 is exhausted
                excess += 1
                g1_idx += 1
                continue

            in_num1 = gene1.innovation_number
            in_num2 = gene2.innovation_number

            if in_num1 == in_num2: # Matching genes
                if gene1.parameters and gene2.parameters:
                    weight_diff += abs(gene1.parameters[0] - gene2.parameters[0])
                matching += 1
                g1_idx += 1
                g2_idx += 1
            elif in_num1 < in_num2: # Disjoint gene in genome1
                disjoint += 1
                g1_idx += 1
            else: # Disjoint gene in genome2
                disjoint += 1
                g2_idx += 1
        
        N = max(len(genome1.genes), len(genome2.genes))
        if N == 0: return 0.0

        distance = (options.c1 * excess / N) + \
                   (options.c2 * disjoint / N) + \
                   (options.c3 * (weight_diff / matching if matching > 0 else 0))
        
        return distance
    
class Species:
    """Manages a collection of similar genomes."""
    def __init__(self, first_member: Genome):
        self.representative: Genome = first_member
        self.members: List[Genome] = [first_member]
        self.fitness: float = 0.0
        self.stagnation: int = 0

    def adjust_fitness(self) -> None:
        """Calculates the adjusted fitness for the species (fitness sharing)."""
        if not self.members: return
        self.fitness = sum(member.fitness for member in self.members) / len(self.members)
    
    def add_member(self, member: Genome) -> None:
        self.members.append(member)

class Population:
    """Manages the entire population of genomes and the evolutionary process."""
    def __init__(self, num_qubits: int, options: QNEATOptions):
        self.num_qubits = num_qubits
        self.options = options
        self.global_innovation_counter = {'count': 0}
        self.global_innovation_dict = {}
        
        self.population: List[Genome] = self._create_initial_population()
        self.species: List[Species] = []
        self.generation: int = 0
        self.writer = SummaryWriter()

    def __getstate__(self) -> Dict:
        """Tell pickle what to save. Exclude the non-serializable writer."""
        state = self.__dict__.copy()
        del state['writer']
        return state

    def __setstate__(self, state: Dict) -> None:
        """Tell pickle how to load. Restore attributes and re-create the writer."""
        self.__dict__.update(state)
        self.writer = SummaryWriter()

    def _create_initial_population(self) -> List[Genome]:
        pop = []
        for _ in range(self.options.population_size):
            genome = Genome(self.num_qubits, self.options, 
                            self.global_innovation_counter, self.global_innovation_dict)
            # You might start with a minimal topology or random small topologies
            pop.append(genome)
        return pop

    def run_evolutionary_cycle(self) -> None:
        """Executes one full generation of the QNEAT algorithm."""
        # 1. Speciate
        for species in self.species:
            species.members = [] # Clear members from previous generation

        for genome in self.population:
            found_species = False
            for species in self.species:
                if Genome.compatibility_distance(genome, species.representative, self.options) < self.options.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break
            if not found_species:
                self.species.append(Species(genome))
        
        self.species = [s for s in self.species if s.members] # Remove empty species

        # 2. Calculate adjusted fitness for each species
        total_adjusted_fitness = 0
        for species in self.species:
            species.adjust_fitness()
            total_adjusted_fitness += species.fitness
        
        # 3. Create the next generation
        next_gen_population = []
        for species in self.species:
            # Determine number of offspring for this species
            if total_adjusted_fitness > 0:
                num_offspring = round((species.fitness / total_adjusted_fitness) * self.options.population_size)
            else:
                num_offspring = round(self.options.population_size / len(self.species))

            # Sort members by fitness and cull the weak
            species.members.sort(key=lambda g: g.fitness, reverse=True)
            num_survivors = max(self.options.species_elitism, round(len(species.members) * self.options.survival_rate))
            survivors = species.members[:num_survivors]

            if not survivors: continue

            # Elitism: carry over the best from the species
            if self.options.species_elitism > 0:
                next_gen_population.append(copy.deepcopy(survivors[0]))
                num_offspring -= 1

            # Generate offspring via crossover and mutation
            for _ in range(num_offspring):
                parent1 = random.choice(survivors)
                if random.random() < self.options.crossover_rate and len(survivors) > 1:
                    parent2 = random.choice(survivors)
                    child = Genome.crossover(parent1, parent2)
                else: # Asexual reproduction
                    child = copy.deepcopy(parent1)
                
                child.mutate()
                next_gen_population.append(child)

        self.population = next_gen_population
         # --- Logging ---
        # Add a check to prevent division by zero if population is empty
        if not self.population:
            print("Population extinct. Ending evolution.")
            return

        best_genome = max(self.population, key=lambda g: g.fitness)
        self.writer.add_scalar('Fitness/Max', best_genome.fitness, self.generation)
        
        avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
        self.writer.add_scalar('Fitness/Average', avg_fitness, self.generation)
        
        self.writer.add_scalar('Population/Num_Species', len(self.species), self.generation)
        self.writer.flush()
        self.generation += 1
        
    def close_writer(self):
        self.writer.close()

    def save_checkpoint(self, path: str):
        """Saves the entire Population object to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"--- Saved checkpoint to {path} ---")

    @staticmethod
    def load_checkpoint(path: str) -> 'Population':
        """Loads a Population object from a checkpoint file."""
        with open(path, 'rb') as f:
            population = pickle.load(f)
        print(f"--- Loaded checkpoint from {path} ---")
        return population