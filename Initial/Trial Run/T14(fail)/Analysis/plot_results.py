import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

# This will be imported in train.py, so we need a forward reference
from Quantum_Core.qneat import Genome 

def genome_to_image(genome) -> np.ndarray:
    """
    Creates a matplotlib figure visualizing a genome's circuit and returns it
    as a NumPy array for TensorBoard logging.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    num_qubits = genome.num_qubits
    
    # Draw qubit lines
    for i in range(num_qubits):
        ax.plot([0, len(genome.genes) + 1], [i, i], color='black', zorder=1)

    # Draw gates
    for i, gene in enumerate(genome.genes):
        qubits = gene.target_qubits
        if gene.gate_type == 'cnot':
            # Draw the vertical line connecting the qubits
            ax.plot([i + 1, i + 1], [qubits[0], qubits[1]], color='blue', zorder=2)
            # Draw the control circle
            ax.plot(i + 1, qubits[0], 'o', color='blue', markersize=8, zorder=3)
            # Draw the target 'X'
            ax.plot(i + 1, qubits[1], 'x', color='blue', markersize=10, markeredgewidth=2, zorder=3)
        else: # Single-qubit gate
            ax.plot(i + 1, qubits[0], 's', color='red', markersize=12, label=gene.gate_type, zorder=3)

    ax.set_yticks(range(num_qubits))
    ax.set_ylabel("Qubits")
    ax.set_xlabel("Gate Step")
    ax.set_title(f"Genome Architecture (Fitness: {genome.fitness:.2f})")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0.5, len(genome.genes) + 0.5)
    ax.set_ylim(-0.5, num_qubits - 0.5)
    
    # --- Definitive Method to Convert Plot to NumPy Array ---
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = np.array(Image.open(buf).convert('RGB'))
    
    plt.close(fig)
    buf.close()
    
    return image