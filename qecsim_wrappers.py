"""
Some wrappers around the default qecsim python API, mainly to hide any binary symplectic
representation jargon, make the printing nicer, and most switch X and Z everywhere. 
"""

import numpy as np
from functools import reduce
from IPython.display import HTML
from qecsim.models.toric import ToricCode
from qecsim.models.generic import PhaseFlipErrorModel
from qecsim import paulitools as pt

def print_code(code: ToricCode):
    print(code.ascii_art())

# add some color to Pauli errors in HTML output, and switch basis
def html_pauli(pauli):
    """
    Adapted from https://github.com/qecsim/qecsim/blob/master/demos/qsu.ipynb
    """
    text = str(pauli)
    # the big switcheroo
    text = text.replace('X', 'z')
    text = text.replace('Z', 'x')
    # then color like nothing happened
    text = text.replace('z', '<span style="color:red; font-weight:bold">Z</span>')
    text = text.replace('Y', '<span style="color:magenta; font-weight:bold">Y</span>')
    text = text.replace('x', '<span style="color:blue; font-weight:bold">X</span>')
    display(HTML('<div class="highlight"><pre style="line-height:1!important;">{}</pre></div>'.format(text)))

def pauli_art(code: ToricCode, pauli: np.ndarray):
    return f"{code.new_pauli(pauli)}"

def syndrome_art(code: ToricCode, syndrome: np.ndarray):
    return f"{code.ascii_art(syndrome)}"

def print_pauli(code: ToricCode, pauli: np.ndarray):
    html_pauli(f"{pauli_art(code, pauli)}")

def print_syndrome(code: ToricCode, syndrome: np.ndarray):
    html_pauli(f"{syndrome_art(code, syndrome)}")

def compute_syndrome(code: ToricCode, error: np.ndarray):
    return pt.bsp(error, code.stabilizers.T)

def pauli_product(pauli1: np.ndarray, pauli2: np.ndarray):
    return pauli1 ^ pauli2

def logical_commutations(code: ToricCode, error: np.ndarray):
    return pt.bsp(error, code.logicals.T)

def effective_logical(code: ToricCode, error: np.ndarray):
    # check if it commutes with all the logicals
    assert np.sum(compute_syndrome(code, error)) == 0, "Error does not commute with stabilizers"

    # initialize lables for the qecsim.models.toric.ToricCode logical operators
    log_labels = np.array(["Z1", "Z2", "X1", "X2"]) # switcheroo

    # find logicals the error anticommutes with
    log_comms = logical_commutations(code, error)

    # then circshift to transform this to the other logical in the pair
    log_ops = np.roll(log_comms, 2) == 1 
    if not np.any(log_ops):
        op = np.zeros(error.shape)
        label = "I"
    else:
        op = reduce(lambda x, y: x ^ y, code.logicals[log_ops])
        label = " * ".join(log_labels[log_ops])

    return op, label

def bit_flip_model():
    return PhaseFlipErrorModel() # switcheroo