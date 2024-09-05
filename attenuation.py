import json
import pandas as pd
import numpy as np

def calculate_compound_mass_attenuation_coefficients(compound_dict, required_energy='1.5'):
    """
    Calculates the mass attenuation coefficient of given a compound.

    Args:
        compound_dict (dict): A dictionary where keys are element symbols and values are their atomic masses,
        weight fractions and elemental mass attenuation coefficients.
        required_energy (float): The energy for which the mass attenuation coefficient is calculated. In MeV.

    Returns:
        float: The mass attenuation coefficient of the compound.
    """
    with open('elements.json', 'r') as json_file:
        element_data = json.load(json_file)
    for element_symbol in compound_dict.keys():
        for element in element_data:
            if element.get('symbol') == element_symbol:
                energy_coeffs = dict(element.get('mass_attenuation_coefficients', {}).items())
                keys = np.array([float(k) for k in energy_coeffs.keys()])
                if required_energy in keys:
                    compound_dict[element_symbol]['massAtt'] = energy_coeffs[required_energy]
                    
                else:
                    closest_energy = keys[np.abs(keys - float(required_energy)).argmin()]
                    compound_dict[element_symbol]['massAtt'] = energy_coeffs[str(closest_energy)]

    compound_mass_attenuation_coeff = float(sum(df['weight_fraction'] * df['massAtt'] for df in compound_dict.values()).iloc[0]) / 10
    print(f'The mass attenuation coefficient of the compound is {compound_mass_attenuation_coeff:.7f} [m^2/kg] for the energy of {required_energy} [MeV]')
    
    return compound_mass_attenuation_coeff

def calculate_weight_fractions(compound_dict):
    """
    Calculates the weight fractions of elements in a compound.

    Args:
        compound_dict (dict): A dictionary where keys are element symbols and values are atomic masses.

    Returns:
        dict: A dictionary containing the weight fractions of each element.
    """
    total_mass = sum(df['ratio'] * df['atomicMass'] for df in compound_dict.values())
    compound_dict = {k: df.assign(weight_fraction=(df['ratio'] * df['atomicMass']) / total_mass) for k, df in compound_dict.items()}

    return compound_dict

def get_atomic_mass(elements):
    """
    Retrieves the atomic mass of an element or elements of a compund based on the symbol(s) from a JSON file.

    Args:
        elements (str or dict): The element symbol (e.g., 'H', 'O').

    Returns:
        float: Atomic mass(es) of the element (in g/mol), or None.
    """
    if type(elements) is dict:
        element_symbols = list(elements.keys())  # Split the formula string by spaces
    
        with open('elements.json', 'r') as json_file:
            element_data = json.load(json_file)
            for symbol in element_symbols:
                for element in element_data:
                    if element.get('symbol') == symbol:
                        elements[symbol]['atomicMass'] = float(element.get('atomicMass')[:-3])
                    else: continue
        return elements
    
    elif type(elements) is str:
        with open('elements.json', 'r') as json_file:
            element_data = json.load(json_file)
        for element in element_data:
            if element.get('symbol') == elements:
                print(f"Atomic masses of the given element is {element.get('atomicMass')}")
                return float(element.get('atomicMass')[:-3])
            else: continue
        
    else:
        print('The argument of the function must be a dictionary or a string')
        return None

def extract_elements_and_ratios(compound_formula):
    """
    Extracts the symbol and the ratio of the elements from the given compound formula.

    Args:
        compound_formula (str): The formula of a compound with spaces between each element.
        Example: for 'H$_2$O' write 'H2 O' or for 'Cu$_3$(PO$_4$)$_2$' write 'Cu3 P2 O8'

    Returns:
        dict: A dictionary whose element symbols as keys, and ratios in a dataframe as values.
    """
    elements = compound_formula.split()  # Split the formula string by spaces
    element_symbol = {}
    for element in elements:
        # Extract the element symbol and count (if present)
        symbol = ""
        count = 1
        for char in element:
            if char.isalpha():
                symbol += char
            elif char.isdigit():
                count = int(char)
            else:
                # Handle unexpected characters (e.g., spaces)
                pass
        # Add the element symbol and count to the dictionary
        element_symbol[symbol] = pd.DataFrame({'ratio':[count]})
    return element_symbol

def prompt_user_for_formula():
        """
        Retrieves the formula of the compound.

        Returns:
            str: A string of the symbols and ratios of the given compound's elements.
        """
        print("Enter the compound formula with spaces between each element:")
        print("Example: for 'H$_2$O' write 'H2 O' or for 'Cu$_3$(PO$_4$)$_2$' write 'Cu3 P2 O8'")
        return input("Compound formula: ")