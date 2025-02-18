


# TODO Create particle ensemble
class Particles:
    """
        Args:
    number: int: Number of particles
    energy: float: Energy of the particles in eV
    charge: float: Charge of the particles in e    
    mass: float: Mass of the particles in amu
    """
    def __init__(self, number: int, mass: float = 4, charge: float = 2, energy: float = 3.52e6) :
        self.number = number
        self.energy = energy*1.602176634e-19
        self.energy_eV = energy
        self.charge = charge*1.602176634e-19
        self.charge_e = charge
        self.mass = mass*1.66053906660e-27
        self.mass_amu = mass
