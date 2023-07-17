# Bachelor_2023
Code for my Bachelor Thesis 'Fast Ice Detection under Uncertainty'.

- monte_carlo.py gives a simple class (named Handler() ) for all the different operations Monte Carlo simulations offer on the problem.
- models.py has the Boostrap as well as the Guided model.
- The folder "simulation" contains both the Perlin Noise and a class for the simulated observations.

- "particles" is a copy of the module "particles" from
N. Chopin, https://github.com/nchopin/particles, commit: 20c2730ed6ab8f68fcc700390dbf8db739f34d81
with the only changed bit being the data type of the binomial distribution in distributions.py to obtain less data size.
