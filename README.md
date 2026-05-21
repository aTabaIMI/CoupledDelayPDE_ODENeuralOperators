# CoupledDelayPDE_ODENeuralOperators
This repository contains numerical scripts for the approximation of nonlinear kernel operators arising in a backstepping control design. The kernels are derived from a coupled ODE–hyperbolic PDE system and a parabolic PDE, and are learned via Deep Operator Networks (DeepONets). The scripts implement training DeepONets to approximate the kernel equations (ODE, hyperbolic PDE on a triangular domain, parabolic PDE on a rectangular domain) which validate exponential stability in the L² norm and demonstrates the efficiency of direct function‑evaluation based kernel computation. The methods and results are described in the paper currently under review by Automatica:  
*“Deep Neural Backstepping Control for Coupled Delay PDE-ODE Systems”* (under review by Automatica).  
A preprint will be linked here upon journal decision. 
## Contributors
- Malihe Abdolbaghi – conceptualization, implementation, simulations, proof read.
- Mohammad Keyanpour - conceptualization, proof read, validation.  
- Seyed Amir Hossein Tabatabaei - validation
