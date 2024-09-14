# PALLAS
## Project of RCOS 2024 FALL BY QI JIN

### Brief Introduction of project:
PALLAS(PArtial body simuLation with LAtend neural kinematic Solver) This project is simulation lower body kinematic information based on upper body information (from IMU sensors)

#### Detailed:
IMU is a kind of units which can obtained linear acceleration information and rotation information. By deploying for IMUs in upper body (both wrists, neck and waist) we can solve upper body motion and simulate motion of lower body motion. This project hopes to help people with lower body disabilities to use VR equipments

#### TODO LISTs:
- [x] 1. Simulate IMU data based on the [SMPL MODEL](https://smpl.is.tue.mpg.de/)
- [ ] 2. Decompose motion information to twist angle and shape rotation, [reference link](https://github.com/Jeff-sjtu/HybrIK?tab=readme-ov-file)
- [ ] 3. [Inversible neural inverse kinematic solver](https://arxiv.org/abs/1605.08803)

