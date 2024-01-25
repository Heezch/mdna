goal: import pymdna as mdna

I have the following modules containg the following classes:

spline.py
    - SplineFrames, Twister

utils.py
    - RigidBody, Shapes

geometry.py
    - NucleicFrames, ReferenceBase 

generators.py
    - SequenceGenerator, StructureGenerator

Note, 
    NucleicFrames, ReferenceBase, TwistFrames depend on RigidBody
    StructureGenerator depends on SequenceGenerator
