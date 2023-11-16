# Hilbert Curve Visualizer

- This is a script that interpolates between different order hilbert curves.
- A hilbert curve is a space filling curve that can be used to map a 1D sequence to a 2D space recursively.
- Find out more about hilbert curves [Here](https://en.wikipedia.org/wiki/Hilbert_curve)
- Multiple effects are applied to acheive this effect:
    - Random walk through HSV space to generate colors
    - The colours are then mapped to a hilbert curve
    - A gradient is created between the new and old colours and is shifted around the curve
    - we interpolate between different order hilbert curves to create a smooth transition
    - The curve is translate,scaled and rotated using multiple overlapping sin waves to create a a sort of smooth but random effect 

![demovisualizer](demo.gif)


# Setup

- Clone the repo `git clone https://github.com/rudolfKischer/curvyhilburt.git`
- Make sure python3 is installad on your system
- navigate to the repo `cd curvyhilburt`
- run the setup script
    - Mac/Linux: `python3 setup.py`
    - Windows: `py setup.py`
- subsequent runs can be done by activating the enviorment and running `hilbert.py`
    - Mac/Linux: 
        ```
        source venv/bin/activate
        python3 hilbert.py
        ```
    - Windows: 
        ```
        venv\Scripts\activate
        py hilbert.py
        ```