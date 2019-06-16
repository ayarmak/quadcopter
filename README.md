# Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

In this project, I have designed an agent to fly a quadcopter, and then trained it using a reinforcement learning algorithm to take off from the ground.

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/ayarmak/quadcopter.git
cd quadcopter
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6.3
source activate quadcop
pip install -r requirements.txt
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

5. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.
