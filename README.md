
## Project title: Réttir AI
[the project video](https://youtu.be/Dn9UeYrt50c)

# Introduction:

This project draws inspiration from Iceland’s traditional sheep festival, exploring how interactive hand 
gestures can control animated sheep in real time. Herding plays a crucial role in these festivals and holds 
deep cultural significance in Iceland. By blending tradition with modern interactive technology, this 
project allows users to experience virtual herding through hand gestures.
The animated sheep are created using Stable Diffusion. Hand gestures are captured via MediaPipe to 
trigger specific animations. This approach connects the digital and physical worlds, combining 
traditional festival elements with modern interaction techniques. The project highlights the potential of 
gesture-based interactions in entertainment and education.

# Setup instructions:


**1. Install Conda**

If you haven't installed Conda, you can download and install Miniconda or Anaconda for your operating system.

**2. Create and Activate the Conda Environment**

Open terminal and copy and paste these three commands (one at a time) to create a new conda environment for this unit.

```
conda create --name aim python=3.10
```

```
conda activate aim
```

```
conda install -c conda-forge -y ipython jupyter
```

**3. Install Required Packages**

Run the following command to install the necessary dependencies:

```
pip install opencv-python mediapipe numpy imageio pillow ipython stable-diffusion-pytorch
```

**4. Download Required Files**

  4.1. Ensure you have the required animation files inside a movement/ folder in your project directory:

       movement/gather.gif

       movement/alarm.gif

       movement/turnleft.gif

       movement/turnright.gif

       movement/idle.gif

   4.2. Stable Diffusion related model files

   The exact model path and download depends on how the library stable-diffusion-pytorch is implemented.

**5. Running the Code**

```
python main.py
```
