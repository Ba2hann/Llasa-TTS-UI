# Llasa-TTS-UI
- This is a UI "Project" for the Llasa-3B. It's almost entirely AI coded, and my understanding is limited.


## Run the install.bat
Please make sure that you have Python 3.9 and Cuda Toolkit installed.

## Run the run.bat
That should be it!

## Common Fixes
Make sure you are using an updated version of pip:

python -m pip install --upgrade pip setuptools wheel

Make sure you have an NVIDIA GPU as your cuda:0 device.

This is for Windows, and tried on Windows 10.

Some packages may get updated and cause issues in future (especially gradio). These are the most updated working versions:

gradio==4.44.1

huggingface-hub==0.29.3

bitsandbytes==0.45.3

accelerate==1.5.2

numpy==2.0.2

optimum-quanto==0.2.7

triton-windows==3.2.0.post17

**Unfortunately I am not quite equipped enough yet to help with detailed questions and issues. Good luck!**
