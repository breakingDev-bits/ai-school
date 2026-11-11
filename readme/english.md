# Neuron Training Visualizer (Neural Network Training Visualizer)

This project is a mini-application for real-time visualization of training a simple neural network (a single-layer perceptron). The project consists of a **C++** backend (which computes training epochs using gradient descent) and a **Python** frontend utilizing **Tkinter** and **ctypes** for integration and graphical rendering.

---

## 🛠 C++ Backend Build

To run the program, you first need to compile the C++ file (`nn.cpp`) into a dynamic library (`.so`, `.dylib`, or `.dll` depending on your operating system).

> **Important!** Use optimization level `-O2` or `-O1`. To ensure correct behavior with `ctypes`, avoid level `-O3` (level `-O2` is recommended).
> * **Windows:** Use **MinGW** (MSVC is not supported).
> * **Linux:** Use `Clang++` or `G++` (under WSL, `G++` works best).
> 
> 

### Build Commands:

* **Linux:**
```bash
g++ -O2 -shared -fPIC nn.cpp -o libnn.so

```


* **macOS:**
```bash
g++ -O2 -shared -fPIC nn.cpp -o libnn.dylib

```


* **Windows (MinGW):**
```bash
g++ -O2 -shared nn.cpp -o nn.dll

```



---

## 📦 Requirements & Execution

### 1. System Requirements

* Python 3.x
* The `tkinter` package (usually included with Python on Windows/macOS; on Linux, install it via your package manager, e.g., `sudo pacman -S tk` on Arch Linux).

### 2. Running the Application

1. Place the compiled library (`libnn.so`, `nn.dll`, or `libnn.dylib`) in the same directory as the Python script.
2. Run the script:
```bash
python <script_name>.py

```



---

## 🚀 How to Use the Application

1. **Configuration Parameters (Top Panel):**
* Enter the input values `x` and `y`, the target value (`відповідь`), learning rate (`lr`), maximum epochs, and the error threshold (`поріг помилки`).
* Click the **"Нова модель"** (New Model) button to initialize new random weights and compute the epochs in the C++ backend.


2. **Playback Controls:**
* **"Крок"** (Step) — advance to the next training epoch.
* **"Старт / Пауза"** (Start / Pause) — automatically play through the training process with a specified delay (controlled by the "затримка, мс" slider).
* **"В кінець"** (To End) — instantly jump to the final epoch.


3. **Visualizations:**
* **Neural Network Diagram:** Displays current weights (`w0`, `w1`), bias (`bias`), and the calculated prediction.
* **Graphs:** Demonstrate the evolution of predictions (compared to the target value) and the reduction of error across epochs.


4. **Model Testing:**
* In the bottom panel, you can enter arbitrary values for `x` and `y` and click **"Передбачити"** (Predict) to test how the trained model responds to new data using the exported `nn_predict` function.