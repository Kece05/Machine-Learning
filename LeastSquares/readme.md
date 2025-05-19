# 📊 Least Squares - Linear Regression in C++

This project contains a simple implementation of **Linear Regression** using the **Least Squares** method, written in **pure C++** with no external libraries. It reads in data from a CSV file, performs matrix-based regression, and makes predictions based on user input.

---

## 🚀 Features

* Implements the least squares formula:
  **w = (XᵀX)⁻¹ Xᵀy**
* All matrix operations (transpose, multiplication, inverse, determinant) coded from scratch
* Reads feature data from a CSV file
* Predicts target value based on user-entered input
* Calculates and displays Mean Squared Error (MSE)

---

## 📂 File Structure

```
Least-Squares/
│
├── LeastSquares.cpp              # Main program with all logic and matrix functions
├── data.csv              # Example CSV file with training data
```

---

## 📥 Input File Format

Your CSV should have:

* A header row (feature names + target)
* Each subsequent row should contain numeric values

Example: `data.csv`

```
hours_studied,attendance_rate(%),homework_score(%),quiz_avg(%),final_exam_score
8,80,90,70,85
6,70,80,60,75
...
```

---

## 🧪 Example Run

**Input:**

```
Enter the training data file name: data
```

**Output:**

```
Linear Regression Weights:
Bias: 25.825
Weight for feature hours_studied: 2.975
Weight for feature attendance_rate(%): 0.18
Weight for feature homework_score(%): 0.1575
Weight for feature quiz_avg(%): 0.1075

Mean Square Error: 33.750000%

Create prediction of your final exame score:
What is your hours_studied: 10
What is your attendance_rate(%): 90
What is your homework_score(%): 85
What is your quiz_avg(%): 75

Your final exam score is predicted to be: 93.225000
```

---

## 🛠️ Build & Run

### Compile:

```bash
g++ -std=c++11 -o run LeastSquares.cpp
```

Make sure `data.csv` is in the same directory or update the file path accordingly in the code.

---

## 🎓 Educational Purpose

This project is ideal for:

* Learning how linear regression works under the hood
* Understanding matrix algebra in C++
* Practicing CSV parsing and user input/output in C++

---

## ✍️ Author

Developed by \Keller Bice — for learning and experimentation with fundamental machine learning techniques.

---

## 🙏 Acknowledgments

* Special thanks to [ChatGPT](https://openai.com/chatgpt) by OpenAI for assisting with:

  * Structuring the C++ code for linear regression using the least squares method
  * Providing sample CSV data format
  * Creating this `README.md` and project documentation
