# 🚀 RISC-V Processor Simulator

## 📌 Overview  
This project implements a **cycle-accurate simulator** for a **32-bit RISC-V processor** in **Python** and **C++**. It supports both a **single-stage** and **five-stage pipelined processor**, accurately modeling register states, memory behavior, and performance metrics.

---

## 🎯 Features
- ✅ Implements core **RISC-V instructions** (`ADD`, `SUB`, `XOR`, `LW`, `SW`, `BEQ`, etc.).
- ✅ Supports **single-stage and pipelined execution**.
- ✅ Handles **RAW (Read-After-Write) and Control hazards**.
- ✅ Provides **cycle-by-cycle register and memory states**.
- ✅ Computes **performance metrics (CPI, IPC, Execution Cycles)**.

---

## 🏗 Installation & Setup
### Clone the Repository
```bash
git clone https://github.com/roncell/RISC-V-processor.git
cd RISC-V-processor
