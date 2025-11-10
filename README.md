## Project 1: QLED-RLopt
# Reinforcement Learning for Optimizing Layered and Microstructured QLED Architectures

### README.md (English)

# QLED-RLopt
**Reinforcement Learning Optimization Framework for Microstructured Quantum Dot LED Devices**

This project uses reinforcement learning to optimize the architecture of multilayer QLEDs, with a focus on alternating ZnO and QD structures, layer sequencing, and carrier balance. COMSOL or surrogate simulations evaluate performance metrics such as EQE, recombination profiles, and emission uniformity.

## Features
- Modular RL agent with customizable reward functions
- Simulator interface for COMSOL / Lumerical or surrogate models
- Design parameter encoder for geometry, materials, and thicknesses
- Jupyter notebooks for analysis and visualization

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_optimization.py --episodes 100
```

## Directory Overview
- `qled_env/`: Simulation interfaces and design encoders
- `agent/`: RL algorithms
- `data/`: Simulated designs and results
- `surrogate_model/`: Optional model to replace simulator
- `notebooks/`: Visualization and analysis

---

### README.md（简体中文）

# QLED-RLopt
**用于微结构量子点发光二极管（QLED）器件结构优化的强化学习框架**

本项目通过强化学习方法优化多层QLED器件架构，重点探索ZnO与量子点交替排列结构、层叠顺序及电荷平衡问题。系统采用COMSOL仿真或替代代理模型评估关键性能指标，如EQE（外量子效率）、复合分布以及发光均匀性。

## 项目特点
- 模块化强化学习算法，支持自定义奖励函数
- 支持COMSOL / Lumerical仿真或代理模型的接口
- 参数编码器涵盖层结构、材料属性与厚度配置
- 提供Jupyter笔记本用于结果可视化与数据分析

## 快速启动
```bash
pip install -r requirements.txt
python scripts/run_optimization.py --episodes 100
```

## 目录结构说明
- `qled_env/`：仿真接口与结构参数编码器
- `agent/`：强化学习策略与训练逻辑
- `data/`：保存生成的设计参数与仿真结果
- `surrogate_model/`：可选代理模型（用于加速优化）
- `notebooks/`：可视化与分析示例

---

## Mock Data Sample

**data/generated_designs.csv**
```
design_id,ZnO_ratio,QD_layers,HTL_thickness,ZnO_thickness,EQE,recomb_overlap
1,0.4,2,20,30,0.163,0.78
2,0.6,3,25,25,0.178,0.81
3,0.5,2,18,28,0.142,0.75
```

## RL Stub (scripts/run_optimization.py)
```python
from agent.dqn_agent import DQNAgent
from qled_env.simulator_interface import QLEDSimulator
from qled_env.parameter_space import sample_design

agent = DQNAgent()
sim = QLEDSimulator()

for ep in range(100):
    design = sample_design()
    metrics = sim.evaluate(design)
    reward = metrics['EQE'] + 0.1 * metrics['recomb_overlap']
    agent.learn(design, reward)
    print(f"Episode {ep}: EQE = {metrics['EQE']:.3f}")
```

## COMSOL Export Parser Snippet (simulator/comsol_parser.py)
```python
import pandas as pd

def parse_comsol_csv(file_path):
    df = pd.read_csv(file_path)
    carrier_profile = df[['x', 'y', 'z', 'n_electron', 'n_hole']]
    recombination = df['recomb_rate'].sum()
    return {
        'carrier_map': carrier_profile,
        'total_recomb': recombination,
        'EQE': recombination * 0.25  # scaled proxy
    }
```

---

## Project 2: QDLED-3DSim
# AI-Assisted QLED 3D Carrier Simulation Toolkit

### README.md（English）

# QDLED-3DSim
**Toolkit for Simulating and Visualizing Carrier Dynamics in Microstructured QLED Devices**

This repository provides tools for modeling and analyzing 3D carrier transport and recombination behavior in QLEDs with patterned micro/nano-scale architectures. Input custom device geometries, simulate in COMSOL or Lumerical, and use ML models to predict and visualize optoelectronic performance.

## Features
- COMSOL result parser with recombination and carrier profiling
- 3D grid and mesh-based geometry builder
- ML models for surrogate prediction (3D CNN or GNN)
- Plot scripts for spatial carrier and recombination maps

## Quick Start
```bash
pip install -r requirements.txt
python scripts/run_full_simulation.py --config config/default_materials.yaml
```

## Directory Overview
- `simulator/`: Parsers and geometry definition
- `ai_model/`: Featurization, training, and prediction
- `visualization/`: 3D plots and map generators
- `notebooks/`: Data inspection and result comparison

### README.md（简体中文）

# QDLED-3DSim
**量子点LED（QLED）器件三维载流子模拟与可视化工具包**

该工具库旨在对具有微/纳米结构的QLED器件进行三维载流子输运和复合行为建模与可视化。用户可导入自定义几何结构、基于COMSOL或Lumerical运行仿真，并利用机器学习模型预测和分析其光电性能。

## 项目特点
- COMSOL输出解析器，支持提取载流子浓度与复合分布
- 支持构建2D/3D器件结构网格与几何体
- 机器学习代理模型（3D CNN 或图神经网络）
- 三维可视化工具用于绘制电荷分布与复合热图

## 快速启动
```bash
pip install -r requirements.txt
python scripts/run_full_simulation.py --config config/default_materials.yaml
```

## 目录结构说明
- `simulator/`：COMSOL解析器与几何结构定义
- `ai_model/`：特征处理、模型训练与推理模块
- `visualization/`：用于输出3D图形的可视化脚本
- `notebooks/`：结果比对与数据浏览示例

---
Let me know if you'd like a COMSOL example input file, or a surrogate model prototype in PyTorch.
