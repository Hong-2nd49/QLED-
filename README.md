### QLED-RLopt 

- `README.md`  
  项目整体介绍（中英），用例、结构、集成说明。
- `requirements.txt`  
  Python 依赖。
- `qled_env/parameter_space.py`  
  定义 QLED 结构编码与随机采样逻辑。
- `qled_env/reward_function.py`  
  物理驱动奖励函数（EQE、重叠度、惩罚项）。
- `qled_env/simulator_interface.py`  
  统一入口：Mock 物理、COMSOL CSV、代理模型三种评估模式。
- `qled_env/comsol_parser.py`  
  解析 COMSOL/TCAD 导出 CSV，计算 EQE proxy 与重叠度等。
- `agent/dqn_agent.py`  
  强化学习智能体占位实现（接口完整，可替换为真实 DQN/PPO）。
- `surrogate_model/train_surrogate.py`  
  从 `generated_designs.csv` 等数据训练 MLP 代理模型。
- `surrogate_model/predict_performance.py`  
  加载代理模型，对输入设计输出 EQE/overlap/penalty 预测。
- `data/generated_designs.csv`  
  示例/占位数据，用于测试与可视化。
- `data/simulated_results/`  
  存放真实或合成仿真结果 CSV。
- `scripts/run_optimization.py`  
  运行 RL 优化全流程（可选 surrogate/comsol）。
- `scripts/simulate_design.py`  
  预留：单一设计评估脚本（可用于调试）。
- `notebooks/01_explore_parameter_space.ipynb`  
  探索参数空间、载入 CSV。
- `notebooks/02_visualize_rl_results.ipynb`  
  可视化 EQE–结构关系和 RL 日志。
- `tests/test_reward_logic.py`  
  检查奖励函数合理性。
- `tests/test_simulator_interface.py`  
  检查模拟接口可正常返回指标。
- `LICENSE`  
  开源协议（推荐 MIT）。

### QDLED-3DSim 

- `README.md`  
  项目整体介绍（中英），定位为 3D 仿真 + 分析工具。
- `requirements.txt`  
  Python 依赖。
- `config/default_materials.yaml`  
  示例材料参数（可扩展为真实数据）。
- `simulator/comsol_parser.py`  
  仿真结果解析（与 QLED-RLopt 可共享理念/实现）。
- `simulator/geometry_builder.py`  
  用于描述/生成器件几何布局（层厚、平面图案参数）。
- `simulator/mesh_configurator.py`  
  网格、边界条件等配置占位。
- `ai_model/featurize_geometry.py`  
  将 2D/3D 结构 + 材料信息转为特征向量 / 图结构。
- `ai_model/train_model.py`  
  基于仿真数据训练代理模型（如 3D CNN / GNN）。
- `ai_model/evaluate_model.py`  
  评价模型预测 EQE/复合分布的误差与鲁棒性。
- `data/raw_simulations/`  
  存放原始仿真导出数据。
- `data/preprocessed/`  
  存放特征化后的训练数据。
- `visualization/render_3d_carriers.py`  
  生成载流子/复合的 3D/切片图。
- `scripts/run_full_simulation.py`  
  示例脚本：读配置 → 解析仿真 → 输出指标。
- `notebooks/01_inspect_simulation_data.ipynb`  
  浏览单个结构的三维分布。
- `notebooks/02_compare_structures.ipynb`  
  对比不同结构的性能指标。
- `LICENSE`  
  开源协议（推荐 MIT，与 QLED-RLopt 兼容）。

---

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

