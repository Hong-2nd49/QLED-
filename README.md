QLED-RLopt/
├── README.md                  # 项目说明文档
├── qled_env/                  # 环境接口模块
│   ├── simulator_interface.py # 与COMSOL/Lumerical/替代模型的接口连接
│   ├── reward_function.py     # 基于载流子复合、EQE等指标的奖励函数
│   └── parameter_space.py     # 层图案、厚度、掺杂浓度等参数空间定义
├── agent/                     # 强化学习智能体逻辑模块
│   ├── dqn_agent.py           # 深度Q网络（DQN）智能体
│   └── policy_gradient_agent.py # 策略梯度智能体
├── data/                      # 数据存储目录
│   ├── generated_designs.csv  # 生成的器件设计方案
│   └── simulated_results/     # 仿真结果数据
├── notebooks/                 # 交互式分析笔记本
│   └── exploratory_analysis.ipynb # 探索性数据分析（Jupyter Notebook）
├── surrogate_model/           # 替代模型模块（用于加速仿真）
│   ├── train_surrogate.py     # 替代模型训练脚本
│   └── predict_performance.py # 基于替代模型的性能预测脚本
├── scripts/                   # 核心执行脚本
│   ├── run_optimization.py    # 启动优化流程脚本
│   └── simulate_design.py     # 器件设计仿真脚本
├── tests/                     # 测试模块
│   └── test_reward_logic.py   # 奖励函数逻辑测试
├── LICENSE                    # 开源许可证
└── requirements.txt           # 项目依赖库清单
