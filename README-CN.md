# Mini Soft Sensor & Inferential Measurement（迷你软测量与推断测量）

**从零开始、零依赖的 C 语言实现**，涵盖过程控制工程中的软测量与推断测量算法。每个模块对应 MIT、Stanford、Berkeley、CMU 等顶尖大学课程，将教科书中的公式转化为可运行的 C 代码，实现理论与实践的桥接。

## 子模块总览

| 子模块 | 主题 | 参考课程 |
|--------|------|----------|
| [mini-kalman-filter-soft-sensor](mini-kalman-filter-soft-sensor/) | 线性KF、EKF、UKF、自适应噪声估计、Kalman平滑、工业软测量应用 | MIT 6.302, Stanford ENGR205, CMU 24-677 |
| [mini-neural-network-soft-sensor](mini-neural-network-soft-sensor/) | 前馈神经网络质量估计、反向传播/SGD训练、模型验证、性能监控 | MIT 6.390, Stanford CS229, CMU 10-701 |
| [mini-pls-partial-least-squares](mini-pls-partial-least-squares/) | NIPALS、SIMPLS、核PLS、递推/在线PLS、交叉验证、潜变量软测量建模 | MIT 2.171, Stanford ENGR205, UMetrics |
| [mini-quality-estimator-inferential](mini-quality-estimator-inferential/) | 机理/数据驱动/混合推断模型、Kalman质量估计、偏差校正、多速率传感器融合 | MIT 6.302, Stanford ENGR205, Berkeley ME233 |
| [mini-soft-sensor-data-driven-pca](mini-soft-sensor-data-driven-pca/) | PCA分解（SVD/EVD）、核PCA、自适应/递推PCA、T²/SPE过程监控、推断感知 | MIT 2.171, Stanford ENGR205, CMU 24-677 |
| [mini-soft-sensor-data-reconciliation](mini-soft-sensor-data-reconciliation/) | 稳态/动态数据协调、WLS求解器、过失误差检测、测量冗余度分析 | MIT 10.490, Stanford ENGR205, CMU 24-677 |
| [mini-soft-sensor-maintenance-aging](mini-soft-sensor-maintenance-aging/) | 性能指标（RMSE/R²/SPE）、老化漂移检测、自适应模型更新、集成维护策略 | MIT 2.171, Stanford ENGR205, CMU 24-677 |
| [mini-virtual-flow-meter](mini-virtual-flow-meter/) | 基于工艺测量的虚拟流量计、流体物性、管道水力学、状态估计、不确定性量化 | MIT 10.490, Stanford ENGR205, TU Delft P&E |

## 设计理念

- **零外部依赖** — 纯 C（C99/C11），仅使用 `libc` 和 `libm`
- **模块自包含** — 每个目录自带 `Makefile`、`include/`、`src/`、`examples/`、`demos/`、`tests/`
- **理论到代码的映射** — 每个模块包含 `docs/` 目录，内有课程对齐说明
- **工业应用导向** — 所有实现面向真实过程控制场景：质量估计、传感器校验、故障检测、虚拟仪表

## 构建方式

每个模块相互独立。进入模块目录后运行：

```bash
cd mini-kalman-filter-soft-sensor
make all    # 构建全部
make test   # 运行测试
```

需要 **GCC** 和 **GNU Make**。

## 项目结构

```
mini-soft-sensor-inferential/
├── mini-kalman-filter-soft-sensor/        # 卡尔曼滤波软测量应用
├── mini-neural-network-soft-sensor/       # 神经网络推断传感器
├── mini-pls-partial-least-squares/        # 偏最小二乘回归
├── mini-quality-estimator-inferential/    # 推断质量估计与传感器融合
├── mini-soft-sensor-data-driven-pca/      # PCA驱动的软测量与过程监控
├── mini-soft-sensor-data-reconciliation/  # 数据协调与过失误差检测
├── mini-soft-sensor-maintenance-aging/    # 软测量维护与老化管理
└── mini-virtual-flow-meter/               # 虚拟流量计与状态估计
```

## 许可证

MIT
