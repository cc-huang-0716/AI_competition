# PPO-GAT-XGBoost Feature Selector

> Reinforcement Learning–based Feature Selection Framework for Stock Movement Prediction  
> Powered by **PPO + GATv2 + XGBoost** and optimized for **F1-score**.

---

## Installation
pip install torch-geometric "ray[tune]" wandb kafka-python python-dotenv

## Overview

This project uses **PPO（Proximal Policy Optimization）** ，combines **Spatio-Temporal Actor（LSTM + GATv2）**，selects the important features,then uses **XGBoost classifier** to predict the price of the stock. **F1-score** is the reward 。

---

## Model Architecture

```
Input Features ──▶ LSTM ─┐
                        ├──▶ Feature Mask (Sigmoid)
           ──▶ GATv2 ──┘

Selected Features ──▶ XGBoost Classifier ──▶ Prediction
                               ▲
                               │
                       F1-score Reward
                               │
                    PPO Loss Backpropagation
```

---

## Modules Breakdown

| module name | function |
|----------|------|
| `load_data_partition()` 
| `SpatioTemporalActor`   
| `ppo_loss()`            
| `xgb_predict()`        
| `compute_reward()`      
| `train()`               

---

## Outputs

| output file | illustration |
|----------|------|
| `reward_curve.png` 
| `confusion_matrix.png` 
| `Final F1-Score`（console）

---

## Credits

-  Torch, Torch-Geometric, XGBoost
-  Ray Tune for future scalability
-  wandb for optional experiment tracking
-  sklearn for metrics and preprocessing

