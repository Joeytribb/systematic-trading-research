# Iniyan Andrews Joseph
*M.Sc. Computer Science (Data Science)*
Email: iniyandrews@gmail.com | [LinkedIn](https://www.linkedin.com/in/iniyandrews) | [GitHub: Joeytribb](https://github.com/Joeytribb) | Calicut, India

---

## RESEARCH INTERESTS
**Core Interests:** Quantitative Trading Systems, Market Microstructure, Algorithmic Execution, Monte Carlo Risk Modeling, Supervised Learning for Time-Series, and Ensemble Feature Selection.

---

## EDUCATION
**NIELIT, Calicut** — *Calicut, India*
*M.Sc. in Computer Science (Specialization in Data Science)* | Expected Completion: May 2026
*   **Current CGPA:** 8.55/10.0
*   **Relevant Coursework:** Mathematics for Machine Learning, Advanced Statistical Techniques, Probability & Statistics, Data Analytics, Stochastic Processes.

**The American College, Madurai** — *Madurai, India*
*B.Sc. in Computer Science* | 2021 – 2024
*   **CGPA:** 7.4/10.0

---

## QUANTITATIVE RESEARCH & TECHNICAL PROJECTS

**Systematic Trading Research & Bias Auditing** | *[View on GitHub](https://github.com/Joeytribb/systematic-trading-research)*
*Independent Quantitative Researcher* | Python, pandas, scikit-learn
*   **Engineered a zero-bias intraday mean-reversion trading system** capable of compounding a $500 micro-account to $10,000 in a mathematically verified 6.3-month median timeframe (0.02% risk of ruin).
*   **Zero-Bias Execution Architecture:** Identified and eliminated 15 systemic backtest biases and 2 look-ahead bugs (including threshold contamination and same-candle gap fills), dropping naive win-rates from 92% to a highly realistic, out-of-sample 64.25% (+1.69% EV per trade).
*   **Execution Friction & Limits:** Hardcoded worst-case market microstructure friction into the backtester (exchange fees, funding rate drag, and dynamic order-book slippage), analytically proving the strategy caps at ~$541,500 before slippage negates the edge.
*   **Stochastic Sizing:** Applied Kelly Criterion approximations within a 50,000-path Monte Carlo engine to map the exact probability of ruin, formulating a two-stage risk-ratchet.

**Short-Term Stock Prediction: Supervised Learning (Project HOPE)** | *[View on GitHub](https://github.com/Joeytribb/HOPE)*
*Lead Researcher* | Python, Scikit-Learn, LSTM, TensorFlow
*   Conducted rigorous statistical analysis to identify S&P 500 and Nifty 50 stocks primed for 1% daily gains, framing the problem as a supervised learning classification task.
*   Built and evaluated robust feature engineering pipelines to process high-noise financial data, isolating predictive signals and discarding redundant features to prevent deep learning model collapse.
*   Demonstrated that variance-reducing, regularized linear models (190.65% Return, 2.0 Sharpe Ratio) significantly outperformed complex LSTM models by avoiding overfitting on high-dimensional datasets.

**High-Frequency Trading using Deep Machine Learning** | *[View on GitHub](https://github.com/Joeytribb/Bithax-RL-trader)*
*Lead Researcher* | Python, PyTorch, OpenAI Gym
*   Engineered an autonomous trading agent via Proximal Policy Optimization (PPO) that processed a highly complex 42-dimensional feature vector representing dynamic market states.
*   Addressed the curse of dimensionality by validating feature importance; utilized out-of-sample stress testing and look-ahead bias checks to ensure generalized, robust performance.
*   Achieved a Sharpe Ratio of 1.4 by optimizing statistical metrics across both bull and bear market regimes.

---

## TECHNICAL SKILLS
*   **Financial Analytics:** Market Microstructure, Algorithmic Execution, Monte Carlo Simulation, Risk Management (Kelly Criterion, Sharpe, Drawdown), Dealing with Concept Drift.
*   **Data Science & Stats:** Scikit-Learn, Pandas, NumPy, Matplotlib, Statistical Analysis, Feature Engineering.
*   **Machine Learning:** Supervised Learning, Feature Selection, Dimensionality Reduction, Reinforcement Learning (PPO, DQN), PyTorch, TensorFlow.
*   **Programming & Tools:** Python (Advanced), C++, SQL, Git/GitHub, Docker, Linux, LaTeX.
