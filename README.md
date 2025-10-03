# An AI-driven grid optimisation system within the Australian National Energy Market
## Abstract
As Australia progresses towards an era dominated by renewable energy, the issues surrounding curtailments (forced energy losses) from these technologies grow in parallel, as their uncontrollable nature leads to a disconnect between supply and demand. 
This dissertation explores the realm of machine learning to determine if data from the Australian Energy Market Operator (AEMO) on the National Energy Market (NEM) and open-source weather data from Open-Meteo, the industry standard, can be used to train 
a model to predict potential curtailments within the grid. This will deliver cost and environmental savings through the reduction of energy waste and promote investment in a stabilised Australian energy market. Data on Regional Reference Price 
(RRP, sourced from designated reference nodes from each region in the NEM), generation, solar intensity, wind speed and temperature will be utilised to train a series of algorithms to predict when a curtailment will occur and how much energy will 
need to be ejected from the NEM to ensure network stability. The models investigated include a Random Forrest, xgBoost, Artificial Neural Network (ANN), Long-Short Term Memory (LSTM) neural network, an ANN-Particle Swarm Optimisation (ANN-PSO) hybrid, a Transformer 
and a Transformer-PSO hybrid. These models’ ability to predict the volume of lost energy and RRP for forward days will be used to determine the overall performance.

## Context of the Problem
Context of the problem
The world is moving into an age of energy dominated by renewables. Clean Energy Australia’s 2025 report found that 40% of Australia’s total electricity generation came from Renewable Energy Systems (RES) in the 2024 calendar year and the Australian government has set a target to have  82% renewable capacity by 2030 (Clean Energy Council, 2025). 
In January 2018, AEMO reported green energy penetration (the maximum recorded total green energy produced in a month) to be 30.2% of all energy within the NEM; a value that has steadily increased, reaching 75.6% in November 2024 (AEMO, n.d.). This represents an increase of more than 150% over the roughly seven-year period (Figure 1).
Various flaws have emerged with this transition, the key being the disconnect between green energy suppliers and consumers, leading to inefficiencies and curtailments. These flaws cause excess strain on the grid, instability and unpredictable pricing for energy portfolio managers.

## Research Questions
1.	How can AI-driven optimisation and forecasting solutions aid in Australia’s transition to a modern green energy system?
2.	How can AI-based energy management systems help Australian consumers become more self-sufficient within the centralised energy grid?
3.	What AI-driven solutions are most scalable and effective for decentralising and evolving the Australian energy grid, considering factors like cost-efficiency and reliability?
4.	What locations in Australia’s key geographies, Queensland, New South Wales, and Victoria, are most susceptible to green energy curtailments?
5.	Which attributes are most effective at predicting a RES curtailment?

## Methodology
| Methodology (Strategy) | Method (Implementation) |
|---|---|
| Use green energy generation data and weather data to predict green energy curtailments | Use public data from AEMO, Open Electricity and Open-Meteo to predict green energy curtailment. |
| Construct a dataset for green energy curtailment prediction | Price/Demand will be from AEMO, Wind & Solar Generation from Open Electricity, and Temperature, wind speeds and solar intensity will be sourced from Open-Meteo. |
| Design traditional & hybrid model for green energy curtailment prediction | Train and tune a Random Forrest, xgBoost, Artificial Neural Network (ANN), an ANN-Particle Swarm Optimisation (ANN-PSO) hybrid, Transformer and Transformer-PSO hybrid model and compare results. The PSO Algorithm will be used to optimise the hyperparameters of the hybrid models. |
| Evaluate the model’s performance for green energy curtailments by comparing to the true historical values | For classification of the curtailment: Using Accuracy, Precision, Recall and F1 as the base-line measures. Also, a precision-recall curve and receiver operating characteristics (ROC) curve, and the areas under these curves, will be used for greater analysis of the model. For prediction of the volume of lost energy (regression): Mean Absolute Error (MAE), Root Square Error (RSE) and R2 score. |
| Feed forecast and current generation & weather data to the models to predict potential green energy curtailments and the locations that they could occur | Integrate live data into the model for live predictions. |
