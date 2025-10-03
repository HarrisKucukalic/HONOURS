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

## Acknowledgements
I would like to express my deepest thanks to my supervisor, Professor Yu Kai Wang, for his invaluable support and mentorship during this research. His expertise in AI models was instrumental to the creation of this dissertation. 
I would like to also thank Mr Chris Smyth for his contributions and aid in editing and providing feedback. His 20+ years of energy market experience were invaluable and helped added some key insights that would otherwise have been missed.
This work would not have been possible without the publicly available data provided by AEMO, Open-Meteo and Open Electricity, all of whom have created a repository of easily accessible electricity and weather data for free.

## Bibliography
Acun, B., Morgan, B., Richardson, H., Steinsultz, N., & Wu, C. (2024). Unlocking the Potential of Renewable Energy Through Curtailment Prediction. Electrical Engineering and Systems Science. https://arxiv.org/abs/2405.18526
AEMO. (n.d.). NEM data dashboard. AEMO | Australian Energy Market Operator. https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/data-dashboard-nem
AEMO | Australian Energy Market Operator. (n.d.). Aggregated price and demand data. https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
Agrawal, A. (2018). A Conceptual Model for Peer to Peer Energy Trading using Blockchain. https://doi.org/10.15224/978-1-63248-153-5-04
Ahmad, T., Zhang, D., Huang, C., Zhang, H., Dai, N., Song, Y., & Chen, H. (2021, January 5). Artificial intelligence in sustainable energy industry: Status Quo, challenges and opportunities. Journal of Cleaner Production, 293, 125745. https://www.sciencedirect.com/science/article/pii/S0959652621000548
Alassery, F., Alzahrani, A., Khan, A. I., Irshad, K., & Islam, S. (2022, March 5). An artificial intelligence-based solar radiation prophesy model for green energy utilization in energy management system. Sustainable Energy Technologies and Assessments, 52, 102060. https://www.sciencedirect.com/science/article/pii/S2213138822001126
Australian Government - Department of Industry, Science and Resources. (n.d.). Cache://www.industry.gov.au/publications/australias-artificial-intelligence-ethics-principles/australias-AI-ethics-principles - Google search. https://www.industry.gov.au/publications/australias-artificial-intelligence-ethics-principles/australias-ai-ethics-principles
Ayoub, N., Musharavati, F., Pokharel, S., & Gabbar, H. (2018, October 21). ANN Model for Energy Demand and Supply Forecasting in a Hybrid Energy Supply System. IEEE International Conference on Smart Energy Grid Engineering (SEGE). https://ieeexplore-ieee-org.ezproxy.lib.uts.edu.au/document/8499514
Biggar, D. R., & Hesamzadeh, M. R. (2024). Crises in Texas and Australia: Failures of energy-only markets or unforeseen consequences of price caps? Energy Economics, 137, 107810. https://doi.org/10.1016/j.eneco.2024.107810
Bureau of Meteorology. (2025). Climate data online. Australia's official weather forecasts & weather radar - Bureau of Meteorology. https://www.bom.gov.au/climate/data/index.shtml?bookmark=200
Breiman, L. (2001). RANDOM FORESTS. https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
Bunodiere, A., & Lee, H. S. (2020). Renewable energy curtailment: Prediction using a logic-based forecasting method and mitigation measures in Kyushu, Japan. Energies, 13(18), 4703. https://doi.org/10.3390/en13184703
Chen, T., & Guestrin, C. (2016). XGBoost. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794. https://doi.org/10.1145/2939672.2939785
Clean Energy Council. (2024, April 16). Rooftop solar generates over 10 per cent of Australia's electricity. Renewable Energy Australia | Clean Energy Council. https://cleanenergycouncil.org.au/news-resources/rooftop-solar-generates-over-10-per-cent-of-australias-electricity
Clean energy Australia. (2025, May 27). Clean energy Australia 2025 report. Renewable energy in Australia | Clean Energy Council. https://cleanenergycouncil.org.au/news-resources/clean-energy-australia-report-2025
Crismale, D. (2024, July 31). How much energy does the average home use? Finder. https://www.finder.com.au/energy/how-much-energy-does-the-average-home-use#:~:text=The%20average%20annual%20electricity%20usage,in%20Sydney%20is%205%2C237kWh
Cushing, S., & Wheatley, L. (2024, June 20). Hype Cycle for Low-Carbon Energy Technologies, 2024. Gartner. https://www.gartner.com/interactive/hc/5523095?ref=solrAll&refval=426769847
‌Dou, L., You, J., Hong, Z., Xu, Z., Li, G., Street, R. A., &amp; Yang, Y. (2013). 25th Anniversary Article: A Decade of Organic/Polymeric Photovoltaic Research. Advanced Materials, 25(46), 6642–6671. https://doi.org/10.1002/adma.201302563
Durham University. (n.d.). The Ecliptic: the Sun's Annual Path on the Celestial Sphere. https://astro.dur.ac.uk/~ams/users/solar_year.html#:~:text=The%20tilt%20of%20the%20Earth's,with%20the%20celestial%20equator%20horizontal
El Naqa, I., & Murphy, M. J. (2015). What is machine learning? (pp. 3-11). Springer International Publishing. https://link.springer.com/chapter/10.1007/978-3-319-18305-3_1
Espe, E., Potdar, V., &amp; Chang, E. (2018). Prosumer Communities and Relationships in Smart Grids: A Literature Review, Evolution and Future Directions. Energies, 11(10), 2528. https://doi.org/10.3390/en11102528
González, P. A.  and Zamarreño, J.M., Prediction of hourly energy consumption in buildings based on a feedback artificial neural network, Energy and Buildings, 37, pp. 595-601, 6 2005. Elsevier. https://www-sciencedirect-com.ezproxy.lib.uts.edu.au/science/article/pii/S0378778804003032?via%3Dihub
Geeks for Geeks. (2025, April 4). Neural Networks Architecture [Diagram]. Geeks for Geeks. https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/
GeeksforGeeks. (2025, August 13). What is LSTM - Long Short Term Memory? https://www.geeksforgeeks.org/deep-learning/deep-learning-introduction-to-long-short-term-memory/
Gupta, A. (2021, June 1). XGBoost versus random forest. Medium. https://medium.com/geekculture/xgboost-versus-random-forest-898e42870f30 
IBISWorld. (2023, August). Industry Report - Solar Electricity Generation in Australia. https://my-ibisworld-com.ezproxy.lib.uts.edu.au/au/en/industry/D2619b/performance
IBISWorld. (2023, August). Industry Report - Wind and Other Electricity Generation in Australia. https://my-ibisworld-com.ezproxy.lib.uts.edu.au/au/en/industry/D2619a/performance
IBM. (n.d.). What is random forest?. https://www.ibm.com/think/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems
IEEE. (2020, June). IEEE code of ethics. IEEE - The world's largest technical professional organization dedicated to advancing technology for the benefit of humanity. https://www.ieee.org/about/corporate/governance/p7-8.html
Kahlid, M. (2024, May 24). Energy 4.0: AI-enabled digital transformation for sustainable power networks. Computers & Industrial Engineering, 193, 110253. Elsevier. https://www.sciencedirect.com/science/article/pii/S0360835224003747
Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. Proceedings of ICNN'95 - International Conference on Neural Networks. https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf
Krensky, P. (2024, July 30). Hype Cycle for Data Science and Machine Learning, 2024. Gartner. https://www.gartner.com/interactive/hc/5633191?ref=solrAll&refval=426769631
Kuo, P. H., & Huang, C. J. (2018, April 2). A green energy application in energy management systems by an artificial intelligence-based solar radiation forecasting model. Energies, 11, 11040819. MDPI. https://www.mdpi.com/1996-1073/11/4/819
Liu, Z., Sun, Y., Xing, C., Liu, J., He, Y., Zhou, Y., & Zhang, G. (2022, August 2). Artificial intelligence powered large-scale renewable integrations in multi-energy systems for carbon neutrality transition: Challenges and future perspectives. Energy and AI, 10, 100195. Elsevier. https://doi.org/10.1016/j.egyai.2022.100195
Maji, D., Irwin, D., Shenoy, P., & Sitaraman, R. K. (2025). A first look at node-level curtailment of renewable energy and its implications. Proceedings of the 16th ACM International Conference on Future and Sustainable Energy Systems, 293-304. https://doi.org/10.1145/3679240.3734627
McKee, A. (2025, February 5). Step by Random Step: Exploring the Random Walk Model. datacamp. https://www.datacamp.com/tutorial/random-walk
Memmel, E., Steens, T., Schlüters, S., Völker, R., Schuldt, F., & Von Maydell, K. (2023). Predicting renewable curtailment in distribution grids using neural networks. IEEE Access, 11, 20319-20336. https://doi.org/10.1109/access.2023.3249459
Mercer, D. (2024, September 8). Australia is 'wasting' record amounts of green energy. Here's why experts say it's a good thing. Retrieved from https://www.abc.net.au/news/2024-09-08/renewable-energy-wasted-as-australia-greens/104321770
Mohammad Reza Maghami, Jagadeesh Pasupuleti, &amp; Ekanayake, J. (2024). Energy storage and demand response as hybrid mitigation technique for photovoltaic grid connection: Challenges and future trends. Journal of Energy Storage, 88, 111680–111680. https://doi.org/10.1016/j.est.2024.111680
National Health and Medical Research Council. (2018). Australian Code for the Responsible Conduct of Research. https://www.nhmrc.gov.au/about-us/publications/australian-code-responsible-conduct-research-2018#block-views-block-file-attachments-content-block-1
National Institute of Economic and Industry Research (NIEIR). (2016). NIEIR Review of EDD weather standards for Victorian gas forecasting. https://nieir.com.au/wp-content/uploads/2016/07/NIEIR-EDD-Review-April-2016.pdf
Pagani, G. A., &amp; Aiello, M. (2011). Towards Decentralization: A Topological Investigation of the Medium and Low Voltage Grids. IEEE Transactions on Smart Grid, 2(3), 538–547. https://doi.org/10.1109/tsg.2011.2147810
Parag, Y., &amp; Sovacool, B. K. (2016). Electricity market design for the prosumer era. Nature Energy, 1(4). https://doi.org/10.1038/nenergy.2016.32
Peacock, F. (2024, October 14). How much do solar batteries cost in Australia? SolarQuotes.com.au. https://www.solarquotes.com.au/battery-storage/cost/
Rafferty, J. F. (2025, June 13). Why is predicting the weather so difficult for meteorologists? Encyclopedia Britannica. https://www.britannica.com/story/why-is-predicting-the-weather-so-difficult-for-meteorologists
‌Reiter, H. L., &amp; Greene, W. (2016). The Case for Reforming Net Metering Compensation: Why Regulators and Courts Should Reject the Public Policy and Antitrust Arguments for Preserving the Status Quo. Energy Law Journal, 37(2), 373.
Rich, J. (2025, February 27). AEMC updates market price cap for 2025-26. AEMC. https://www.aemc.gov.au/news-centre/media-releases/aemc-updates-market-price-cap-2025-26
Rojek, I., Mrozinski, A., Kotlarz, P., Macko, M., & Mikolajewski, D. (2023, December 14). AI-basedcomputational model in sustainable transformation of energy markets. MDPI. https://www.mdpi.com/1996-1073/16/24/8059
Roy, R. (2022, June 14). Neural Networks: Forward pass and Backpropagation. towards data science. https://towardsdatascience.com/neural-networks-forward-pass-and-backpropagation-be3b75a1cfcc/
Sankarananth, S., Karthiga, M., Suganya, E., Sountharrajan, S., & Bavirisetti, D. P. (2023, August 16). AI-enabled metaheuristic optimization for predictive management of renewable energy production in smart grids. Energy Reports, 10, pp.1299-1312. Elsevier. https://www.sciencedirect.com/science/article/pii/S2352484723011459
Schank, R. C. (1987, December 15). What is AI, anyway?. AI magazine, 8(4), 59-59. https://doi.org/10.1609/aimag.v8i4.623
Şerban, A. C., & Lytras, M. D. (2020, May 8). Artificial intelligence for smart renewable energy sector in Europe—Smart energy infrastructures for next generation smart cities. IEEE Access, 8. IEEE. https://ieeexplore.ieee.org/abstract/document/9076660
Shafiullah, M., Ahmed, S. D., &amp; Al-Sulaiman, F. A. (2022). Grid Integration Challenges and Solution Strategies for Solar PV Systems: A Review. IEEE Access, 10, 52233–52257. https://doi.org/10.1109/access.2022.3174555
Shams, M. H., Niaz, H., Hashemi, B., Jay Liu, J., Siano, P., & Anvari-Moghaddam, A. (2021, October 15). Artificial intelligence-based prediction and analysis of the oversupply of wind and solar energy in power systems. Energy Conversion and Management, 250, 114892. Elsevier. https://doi.org/10.1016/j.enconman.2021.114892
Shell Energy. (2025, June 10). Environmental Certificates Market Update: June 2025. https://shellenergy.com.au/energy-insights/certificates-market-update-june-2025/#ENVIRONMENTAL-CERTIFICATES
Syed, T. (2023, May 15). Weather data EDA | Model training. Kaggle: Your Machine Learning and Data Science Community. https://www.kaggle.com/code/thabresh/weather-data-eda-model-training
The Association for Computing Machinery. (2018). ACM Code of Ethics and Professional Conduct. Association for Computing Machinery. https://www.acm.org/code-of-ethics
TheEnergyExperts.com.au. (2024, March 8). Why Australians are increasingly buying bigger solar power systems (and adding battery storage). TheEnergyExperts.com.au. https://theenergyexperts.com.au/why-australians-are-increasingly-buying-bigger-solar-power-systems-and-adding-battery-storage/#:~:text=About%20180%2C000%20of%20Australia's%203.7,In%20Tariffs%20(FiTs)%20reduced
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. 31st Conference on Neural Information Processing Systems (NIPS 2017). https://arxiv.org/pdf/1706.03762
Wise. (n.d.). Japanese yen to Australian dollars exchange rate history | Currency converter | Wise. https://wise.com/au/currency-converter/jpy-to-aud-rate/history/01-08-2020
YAĞCI, H. E. (2021, February 22). Feature scaling with scikit-learn for data science. Medium. https://hersanyagci.medium.com/feature-scaling-with-scikit-learn-for-data-science-8c4cbcf2daff

