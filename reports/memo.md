To: Tanzania Ministry of Water and Irrigation

From: Karen Warmbein, Hunter Ewing, Aaron Abrahamson

RE: Detecting Water Wells in Need of Repair in Tanzania

## Business Understanding and Need
Tanzania is a sub-Saharan East African country known for its vast wilderness areas. It has a population of 53 million, but half of the population don’t have access to clean water[[1]](#1). There are many organizations focused on reaching the communities that need water like Water Aid and Tanzania Water Project. Beyond the well-established humanitarian interest, however, there are other stakeholders concerned with the availability of unpolluted water, both in the region broadly and Tanzania specifically. Water insecurity has documented migratory effects, the flows of which can be extremely destabilizing, both domestically and for neighboring countries. According to the Fund for Peace, a non-profit think-tank, Tanzania is in fact the most stable country amongst itself and its immediate neighbors. Yet the Fund for Peace[[2]](#2) has issued a “high warning” advisory for the country, classing it as highly susceptible to destabilizing events.

The advisory classifications and rankings of Tanzania’s immediate neighbors are as follows, in descending order of fragility:

 - Malawi - 49, high warning
 - Zambia - 40,
 - Rwanda - 37, high warning
 - Mozambique - 33, high warning
 - Kenya - 25, alert
 - Uganda - 20, alert
 - Burundi - 15, alert
 - D.R.C. - 5, very high alert
 - Somalia - 2, very high alert

Tanzania, with its 53 million people, is the fifth most populous country in Africa. Should their already-tenuous water resources fail them, the country could see a mass exodus. The implications of a large, panicked emigration are sobering: migrants would fan out towards neighboring countries, seeking refuge but finding instead some of the most unstable and impoverished places in the world. This could cause a cascade of state collapses, as food scarcity overtakes refugees and violence becomes less unpalatable. With this view in mind, we make the argument that there stands a very large transnational African community having a common interest in providing for the sustained water needs of Tanzania. To the extent that global players (e.g., permanent members of the UN Security Council, G8 members) are interested in a calm and stable Africa, we make the argument that they should be deeply concerned with Tanzanian water security. Respecting the efficacy and efficiency of international aid, we propose that issuing resources towards a straightforward (if far-reaching) problem in a country that has the potential to weaken nine neighboring states to the point of collapse is a good “return on investment.”

## Project Goal
Our project was aimed at helping the Tanzanian government (and local/international support organizations and government initiatives) detect when a water well will be in need of repair. We include wells that are broken and wells that are functional, but needs repair in this category.

## Data Source
We used “Pump It Up - Data Mining the Water Table”[[3]](#3)  which has 59,400 records with 40 features of wells in Tanzania.

## Modeling and Results
We performed a thorough Exploratory Data Analysis of the dataset, and built several models to detect if a water well is in need of repairs. We tried 5 different classification models, settling on a random forest classifier as the best performer. It had an 82% accuracy overall.  The metric best for our business case, true “Needs Repair” recall, performed the worst, unfortunately, at 77%. While this is still a good predictive model, if given more time we would like to perform more feature engineering to increase this recall score.

## Next steps
These results are interesting in and of themselves. However we would like additional time to explore the following:

 - Perform more feature engineering
 - Analyze our final model on predicting three features (functional wells, functional but needs repair, and broken) verses the two features we predicted. These predictions can be used for a cost estimate of repairing or replacing the wells

Thank you for your time and interest in this issue!

### Citations
<a id="1">[1]</a> - https://www.wateraid.org/

<a id="2">[2]</a> - https://fundforpeace.org/2019/04/10/fragile-states-index-2019/

<a id="3">[3]</a> - https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/

