# Semantic-Enriched SQL generator (SE-SQL)
NL2SQL model for EMNLP 2020 submission, accepted by the Findings of EMNLP.

This readme is to introduce the implementation of Natural language to SQL generation (NL2SQL) service. We proposed an algorithmic framework named Semantic-Enriched SQL generator (SE-SQL) that enables flexibly access database than rigid API in the application while keeping the performance quality for the most commonly used cases.

The script is written in Python 3.6.5. (I am preparing the C++ version), the goal is to allow user retrive information from database by voice interaction (voice regconition is not our scope here, the raw input is text). In general, the input is query sentence such as "how many trips i had last month?". In demo, the query question is constrained as asking sport/drama/airport information given the backend database. The output is the retrived information from database. The UI is deployed as a chatbot through Slack Channel. 
