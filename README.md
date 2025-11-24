# LLMChess

LLMChess is a Python program, to be run through [UV](https://docs.astral.sh/uv/), that allows a human player to play chess against an AI powered by a Large Language Models. It provides a console and a graphical interface.

The features are:
1. Human vs AI and AI vs AI playing style
2. List valid moves for Human
3. Can run console app with many command line parameter to specify many things but especially how to connect to Azure AI Foundry
4. Clear error messages printed out for each possible operation of the application
5. A --verbose mode to debug possible problems

The application is built using the Microsoft Agent Framework to communicate with Azure AI Foundry LLM services. It uses the following environment variables to connect to Azure Ai Foundry LLM service: AZURE_AI_FOUNDRY_ENDPOINT, AZURE_AI_MODEL
