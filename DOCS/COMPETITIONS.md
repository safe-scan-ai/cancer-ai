# Safe Scan: Machine Learning Competitions for Cancer Detection

Welcome to **Safe Scan**, a platform dedicated to organizing machine learning competitions focused on cancer detection. Our goal is to foster innovation in developing accurate and efficient models for cancer detection using machine learning. Here, you will find all the details needed to participate, submit your models, and understand the evaluation process.

## Table of Contents

1. [Overview](#overview)
2. [Competition Trigger and Data Batch Handling](#competition-trigger-and-data-batch-handling)
3. [Model Submission Requirements](#model-submission-requirements)
4. [Evaluation and Scoring](#evaluation-and-scoring)
5. [Configuration and Development](#configuration-and-development)
6. [Command-Line Interface (CLI) Tools](#command-line-interface-cli-tools)
7. [Communication Channels](#communication-channels)
8. [Contribute](#contribute)
   
## Overview

Safe Scan organizes dynamic competitions focused on cancer detection using machine learning. These competitions provide participants with the opportunity to develop and test their models in a responsive and adaptive environment driven by real-world data.

## Competition Trigger and Data Batch Handling

- **Competition Initiation**: Competitions are triggered by data batch insertions from external medical institutions, creating a steady stream of new, non-public data for testing purposes.
- **Data Handling Process**: Medical institutions upload each new batch of data to a central reference repository on Hugging Face, along with a reference entry for the new data batch file.
- **Automatic Detection and Competition Start**: Validators monitor this centralized repository for new data batch entries. When new data is detected, validators initiate a competition by downloading and processing the data batch.

## Model Submission Requirements

- **Format**: All models must be in ONNX format. This ensures uniform testing and allows for broad deployment options, including on mobile and web platforms.
- **License**: All models must be licensed under the MIT License to ensure the open nature of the competition and to allow unrestricted use worldwide.
- **Training Code**: Each submission should include the code used for training the model to ensure transparency and reproducibility.
- **Upload Process**: Models are to be uploaded to Hugging Face. Miners then submit the Hugging Face repository link on the blockchain for evaluation by validators.
- **Timing Constraint**: Only models submitted at least 30 minutes before the competition start time are eligible for evaluation. This requirement ensures that models have not been retrained with the new data batch, maintaining fairness and integrity across the competition.

## Evaluation and Scoring

- **Independent Evaluation**: Each validator independently evaluates the submitted models according to predefined criteria.
- **Scoring Mechanism**: Detailed scoring mechanisms are outlined in the [DOCS](/DOCS/competitions) directory. Validators run scheduled competitions and assess the models based on these criteria.
- **Winning Criteria**: Models are ranked according to the evaluation metrics, with the top 10 performers receiving rewards.
- **Rewards**: Rewards are distributed among the top 10 performing miners according to the following distribution:
  - 1st place: 40%
  - 2nd place: 20%
  - 3rd place: 12%
  - 4th place: 8%
  - 5th place: 6%
  - 6th place: 5%
  - 7th place: 4%
  - 8th place: 3%
  - 9th place: 1%
  - 10th place: 1%
  
  The total reward pool for each competition is the emission allocated for miners divided by the number of competitions held. Miners beyond the top 10 receive minimal participation rewards.

## Command-Line Interface (CLI) Tools

- **Local Testing**: Miners are provided with an easy-to-use command-line interface (CLI) for local testing of their models. This tool helps streamline the process of testing models, uploading to Hugging Face, and submitting to the competition.

## Communication Channels

Stay connected and up-to-date with the latest news, discussions, and support:

- **Discord**: Join our [Safe Scan Discord channel](https://discord.gg/rbBu7WuZ) and the Bittensor Discord in the #safescan channel for real-time updates and community interaction.
- **Dashboard**: Access the competition dashboard on https://dashboard.safe-scan.ai/ 
- **Blog**: Visit our [blog](https://safe-scan.ai/news/) for news and updates.
- **Twitter/X**: Follow us on [Twitter/X](https://x.com/SAFESCAN_AI) for announcements and highlights.
- **Email**: Contact us directly at [info@safescanai.ai](mailto:info@safescanai.ai) for any inquiries or support.

## Development

- **Software Lifecycle**: The project follows a structured software lifecycle, including Git flow and integration testing. These practices ensure robust development and encourage community contributions.


## Contribute

We welcome contributions to this project! Whether you're interested in improving our codebase, adding new features, or enhancing documentation, your involvement is valued. To contribute:

- Follow our software lifecycle and Git flow processes.
- Ensure all code changes pass integration testing.
- Contact us on our [Safe Scan Discord channel](https://discord.gg/rbBu7WuZ) for more details on how to get started.
