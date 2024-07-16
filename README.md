<div align="center">

# **Safe Scan Cancer AI detection **  <!-- omit in toc -->
Bittensor Subnet for improving cancer detection algorithms

[Discord](https://discord.gg/npwNy9tU) [Twitter](https://x.com/SAFESCAN_AI)

</div>

## Introduction

Welcome to Safe Scan Cancer AI Detection, a groundbreaking initiative leveraging the power of AI and blockchain technology to revolutionize cancer detection. Our mission is to make advanced cancer detection algorithms accessible and free for everyone. Through our project, we aim to provide cutting-edge, open-source tools that support early cancer diagnosis for patients and healthcare professionals worldwide.

This repository contains subnet code to run on Bittensor network.  


## Vision & Roadmap
Cancer is one of the most significant challenges of our time, and we believe that AI holds the key to addressing it. However, this solution should be accessible and free for everyone. Machine vision technology has long proven effective in early diagnosis, which is crucial for curing cancer. Yet, until now, it has largely remained in the realm of whitepapers. SAFESCAN is a project dedicated to aggregating and enhancing the best algorithms for detecting various types of cancer and providing free computational power for practical cancer detection. We aim to create open-source products that support cancer diagnosis for both patients and doctors.

To date, many crypto and AI projects, particularly those focused on medicine, have struggled to achieve real-world implementation due to various barriers. Our solution focuses on:

1. **Development of Applications and Software:** Invest in the ongoing development and enhancement of our cancer detection applications and software to ensure they are at the cutting edge of technology.
2. **Medical Device Registration:** Allocate funds to cover the costs associated with registering our solutions as medical devices, ensuring they meet all regulatory requirements for safety and efficacy.
3. **Marketing and Awareness:** Implement comprehensive marketing strategies to raise awareness about our solutions and their benefits, making them known to both potential users and healthcare professionals.
4. **Collaboration and Networking:** Build strong networks with cancer organizations, researchers, and healthcare providers to facilitate the practical implementation and continuous improvement of our technology.
5. **Continuous Improvement of Algorithms:** Reward top researchers, maintain algorithms in the open domain, and constantly expand our anonymized cancer detection dataset through partnerships and user contributions.
6. **Legislative Efforts:** Engage in legislative activities to support the recognition and adoption of AI-driven cancer detection technologies within the medical community.

By focusing on these areas, we aim to overcome the barriers to the practical use of AI in cancer detection and provide a solution that is accessible to everyone.

To expedite the process and navigate the complexities of medical certification, we are beginning our initiatives with authorized clinical trials.  After completing clinical trials of our first project, SELFSCAN – an application for detecting skin cancer through selfies – we will focus on its deployment as a Class II medical device in the USA and Europe, obtaining the necessary FDA and CE approvals. 

Concurrently, with the help of the Bittensor community and our unique tokenomics supporting researchers, we will continuously improve the best cancer detection algorithms. This ensures that, by the time our products are brought to market, our solutions surpass all existing algorithms.

Subsequently, we will focus on detecting other types of cancer, starting with breast and lung cancer.

For more information about our project, roadmap, and progress, visit our website:

[www.safe-scan.ai](http://www.safe-scan.ai/)

www.skin-scan.ai


## Tokenomy

Our tokenomics are unique and designed to support our research and development of new algorithms as well as serving real life tasks.
We have 3 types of entities in our network: Validators, Miners, and Researchers.

## Validators

Validators are receiving 42% of emission and are responsible for following things:

- distributing real-life tasks from our API and synthetic challenges from external Dataset API
- testing researchers
- validating tasks done by miners

## Miners

They are responsible for performing computational tasks on behalf of the validators. They are provided with machine learning models, which they must utilize to carry out these computations.

## Researchers

They play a crucial role in our ecosystem, as they are dedicated to enhancing machine learning models. An online ranking [LINK TO WEBSITE] compiles information from validators to determine the best-performing models.

Researchers with the highest model accuracy receive 15% of emissions as an incentive, fostering the development of advanced technologies in cancer research.

## Applications

As a subnet, we focus on the practical application of our technology. Therefore, we continuously reinvest profits from the subnet into the development of applications and software for cancer detection, medical device registration costs, and marketing. The problem of underutilizing AI algorithms in practical cancer detection is complex and multifaceted. Solving this problem in a form that is accessible, affordable, and user-friendly is our primary goal. Here is what we intend to do in terms of practical applications:

### Skinscan App

Our first product is the [SKINSCAN app](www.skin-scan.ai) which allows us to diagnose skin changes using a simple photo from a phone and determine whether they are potentially cancerous with over 90% accuracy.

Our app increases awareness of skin diseases, informs users about current UV levels, and most importantly, supports the early diagnosis of skin cancer. It helps assess whether a particular skin lesion is potentially cancerous. It also assists doctors by enabling the easy export of changes in skin lesions over time along with their descriptions. Additionally, thanks to feedback from doctors, it will be possible to build a growing dataset and further improve the effectiveness of our skin cancer detection algorithm.

Until we obtain certification as a Class II medical device from the FDA and CE, the app will be available to participants in clinical trials: patients and doctors involved in the trials. Afterward, it will be deployed on the Apple and Google stores.

### AI Powered Breast Screening Tools

Next, we will focus on breast cancer, the most common type of cancer. Studies show that as many as 1 in 8 women may develop breast cancer in their lifetime. Screening is key to diagnosing and effectively treating cancer. However, the number of specialists available to analyze mammogram results is insufficient. In many countries, there is no capacity for timely analysis of mammogram images. Research indicates that AI analysis of mammogram images is as accurate as the best radiologists. By creating open-source software, we can provide the necessary technology wherever it is needed, significantly enhancing early detection capabilities for breast cancer. 

Leveraging the power of Bittensor, we will ensure the project has independent funding and remains free for doctors and patients while continuously improving algorithms thanks to our researchers and a growing dataset. Initially, we will also focus on clinical trials with partner medical units, obtaining the necessary FDA and CE approvals only afterward, and then introducing the ready product as a web application to the market. 

Additionally, we will develop integrations with companies that manufacture mammography machines without built-in AI detection, further extending the reach and impact of our technology.

### AI-Enhanced Lung Cancer Detection Tools

Lung cancer remains a significant global health challenge, being the second most common cancer and the leading cause of cancer mortality worldwide. According to the World Health Organization, lung cancer accounts for approximately 2.1 million new cases and 1.8 million deaths annually. The survival rate for lung cancer improves dramatically with early detection; however, the scarcity of radiologists to interpret CT scans timely impedes early diagnosis and treatment.

AI technology presents a promising solution to this issue. Studies indicate that AI algorithms can detect lung cancer with a sensitivity comparable to that of expert radiologists, significantly reducing diagnostic errors. Our mission is to develop open-source AI software that democratizes access to high-quality diagnostic tools, ensuring that even regions with limited medical resources can benefit from early detection technologies.

By leveraging Bittensor's innovative platform, we will ensure sustainable funding for our project, allowing us to offer our solution free of charge to healthcare providers and patients. This funding model will support continuous improvement of our AI algorithms, driven by a growing dataset and cutting-edge research contributions.

Initially, we will conduct rigorous clinical trials in partnership with leading medical institutions to validate our technology. Post-validation, we will seek FDA and CE approvals, ensuring our software meets the highest regulatory standards. Once approved, our AI-enhanced detection tool will be deployed as a web application, making it easily accessible to healthcare providers worldwide.

In addition, we aim to collaborate with manufacturers of CT scanners who currently lack integrated AI detection capabilities. By integrating our AI software with their existing hardware, we can extend the benefits of our technology, enhancing the diagnostic capabilities of their machines and ultimately improving patient outcomes.

## Participation

### Validator

1. Contact us on [![Discord](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/npwNy9tU) to request the Dataset API key which will enable you to pull resources for testing researchers and generation of synthetic queries.
2. Setup config parameters which can be found in ./neurons/cancer_ai/utils/config.py. You can either provide these parameters with the flags when running the validator.py or adjust the default values for the config parameters in the config.py file directly. The fetched in previous step api key is one of the parameters.
3. Run the validator.py with our subnet <subnet_id> flag. Example can be found here: https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_mainnet.md

### Miner

1. Setup config parameters which can be found in ./neurons/cancer_ai/utils/config.py. You can either provide these parameters with the flags when running the miner.py or adjust the default values for the config parameters in the config.py file directly. A miner can be both a Regular miner (offering computational power) and the Researcher. To proceed with just Regular miner (without announcing to the subnet participation as a Researcher) make sure that the --researcher flag is set to false (the default value).

2. Run the miner.py script with our subnet <subnet_id> flag. Example can be found here: https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_mainnet.md

### Researcher

Note that if you are planning to participate as the Researcher it is adviced to run the Researcher Miner in the immunity period (right after registration). The Researcher Miner is at the same time performing Regular Miner job, but as the reward system is based on the response pace it is possible that the general score of the Miner will drop if you are executing both Researcher and Miner tasks on the same machine.
Hence, if you are already running a Regular Miner successfully after the immunity period it is also adviced to adjust the miner.py to handle the Researcher tasks asynchronously and/or introducing proxy for handling the Researcher processing on another machine.

1. Setup config parameters which can be found in ./neurons/cancer_ai/utils/config.py. You can either provide these parameters with the flags when running the miner.py or adjust the default values for the config parameters in the config.py file directly. A miner can be both a Regular miner (offering computational power) and the Researcher. To proceed as the Researcher Miner make sure that the --researcher flag is set to true.

2. Run the miner.py script with our subnet <subnet_id> flag. Example can be found here: https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_mainnet.md

3. When the testing is done the outcome can be found on [![Statistics API]](https:cancer-ai/stats). If it appears that your Researcher models is better then our current model, reach us on [![Discord](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor). We will then test your model outside of the subnet to confirm on its accuracy and hopefully introduce it as a new base model for the Subnet and the Skinscan App!
