# U-CREAT: Unsupervised Case Retrieval using Events extrAcTion

The repository contains the full codebase of experiments and results of the ACL 2023 paper **"U-CREAT: Unsupervised Case Retrieval using Events extrAcTion"**.

![/Images/ucreat_pipeline.png](/Images/ucreat_pipeline.png)

## Contributions
We make the following contributions:

1. Considering the lack of available benchmarks for the Indian legal setting, we create a new benchmark for Prior Case Retrieval focused on the Indian legal system (IL-PCR) and provide a detailed analysis of the created benchmark. Due to the large size of the corpus, the created benchmark could serve as a helpful resource for building information retrieval systems for legal documents. We release the corpus and model code for the purpose of research usage.
2. We propose a new framework for legal document retrieval: U-CREAT (Unsupervised Case Retrieval using Events Extraction), based on the events extracted from documents. We propose different event-based models for the PCR task. We show that these perform better than existing state-of-the-art methods both in terms of retrieval efficiency as well as inference time.
3. We show that the proposed event-based framework and models generalize well across different legal systems (Indian and Canadian systems) without any law/demography-specific tuning of models.

## Models and Data 
1. `Model Checkpoints` : The checkpoints for the finetuned bert and distilbert models mentioned in the paper are available for download at [here](https://1drv.ms/f/s!Ao1lGmmnesu6l7cSBRElaJIhVNzwxw?e=QqQy2S).
2. `Data` : The Dataset is ONLY for research use and NOT for any commercial use. The IL-PCR dataset is available for free via [request](https://forms.gle/X4z3hfVpj3FDFhfaA). For the COLIEE'21 dataset please refer to [COLIEE-2021](https://sites.ualberta.ca/~rabelo/COLIEE2021/).
3. Each algorithm in the `Models/` subdirectory uses the corpus stored in the local `data/` subdirectory. Please see the algorithm specific READMEs for an explanation of how to prepare the corpus for running with the codebase.

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The ILPCR dataset and UCREAT software follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.
## Citation

```
@inproceedings{joshi-etal-2023-ucreat,
    title = "{U-CREAT}: Unsupervised Case Retrieval using Events extrAcTion",
    author = "Joshi, Abhinav  and
      Sharma, Akshat  and
      Tanikella, Sai Kiran and
      Modi, Ashutosh",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = july,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    abstract = "The task of Prior Case Retrieval (PCR) in the legal domain is about automatically citing relevant (based on facts and precedence) prior legal cases in a given query case. To further promote research in PCR, in this paper, we propose a new large benchmark (in English) for the PCR task: IL-PCR (Indian Legal Prior Case Retrieval) corpus. Given the complex nature of case relevance and the long size of legal documents, BM25 remains a strong baseline for ranking the cited prior documents. In this work, we explore the role of events in legal case retrieval and propose an unsupervised retrieval method-based pipeline U-CREAT (Unsupervised Case Retrieval using Events Extraction). We find that the proposed unsupervised retrieval method significantly increases performance compared to BM25 and makes retrieval faster by a considerable margin, making it applicable to real-time case retrieval systems. Our proposed system is generic, we show that it generalizes across two different legal systems (Indian and Canadian), and it shows state-of-the-art performance on the benchmarks for both the legal systems (IL-PCR and COLIEE corpora).",
}
```

## Contact
In case of any queries, please contact <ashutoshm.iitk@gmail.com>, <ajoshi@cse.iitk.ac.in>, <akshat.iitkanpur@gmail.com>, <saikirantanikella@gmail.com>
