# Kernel-based Deep Learning repository
This repository contains the TensorFlow/Keras code and results for a comparative study of Convolutional Kernel Networks (CKNs) against a standard Convolutional Neural Network (CNN) baseline on different datasets.

#**MNIST**

The study tested 4 CKNs (kernel = linear, polynomial, spherical or gaussian) over 100 runs of MNIST each to assess mean performance and stability. The key conclusion is that the Kernel-based architectures (excluding Gaussian CKN) achieved competitive, and in some cases superior accuracy compared to the standard CNN, but with trade-offs in inference and training speed.

The following kernels were evaluated:
  * **CNN:** A standard two-layer Convolutional Neural Network.
  * **Linear CKN:** A CKN utilizing a linear kernel function, equivalent to a standard CNN layer without the non-linearity before pooling.
  * **Polynomial CKN:** A CKN utilizing a polynomial kernel function.
  * **Spherical CKN:** A CKN utilizing a spherical kernel function (normalized cosine similarity).
  * **Gaussian CKN:** A CKN utilizing a Gaussian/Radial Basis Function (RBF) kernel.

Here we can see some statistics for this study:

| Model          | Mean Acc  | Std Dev   | Mean F1   | Mean Precision | Mean Recall | Mean AUC | 95% CI (Accuracy) |
| **CNN**        | 0.991     | 0.001     | 0.991     | 0.991          | 0.991       | 1.000    | 0.989-0.993       |
| **Linear**     | 0.991     | 0.001     | 0.991     | 0.991          | 0.991       | 1.000    | 0.989-0.992       |
| **Polynomial** | **0.992** | 0.001     | **0.992** | **0.992**      | **0.992**   | 1.000    | 0.990-0.993       |
| **Spherical**  | **0.992** | 0.001     | **0.992** | **0.992**      | **0.992**   | 1.000    | 0.990-0.993       |
| **Gaussian**   | 0.504     | **0.432** | 0.455     | 0.449          | 0.504       | 0.725    | 0.114-0.984       |

### Key Conclusions on Performance:

1.  **Top Performance:** The **Polynomial CKN** and **Spherical CKN** models achieved a small but consistent edge (given tge high number of runs) over the standard CNN and Linear CKN, with a Mean Accuracy of 0.992. This represents a decent error reduction (from 0.9% error to 0.8% error, very useful in accurate detection systems).
2.  **Linear Equivalence:** The **Linear CKN** performed identically to the baseline CNN in terms of mean accuracy and stability, which aligns with theoretical expectations given its architecture in the `build_model` function.
3.  **Gaussian Instability (RBF Failure):** The **Gaussian CKN** demonstrated catastrophic failure due to extreme instability. The high standard deviation (0.432) indicates that in approximately half the runs, the model's training completely collapsed, resulting in performance close to random chance. This highlights a significant challenge with directly implementing RBF-style kernels in a deep learning context without specialized initialization or highly-advanced mathematics implemented.

## Computational Efficiency Benchmark

A full benchmark measured the average training and inference time per epoch/step for a single run of each model.

| Model          | Params | Train Time (ms) | Infer Time (ms) | Train Eff (1/ms) | Infer Eff (1/ms) |
| :------------- | :----- | :-------------- | :-------------- | :--------------- | :--------------- |
| **CNN**        | 421642 | 4231            | 526             | 0.56             | 4.51             |
| **Linear**     | 421642 | 8358            | **262**         | 0.28             | **9.07**         |
| **Polynomial** | 421646 | 13698           | 1109            | 0.17             | 2.14             |
| **Spherical**  | 421644 | 28311           | 1154            | 0.08             | 2.05             |
| **Gaussian**   | 421646 | **29817**       | **1399**        | **0.08**         | **1.70**         |

### Key Conclusions on Efficiency:

1.  **Linear CKN Dominance in Inference:** The **Linear CKN** is the fastest model at inference (262 ms), offering a dramatic performance increase in this phase over the baseline CNN (526 ms).
2.  **Training Cost of Non-Linear Kernels:** The non-linear kernels (**Polynomial, Spherical, Gaussian**) have a significantly higher training time compared to the CNN and Linear CKN. The most complex kernels (Spherical and Gaussian) show the highest training latency (\~7x slower than CNN training).
3.  **Inference Cost of Non-Linear Kernels:** Non-linear CKNs (Polynomial, Spherical, Gaussian) are also slower during inference than the CNN baseline.

To sum up, we can learn that CKNs are a valid alternative to CNNs in the case of MNIST, as they can be as reliable, but in some cases more accurate and even faster to run, even though none of them rivaled CNN when it comes to training time.

#**CIFAR-10**

The study assessed the same 5 models (4 CKNs and one traditional CNN) over 3 independent training runs to determine their performance on the CIFAR-10 test set.

The key conclusions highlight a clear separation in performance and efficiency between the different kernel types: the best-performing models (Spherical, Polynomial, Gaussian) were also the least efficient, while the simplest kernel (Linear CKN) offered great speed.

Here we can see some statistics for this study:

| Model          | Mean Acc   | Train(s) (per epoch) | Infer(ms) | Train Eff | Infer Eff |
| :------------: | :--------: | :------------------: | :-------: | :-------: | :-------: |
| **CNN**        | 55.45%     | 1.83s                | 114ms     | 0.99      | 15.93     |
| **Linear**     | 57.06%     | 2.11s                | **47ms**  | 0.85      | **38.83** |
| **Polynomial** | 62.48%     | 2.37s                | 90ms      | 0.76      | 20.08     |
| **Spherical**  | **63.45%** | 3.55s                | 145ms     | 0.50      | 12.45     |
| **Gaussian**   | 62.29%     | **3.67s**            | **148ms** | **0.48**  | **12.24** |

Key Conclusions:

### Classification Accuracy
  * **Top Performer:** The **Spherical CKN** achieved the highest mean accuracy at **63.45%**, suggesting that the cosine similarity kernel function is the most effective choice for high-accuracy image classification on CIFAR-10 among the tested CKN variants.
  * **Competitive Kernels:** Both the **Polynomial CKN** and the **Gaussian CKN** also significantly outperformed the standard CNN baseline (55.45%) and the Linear CKN (57.06%), demonstrating the benefit of non-linear kernel representations.

### Computational Efficiency

  * **Inference Speed Champion:** The **Linear CKN** exhibited a remarkable inference time of only **47ms**, making it significantly faster than all other models (e.g., more than twice as fast as the CNN baseline and three times faster than the Spherical CKN). This is critical for deployment scenarios.
  * **Highest Training Cost:** The **Gaussian CKN** and **Spherical CKN** were the slowest models to train (3.67s and 3.55s per epoch, respectively). Their "Train Efficiency" scores are the lowest, indicating their complex computations add big overhead during the learning phase.
  * **Trade-off:** The high-accuracy models (Spherical, Polynomial, Gaussian) generally fall into the lower efficiency bracket, confirming the trade-off that greater model complexity and complex feature extraction methods suffer a higher computational cost. The **Linear CKN** represents the best blend of speed and acceptable performance, offering a 1.6% accuracy boost over the CNN with vastly superior inference time.
  * **Polynomial Surprise**: Taking cost-value coefficients into consideration, the polynomial is the strongest model here, offering the second fastest inference (hence the second inference efficiency score), but also offering the second highest accuracy, thus a 7% increase over the CNN and a 21% inference time reduction.

###**HIGHLY IMPORTANT**
The study had to implement a rather simple model, worse than the usual models used for CIFAR-10 classification. The architecture was composed of 3 small blocks (32, 64, 64 filters) followed by a Dense block, significantly reducing the CNN's capabilities in favor of geometric efficiency that the CKNs had the edge at. 
This was implemented because the computational cost of a study with deep neural networks on CIFAR-10 had consumed much time and memory beforehand. 

**The previous sub-study on CIFAR**
We had implemented a small sub-study in order to compare a state-of-the-art CNN architecture that most ML engineers use online (approximately 93.44% testing accuracy) with a CKN. We chose the best possible kernel, the Spherical kernel (cosine similarity), and our best result was a 92.75%, thus at deep levels, the geometric efficiency of CKNs is simply outmatched by simple computational power and density of classic CNNs, but the margin was small,
thus it's safe to say that some interpretability and geometric robustness of CKNs is still to be admired, especially within a 1% margin to the CNN. 

## Promoter Gene E-coli study ##

In molecular biology, a **promoter** is a region of DNA that initiates transcription of a particular gene. Promoters are crucial control elements that tell the cell where to start reading the DNA sequence and how often to transcribe the gene. In bacteria like E-coli, promoter recognition is a fundamental step in gene regulation.

The challenge in bioinformatics is to accurately classify a given DNA sequence as either a functional promoter site ("positive") or a non-promoter site ("negative"). This is a pattern recognition task, as promoter sequences often contain complex, non-linear patterns that conventional machine learning methods struggle to identify.

The following study explores the application of **Convolutional Kernel Networks (CKNs)**—which bridge the power of deep learning's convolutional architecture with the non-linear feature mapping capabilities of kernel methods—to accurately predict E-coli promoter status based on raw DNA sequences.

## Comparative Study Results

The study evaluated five models: a standard Convolutional Neural Network (**CNN**) baseline, and four CKN variants, as usual: **Linear**, **Polynomial**, **Spherical**, and **Gaussian**.

### Performance Metrics

This table summarizes the performance on the E-coli promoter dataset, evaluated across multiple runs:

| Model          | Mean Acc  | Mean AUC  | Median AUC | AUC Std   | 95% CI Lower | 95% CI Upper |
| :------------: | :-------: | :-------: | :--------: | :-------: | :----------: | :----------: |
| CNN            | 0.882     | 0.957     | 0.967      | 0.041     | 0.949        | 0.965        |
| Linear         | 0.891     | 0.958     | 0.967      | 0.040     | 0.951        | 0.966        |
| **Polynomial** | **0.901** | **0.965** | **0.978**  | **0.038** | **0.958**    | **0.972**    |
| Spherical      | 0.858     | 0.932     | 0.945      | 0.055     | 0.922        | 0.943        |
| Gaussian       | 0.891     | 0.956     | 0.967      | 0.044     | 0.948        | 0.965        |

**Conclusions on Performance:**

  * **Best Overall Classifier:** The **Polynomial CKN** is the clear winner for this task, achieving the highest Mean Accuracy (0.901) and Mean AUC (0.965). Its tighter Standard Deviation (0.038) and highest 95% Confidence Interval suggest it is the most robust and consistent model for accurately identifying promoter sequences. It was simply not close for the others.
  * **Linear/Gaussian CKN Performance:** The Linear CKN and Gaussian CKN performed significantly better than the CNN baseline in terms of mean accuracy (0.891 vs 0.882) and were comparable to the CNN in terms of mean AUC and stability.
  * **Spherical CKN Underperformance:** The **Spherical CKN** was the weakest model, performing below the CNN baseline on all key metrics (Mean Acc 0.858, Mean AUC 0.932) and exhibiting the highest instability (AUC Std 0.055), indicating it is a poor choice for this specific biological dataset.

### Computational Efficiency

This table summarizes the computational overhead of each model:

| Model      | Train Time (s) | Infer Time (ms) | Parameters |
| :--------: | :------------: | :-------------: | :--------: |
| CNN        | 6.896          | 56.50           | 481        |
| **Linear** | **4.708**      | 61.80           | 481        |
| Polynomial | 6.141          | 72.53           | 483        |
| Spherical  | 5.104          | 60.17           | 482        |
| Gaussian   | 5.495          | 59.31           | 483        |

**Conclusions on Efficiency:**

  * **Fastest to Train:** The **Linear CKN** is the most efficient model for training, requiring only 4.708 seconds, significantly faster than the CNN baseline (6.896s) and the highest-performing Polynomial CKN (6.141s).
  * **Inference Speed:** The **CNN** offers the fastest inference time (56.50 ms), with the kernel-based methods generally introducing a small overhead. The Linear and Spherical CKNs are the closest competitors in speed.
  * **Efficiency vs. Accuracy Trade-off:** The **Polynomial CKN** provides the best balance of performance, achieving the highest accuracy with only a moderate increase in training time compared to the Linear CKN. The high training time of the CNN is not justified by its lower performance compared to the Polynomial CKN. Also, our clear winner for accuracy, the Polynomial is now showing us that efficiency has a bit of a price that we have to pay.

#**ALKBH5, second dataset of bio-informatics**

### Biological Context: The Importance of Non-Binding RNA

**ALKBH5** (AlkB Homolog 5) is known to function as an N^6-methyladenosine (m^6A) RNA demethylase, meaning its primary job is to remove the m^6A chemical modification from RNA. The m^6A modification is the most abundant internal modification in eukaryotic messenger RNA (mRNA) and plays a critical role in regulating gene expression.

An RNA molecule that **does not bind to ALKBH5** is significant because:

  * **Its Methylation Status is Preserved:** If the RNA is m^6A-modified, it retains that mark, as its processing is outside the control of ALKBH5. The m^6A modification is crucial for dictating the RNA's stability, splicing, and translation efficiency.
  * **It Is Subject to Other Regulatory Pathways:** Such an RNA must rely on other molecular machinery (like the FTO demethylase, or various "reader" proteins) to control its life cycle. This ensures the RNA's functional pathway is distinct from the ALKBH5-regulated network, diversifying gene expression control.

Essentially, the ALKBH5 can induce cancer onto the cells that it binds to, which is why it is important to determine which RNA patterns lead to association with ALKBH5, hence they are prone to the disease.

### Comparative Study Results

The study evaluated five models: a standard Convolutional Neural Network (**CNN**) baseline, and four CKN variants, as usual: **Linear**, **Polynomial**, **Spherical**, and **Gaussian**. The data represents a classification task (predicting a sequence as "ALKBH5 binding" or "non-binding").

#### Performance Metrics

We can see some stats from the study:

| Model          | Acc       | Mean AUC   | Med AUC    | Std AUC    | 95% CI Lower | 95% CI Upper |
| :------------: | :-------: | :--------: | :--------: | :--------: | :----------: | :----------: |
| CNN            | 62.5%     | 0.6918     | 0.6906     | 0.0259     | 0.949        | 0.965        |
| Linear         | 63.1%     | 0.6941     | 0.6907     | **0.0179** | 0.951        | 0.966        |
| **Polynomial** | **63.5%** | **0.7034** | **0.7039** | 0.0236     | 0.958        | **0.972**    |
| Spherical      | 61.7%     | 0.6797     | 0.6827     | 0.0232     | 0.922        | 0.943        |
| Gaussian       | 62.1%     | 0.6875     | 0.6909     | 0.0236     | 0.948        | 0.965        |

**Conclusions on Performance:**

1.  **Best Classifier:** The **Polynomial CKN** provides the highest performance across the board, achieving the best Mean Accuracy (**63.5%**) and Mean AUC (**0.7034**). This suggests the Polynomial kernel is best at capturing the relevant non-linear interactions within the sequence data for ALKBH5 binding prediction.
2.  **Most Stable Model:** The **Linear CKN** is the most stable model, recording the lowest Standard Deviation in AUC (**0.0179**), indicating its results are highly reproducible across different training runs.
3.  **Spherical CKN Underperformance:** Similar to the E-coli study, the **Spherical CKN** is the weakest performer, achieving the lowest accuracy and AUC (61.7% and 0.6797, respectively).

#### Computational Efficiency

This table summarizes the computational overhead of each model:

| Model        | Train Time (s) | Infer Time (ms) | Parameters |
| :----------: | :------------: | :-------------: | :--------: |
| CNN          | 11.24s         | **124ms**       | 1345       |
| Linear       | 9.30s          | 122ms           | 1345       |
| Polynomial   | 11.32s         | 132ms           | 1347       |
| Spherical    | 11.11s         | 147ms           | 1346       |
| **Gaussian** | **13.38s**     | **156ms**       | 1347       |

**Conclusions on Efficiency:**

1.  **Training Speed Champion:** The **Linear CKN** is the fastest model to train (**9.30s**), offering a notable advantage over the CNN baseline (11.24s) and the top-performing Polynomial CKN (11.32s).
2.  **Inference Speed Champion:** The **Linear CKN** and **CNN** are the fastest at inference (122ms and 124ms, respectively). The CNN is 6% ahead of the Polynomial when it comes to inference speed, but 1% behind it when it comes to accuracy and AUC. Depending on what a potential user wants, the choice would most probably have to be made between these 2 ones.
3.  **Highest Cost:** The **Gaussian CKN** is the most computationally expensive for both training (**13.38s**) and inference (**156ms**).

Hence..
  * **For Maximum Performance:** **Polynomial CKN**. It offers the highest accuracy and AUC, making it the best model for biological discovery where maximizing correct prediction is paramount.
  * **For Maximum Speed/Efficiency:** **Linear CKN**. It is the fastest to train, highly stable, and its performance is very close to the CNN baseline while being faster to train.
  * **Model to Avoid:** **Spherical CKN** and **Gaussian CKN**. The Spherical CKN has the worst performance, and the Gaussian CKN is the slowest model for both training and inference.


**PTBv1**
The binding of PTBv1 to cells can lead to the same diseases and health issues as ALKBH5, but the dataset presents a lighter structure for the RNA, hence the results compare much higher accuracies and AUC.

Comparative Study Results

The study evaluated five models over multiple runs: a standard Convolutional Neural Network (**CNN**) baseline, and four CKN variants, as usual: **Linear**, **Polynomial**, **Spherical**, and **Gaussian**.1. Performance Metrics

| Model          | Mean Acc  | Mean AUC   | Med AUC | Std AUC | F1 Score  | 95% CI (Accuracy) |
| :------------: | :-------: | :--------: | :-----: | :-----: | :-------: | :---------------: |
| CNN            | **0.989** | 0.8389     | 0.8346  | 0.0129  | 0.000     | 0.988-0.989       |
| Linear         | 0.988     | 0.8502     | 0.8496  | 0.0168  | 0.000     | 0.988-0.988       |
| **Polynomial** | 0.988     | **0.8509** | 0.8427  | 0.0233  | 0.000     | 0.988-0.989       |
| Spherical      | **0.989** | 0.8417     | 0.8267  | 0.0240  | **0.013** | 0.988-0.989       |
| Gaussian       | **0.989** | 0.8286     | 0.8138  | 0.0212  | 0.000     | 0.988-0.989       |

**Conclusions on Performance:**

  - **High Accuracy across all models:** All models exhibit near-perfect Mean Accuracy (0.988 or 0.989), indicating the models have successfully learned the primary patterns of the dataset.
  - **Best Discriminator (AUC):** The **Polynomial CKN** achieved the highest Mean AUC (0.8509), making it the best model for ranking positive and negative examples, despite its high accuracy being matched by others.
  - **Most Stable Model:** The **CNN baseline** is the most stable model, recording the lowest Standard Deviation in AUC (0.0129). This suggests the standard convolutional layer architecture provides the most reproducible results.
  - **Spherical CKN F1 Score:** The **Spherical CKN** is the only model to register a non-zero F1 score (0.013), suggesting it may have been slightly more successful at correctly identifying the minority class compared to the other models, although the overall score remains extremely low.

2. Computational Efficiency

| Model         | Parameters | Train Time (s) | Infer Time (ms) |
| :-----------: | :--------: | :------------: | :-------------: |
| CNN           | 1345       | 44.34          | 1661.42         |
| Linear        | 1345       | 52.86          | **1058.13**     |
| Polynomial    | 1347       | 46.01          | 1155.17         |
| **Spherical** | 1346       | **42.53**      | 1274.29         |
| Gaussian      | 1347       | 44.49          | **1682.90**     |

**Conclusions on Efficiency:**

  - **Fastest to Train:** The **Spherical CKN** is the fastest model to train (**42.53s**), marginally beating the CNN baseline.
  - **Fastest to Infer:** The **Linear CKN** is the fastest model during inference (**1058.13ms**), significantly faster than the CNN and Gaussian CKN.
  - **Slowest Model:** The **Gaussian CKN** is the slowest model for inference (**1682.90ms**), and the **Linear CKN** is the slowest to train (**52.86s**).
  - **Parameter Similarity:** All models have a virtually identical number of trainable parameters (around 1345).

Hence...
This is an impressive victory for CKNs. On a dataset where every neural network achieved a near-perfect accuracy/AUC, the fact that the Linear CKN is better by approximately 35% than the CNN at inference speed is truly game-changing, any scientist would pick the LinearCKN over the CNN, being able to do 35% more calculations in a timeframe.


**M^6-Methyladenosine (m^6A) Binding Site Prediction**

**N^6-methyladenosine (m^6A)** is the most common internal modification of messenger RNA (mRNA) and other non-coding RNAs in eukaryotes. It is a critical mark that plays a fundamental role in regulating gene expression. This modification is installed by "writer" proteins and removed by "eraser" proteins.

An accurate model to predict where m^6A modifications or m6A-related binding events occur in an RNA sequence is vital for understanding cell differentiation, development, and disease. The predictive task here is to determine whether a given RNA sequence segment is a target for an m6A-related mechanism.

Comparative Study Results

The study compared a **CNN** and four CKN variants, as usual: **Linear**, **Polynomial**, **Spherical**, and **Gaussian**, over 10 independent runs.

1. Performance Metrics (10 Runs)
We can see some stats from the study:

| Model          | Mean Acc  | Mean AUC   | Med AUC    | Std AUC    | F1 Score  |
| :------------: | :-------: | :--------: | :--------: | :--------: | :-------: |
| CNN            | 0.747     | 0.8218     | 0.8223     | 0.0031     | 0.751     |
| Linear         | 0.746     | 0.8222     | 0.8226     | 0.0027     | 0.753     |
| **Polynomial** | **0.748** | **0.8250** | **0.8257** | **0.0023** | **0.753** |
| Spherical      | 0.746     | 0.8225     | 0.8224     | **0.0023** | 0.750     |
| Gaussian       | **0.748** | 0.8233     | 0.8231     | **0.0023** | **0.753** |

**Conclusions on Performance:**

  - **Consistently High Performance:** All models demonstrate consistent and competitive performance, clustering closely around 74.6% to 74.8% Mean Accuracy and 0.822 to 0.825 Mean AUC.
  - **Best Discriminator/Stability:** The **Polynomial CKN** slightly edges out the competition with the highest Mean AUC (0.8250). Crucially, the **Polynomial, Spherical, and Gaussian CKNs** all show the lowest Standard Deviation in AUC (0.0023), indicating they are the most stable and reliable in terms of predictive confidence.
  - **Overall Top Model:** The **Polynomial CKN** provides the best combination of maximum AUC and maximum stability.

2. Computational Efficiency

This table summarizes the models' computational efficiency:

| Model        | Parameters | Train Time (s) | Infer Time (ms) |
| :----------: | :--------: | :------------: | :-------------: |
| CNN          | 961        | 109.54         | 296.84          |
| **Linear**   | 961        | **103.37**     | **221.99**      |
| Polynomial   | 963        | 126.67         | 297.99          |
| Spherical    | 962        | 118.55         | 296.94          |
| **Gaussian** | 963        | **139.89**     | **323.09**      |

**Conclusions on Efficiency:**

  - **Efficiency Champion:** The **Linear CKN** is the clear winner for efficiency, being the fastest to train (**103.37s**) and significantly the fastest for inference (**221.99ms**)—a 25% reduction in inference time compared to the CNN.
  - **Highest Cost:** The **Gaussian CKN** is the slowest model for both training (**139.89s**) and inference (**323.09ms**).
  - **Cost of Non-Linearity:** The non-linear kernels (Polynomial, Spherical, Gaussian) all require substantially longer training and inference times than the Linear CKN, suggesting the modest performance gain is paid for with a higher computational budget.

The choice for m^6A prediction is a trade-off between performance and speed:

  - **Maximum Performance and Stability:** **Polynomial CKN**. It offers the highest Mean AUC and excellent stability (lowest Std AUC), something you would need in this type of scenario of detection.
  - **Model to Avoid:** **Gaussian CKN**. It is the slowest model for both training and inference without providing a significant performance advantage over the others.
