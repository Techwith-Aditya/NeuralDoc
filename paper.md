# Review Paper

## Abstract
Convolutional Neural Networks (CNNs) are a dominant approach in image analysis, demonstrating efficacy in various computer vision tasks.  Their ability to extract hierarchical spatial features from images, combined with their translational invariance, makes them well-suited for tasks like image classification and object detection.  While traditional CNN architectures excel, challenges remain in computational efficiency and adaptability across diverse data types.  Recent studies explore dynamic architectures, such as Self-Expanding Neural Networks (SENN), adapting network size during training to enhance efficiency and prevent over-parameterization.  A common theme across the abstracts is the improved performance of CNNs for image-related tasks compared to traditional ANNs.  Furthermore, the use of deep learning frameworks for fusion of inertial and DVL measurements in AUV navigation, and benchmarking suites for RNA structure modeling are also highlighted as key areas of research.  Techniques for dealing with the challenges of limited dataset size, high dimensionality of image data, and potential overfitting are also addressed.  The importance of explicit considerations of cross-correlations in navigation filter systems is demonstrated in the context of inertial and DVL data fusion.  Finally, the abstracts also cover the increasing importance of methods for improving the computational efficiency of deep learning models.


## Introduction
This collection of papers investigates various aspects of convolutional neural networks (CNNs) and their applications in diverse fields, including image classification, structural biology, and autonomous underwater vehicle (AUV) navigation.  A common thread is the exploration of efficient and adaptable deep learning models for complex data.  The papers highlight the limitations of traditional CNN architectures in terms of computational complexity and adaptability to varying data complexities.  Furthermore, the need for robust models capable of handling potentially large datasets, such as the CIFAR-10 image dataset, or the challenges associated with predicting extreme events in complex systems, is also emphasized.

The research problem in several papers centers on optimizing CNN architectures for specific tasks.  This involves investigating the influence of different layer configurations, particularly in the initial convolutional layers, on model performance and convergence speed.  For example, the impact of including the first convolutional layer's goodness in the training process, and its effect on accuracy and convergence are explored.  A further problem addressed is the resource-intensive nature of neural architecture search (NAS), motivating the development of dynamically expanding architectures such as Self-Expanding Neural Networks (SENN) and their application to CNNs.  Other papers emphasize the development of effective methods for fusion of sensor data (inertial and DVL measurements in AUVs) and prediction of extreme events in complex systems (ship responses to extreme wave conditions).  Existing methods, such as Monte Carlo simulations or extrapolation using Weibull distributions, are recognized as computationally expensive or relying on strong prior assumptions, driving the investigation of alternative strategies, such as machine learning-based approaches (LSTMs and Gaussian Process Regression).

The overarching objectives of these studies are to develop more efficient, adaptable, and robust deep learning models, particularly CNNs. This involves refining CNN architectures to improve training speed, accuracy, and generalization capability. Methods explored include designing dynamic expansion mechanisms, integrating sensor fusion techniques, and adopting multi-fidelity approaches for improved efficiency in predicting extreme responses. Ultimately, the papers seek to address critical gaps in current deep learning methodology and broaden the practical application of deep learning models in diverse domains.


## Litrature Review
Deep learning, particularly convolutional neural networks (CNNs), has demonstrated significant potential for image classification tasks.  Ciresan et al. [1, 3, 4]  highlight the effectiveness of CNNs in various image classification scenarios, including handwritten character recognition and image classification.  Their studies consistently show that CNN architectures, employing multiple convolutional layers,  outperform traditional methods in accuracy and robustness, underscoring the importance of feature extraction and network depth for optimal performance.  This work emphasizes the adaptability of CNNs to different tasks.  Further research from Ciresan et al. [2] demonstrates CNN applicability in medical image analysis, specifically mitosis detection in breast cancer histology.   These studies collectively showcase the versatility of CNNs across diverse image recognition problems.

Early research on image processing with neural networks, including work by Egmont-Petersen et al. [5] and LeCun et al. [12, 40], laid the groundwork for subsequent developments.  LeCun et al. [13] explored the application of backpropagation to handwritten digit recognition, demonstrating a crucial early success in neural network-based image processing.   Nebauer [14]  evaluated the performance of CNNs for visual recognition, contributing to the understanding of their capabilities. Simard et al. [15] further explored best practices for applying CNNs to document analysis, refining the application techniques.  Farabet et al. [6] explored hardware acceleration for CNNs, which is crucial for real-time processing.


The application of CNNs has expanded beyond traditional image recognition tasks.  Ji et al. [9] extended CNN architectures to three-dimensional data, demonstrating their use for human action recognition.  Similarly, Karpathy et al. [10] employed CNNs for large-scale video classification, showcasing their applicability in complex, real-world scenarios. Krizhevsky et al. [11] revolutionized image classification with ImageNet, showcasing the state-of-the-art capabilities of deep convolutional networks.

Other research has investigated and refined different aspects of CNN architecture and training.  Hinton et al. [8] proposed techniques for improving neural networks by preventing co-adaptation of feature detectors, contributing to more robust models.  Zeiler and Fergus [20, 21] explored techniques like stochastic pooling for regularization and visualization within CNNs, aiding in better model understanding.  These improvements in training and architecture enhance the reliability and generalizability of CNNs.   Srivastava [16] introduced dropout, which helps in preventing overfitting, a critical factor in training large neural networks.  While this work focused on the specific technique of dropout, it contributes to the overall advancement of CNNs.

Recent research, represented by papers such as Cheng et al [22,23],  highlights the continuous evolution of CNN applications, now extending to visual defect detection and estimating energy and time usage in 3D printing. A significant trend is the application of deep learning to various aspects of autonomous underwater vehicle (AUV) navigation, encompassing topics like velocity log optimization, navigation in challenging environments (e.g., no Doppler velocity log), and velocity forecasting [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106].  Many of these works are explicitly concerned with developing data-driven approaches to enhance AUV capabilities.


## Methodology
The methodologies presented in the provided papers encompass diverse approaches to machine learning, particularly in image analysis and data modeling.  Several papers focus on convolutional neural networks (CNNs), exploring their architecture and application in image analysis tasks.  One paper highlights the simplification of CNN architecture achievable by exploiting the characteristics of specific input types. Another paper details the implementation of a feedforward algorithm (FF) for labeled datasets.  A third paper, using a Python-based framework, describes a modular approach for programmatic access to RNA 3D structure modeling datasets, emphasizing reproducibility.  This involves automated download, representation selection (e.g., Pytorch Geometric graphs), and the option for end-to-end data processing.

A distinct methodology is presented that details the construction of a nonlinear model, specifying the generation of the matrix B using singular value decomposition (SVD) and different multivariate distributions for input vectors (multivariate normal, hyperbolic, and t-distribution). This model also considers varying link functions, including linear and nonlinear functions (e.g., sine, cubic).

Furthermore, one paper introduces a scattering transform as a simplified model for understanding CNN operations, linking feature transformations to wavelet transforms.  Another paper focuses on criteria for dynamically expanding CNN architectures.  Finally, a methodology for cross-correlation-aware deep inertial navigation system/dynamic velocity logger (INS/DVL) fusion is described, integrating data-driven and model-based techniques. This approach utilizes a deep learning-based velocity estimation within an error-state Kalman filter, explicitly modelling cross-covariances to address limitations of traditional Kalman filtering approaches, which often assume uncorrelated noise.  Results demonstrate improved accuracy, consistency, and theoretical grounding compared to model-based least-squares methods.  Methodologies show a mix of mathematical modeling, data manipulation through established Python libraries (Pytorch Geometric), and neural network implementations.


## REFERENCES

This collection of papers details various methods and techniques related to autonomous underwater vehicle (AUV) navigation, including Doppler velocity log (DVL) calibration and forecasting in challenging conditions.  Different papers explore using deep learning, inertial navigation, and Kalman filtering for enhancing DVL measurements and improving AUV navigation in DVL-denied environments.  The references cover theoretical frameworks, practical applications, and optimization approaches.  Methods like LSTM networks and convolutional neural networks (CNNs) are discussed for estimating and predicting critical parameters, such as velocity and acceleration.  The papers also touch on existing datasets and benchmarks for evaluating these methods.


## 2.1 Long Short-Term Memory (LSTM)

LSTM networks, a type of recurrent neural network, are presented as a powerful tool for time series data analysis.  They can model both short-term and long-term dependencies in data, making them effective for forecasting and prediction tasks related to underwater navigation.  The architecture of an LSTM cell, including input, output, and memory components, is described along with the mathematical equations representing the cell's operations.  The paper discusses training procedures, hyperparameter selection, and the crucial role of time resolution in shaping the LSTM network's performance and accuracy.


## 1.5 Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are described as a class of neural networks well-suited for image-based data, like those encountered in visual defect detection and energy/time estimation in 3D printing.  The hierarchical architecture of CNNs, using convolutional filters and non-linearities, is explained.  The optimization process, relying on stochastic gradient descent and backpropagation, is detailed. The paper emphasizes the effectiveness of CNNs in image processing tasks, highlighting their ability to extract features from the input signal.


## 1.6 A mathematical framework for CNNs

A mathematical framework for analyzing CNN properties, based on wavelet scattering, is introduced.  This theory aims to understand how CNNs compute invariants and separate variations at different scales.  This framework is seen as a foundational step towards a broader understanding of CNNs and is presented as a key concept in the analysis.


## 5 General Convolutional Neural Network Architectures

The scattering transform, a simplified representation of CNNs, is analyzed. The transform's inherent limitations, including high variance and information loss from single-channel convolutions, are highlighted. The paper suggests that a more comprehensive framework, considering channel combinations and adaptive local symmetries, is required to analyze general CNN architectures effectively. This framework is beyond the scope of the current paper.


## 2 The need for wavelets

The paper explains why wavelets are essential for analyzing local variations in data.  The Fourier transform, while powerful for frequency analysis, lacks the capacity to capture local information. Windowed Fourier transforms are presented as a potential solution, but the limitations of their approach are also identified. This discussion highlights the importance of wavelet transformations for analyzing signals where local information is crucial.


## Keywords

The provided text lists keywords related to machine learning, neural networks, and computational efficiency in the context of computer vision and pattern recognition. The listed terms provide a summary of the topic's area of focus.


## REFERENCES

This collection of papers presents a curated list of publications related to RNA (ribonucleic acid) and protein-related topics.  The papers span various methodologies, including computational approaches, machine learning, and experimental research.  These references cover topics such as 3D structure prediction, sequence design, RNA-small molecule binding, protein-ligand interactions, and various datasets/benchmarks for evaluating models.


## II.3 Structures and Training Details of Neural Networks

The structure and training details of a neural network are described. The network uses a specific function, ReLU (Rectified Linear Unit) and operates within a given framework. The training employs a batch-training strategy using an Adam optimizer for a specified number of epochs, with the batch size being a fraction of the total sample size.  These details relate to the implementation and optimization procedure.


## I.4 Reduced Rank Estimator, Neural Network Estimator and Ours

This section details the methods for estimating latent spaces and their relationship to neural networks. Methods involving reduced rank regression and neural networks are compared and contrasted. The paper demonstrates how reduced rank regression provides a closed-form solution under certain conditions.  Neural network estimators are presented as a way to estimate similar parameters, relying on minimizing a loss function.  The differences in these approaches and the potential benefits of the proposed method are discussed, along with how different types of data distributions affect estimation methods.


## III.1.3 Structure and Training Details of Autoencoder

The structure of an autoencoder (AE) is outlined with specific layer types and their dimensions.  The training procedure focuses on minimizing the reconstruction squared error loss using a batch-training strategy and Adam optimizer for a specified number of epochs and batch size. The method and optimization are detailed.


## 2 CNN architecture

The paper outlines the architectural components of CNNs, emphasizing the convolutional layers' role in processing image data. Convolutional layers use learnable kernels for extracting features from the input signal.  The organization of neurons, receptive field size, and hyperparameters (depth, stride, padding) governing convolutional layers' output are elaborated. The discussion touches upon parameter sharing for reducing complexity.


## Results
Convolutional neural networks (CNNs) demonstrate strong performance across various image classification tasks, including handwritten character recognition, mitosis detection in breast cancer histology, and image processing.  Studies highlight the effectiveness of CNN committees and multi-column architectures for improved accuracy.  Significant improvements in image classification were reported using deep CNNs on the ImageNet dataset.  The application of CNNs extends to human action recognition and large-scale video classification.  Furthermore, the review literature underscores the importance of CNNs in visual document analysis and their potential for hardware acceleration.  The analysis of different CNN configurations and hyperparameters reveals optimal settings for achieving peak performance.  Techniques like stochastic pooling are shown to enhance CNN regularization. Qualitative analysis, such as Class Activation Maps, is also utilized in the context of explainable AI for better understanding.  Comparative studies using datasets like CIFAR-10, alongside various performance metrics (e.g., validation accuracy, number of parameters), showcase the tradeoffs involved in CNN design.  Other research demonstrates the applicability of CNNs to tasks such as pedestrian detection and object detection.  The results from the analysis of RNA-related tasks using recurrent graph convolutional networks (RGCNs) reveal varying performance across different tasks, with specific models achieving F1-scores, AUC values, and global balanced accuracies that vary depending on the specific task and hyperparameters.  The comparison of these methods to previously published results underscores the potential for improvement and the impact of different architectures and training protocols.  The papers also discuss various deep learning architectures and methodologies, with particular focus on their ability to learn representations, and methods for gradient estimation in implicit models.  Finally,  the literature review provides a broad overview of the field, summarizing  key advances in deep learning, such as the role of deep scattering spectrum and wavelets, including detailed analyses of methods like support vector networks, along with specific examples of applications within the domain of autonomous underwater vehicle (AUV) navigation, exploring various factors, such as Doppler Velocity Log calibration.


![Figure](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\cnn2\image94-page3.png)

**Figure Caption:** Figure 1: Architecture of a Convolutional Neural Network (CNN), as described by LeCun et al. [7].  This depicts the fundamental structure of a CNN, illustrating the hierarchical feature extraction process.


## Conclusion
Convolutional Neural Networks (CNNs) demonstrate significant potential in image analysis, particularly in biomedical imaging where they offer an improvement over subjective visual inspection.  The papers highlight CNNs' ability to extract spatial information, leading to more objective and efficient analysis.  Simplified architectures, like those based on scattering transforms, can be effective, offering a pathway to a deeper theoretical understanding of CNN operations.  Moreover, improved architectures, characterized by deeper convolutional layers, batch normalization, and dropout, can enhance classification accuracy and generalization, as evidenced in experiments on the CIFAR-10 dataset.  These findings suggest a trend towards more sophisticated CNN designs that improve upon earlier models in both accuracy and theoretical understanding.

Practical implications span various domains, including biomedical diagnostics, where objective analysis reduces human error and workload, and computer vision generally, where improved accuracy and generalization enable more effective image classification systems.  A key implication is the potential for explainable AI tools like class activation maps, which provide insight into the model's reasoning.

While the studies demonstrate the effectiveness of CNNs, limitations exist.  The choice of hyperparameters can influence results, and the potential for certain optimizations, like symmetry enforcement of the loss function or selection of “hard” labels, has not been fully explored.  Moreover, the scalability of specific architectures, especially with large and complex datasets, still needs further investigation.  The papers also highlight the potential to surpass traditional backpropagation in specific applications, particularly in neuromorphic hardware and unsupervised learning.  Further research into the interplay between FF training (forward-forward) and biological neuronal systems is also indicated.  Future research should explore the generalization capabilities of CNNs with more complex datasets and address potential optimizations.  Additional investigation into the theoretical underpinnings of CNNs, especially for their application to the prediction of functions of RNA structures and protein design, is equally crucial.  This includes developing and evaluating benchmarking suites for specific biological tasks, thereby promoting comparable and reproducible model development across domains.


## References
[1]  Ciresan, D., Meier, U., & Schmidhuber, J. (2012). Multi-column deep neural networks for image classification. In *Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on* (pp. 3642–3649). IEEE.
[2]  Ciresan, D. C., Giusti, A., Gambardella, L. M., & Schmidhuber, J. (2013). Mitosis detection in breast cancer histology images with deep neural networks. In *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2013* (pp. 411–418). Springer.
[3]  Ciresan, D. C., Meier, U., Masci, J., Gambardella, L. M., & Schmidhuber, J. (2011). Flexible, high performance convolutional neural networks for image classification. In *IJCAI Proceedings-International Joint Conference on Artificial Intelligence*, *22*(1), 1237.
[4]  Ciresan, D. C., Meier, U., Gambardella, L. M., & Schmidhuber, J. (2011). Convolutional neural network committees for handwritten character classification. In *Document Analysis and Recognition (ICDAR), 2011 International Conference on* (pp. 1135–1139). IEEE.
[5]  Egmont-Petersen, M., de Ridder, D., & Handels, H. (2002). Image processing with neural networks: a review. *Pattern Recognition*, *35*(10), 2279–2301.
[6]  Farabet, C., Martini, B., Akselrod, P., Talay, S., LeCun, Y., & Culurciello, E. (2010). Hardware accelerated convolutional neural networks for synthetic vision systems. In *Circuits and Systems (ISCAS), Proceedings of 2010 IEEE International Symposium on* (pp. 257–260). IEEE.
[7]  Hinton, G. (2010). A practical guide to training restricted boltzmann machines. *Momentum*, *9*(1), 926.
[8]  Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of feature detectors. *arXiv preprint arXiv:1207.0580*.
[9]  Ji, S., Xu, W., Yang, M., & Yu, K. (2013). 3D convolutional neural networks for human action recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *35*(1), 221–231.
[10] Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., & Fei-Fei, L. (2014). Large-scale video classification with convolutional neural networks. In *Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on* (pp. 1725–1732). IEEE.
[11]  Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems* (pp. 1097–1105).
[12] LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to handwritten zip code recognition. *Neural computation*, *1*(4), 541–551.
[13] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278–2324.
[14] Nebauer, C. (1998). Evaluation of convolutional neural networks for visual recognition. *IEEE Transactions on Neural Networks*, *9*(4), 685–696.
[15] Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best practices for convolutional neural networks applied to visual document analysis. In *null* (p. 958). IEEE.
[16] Srivastava, N. (2013). *Improving neural networks with dropout*. PhD thesis, University of Toronto.
[17] Szarvas, M., Yoshizawa, A., Yamamoto, M., & Ogata, J. (2005). Pedestrian detection with convolutional neural networks. In *Intelligent Vehicles Symposium, 2005. Proceedings. IEEE* (pp. 224–229). IEEE.
[18] Szegedy, C., Toshev, A., & Erhan, D. (2013). Deep neural networks for object detection. In *Advances in Neural Information Processing Systems*.
[19] Tivive, F. H. C., & Bouzerdoum, A. (2003). A new class of convolutional neural networks (siconnets) and their application of face detection. In *Neural Networks, 2003. Proceedings of the International Joint Conference on*, *3*, 2157–2162. IEEE.
[20] Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization of deep convolutional neural networks. *arXiv preprint arXiv:1301.3557*.
[21] Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In *Computer Vision–ECCV 2014* (pp. 818–833). Springer.
[22]  Cheng, Q., Qu, S., & Lee, J. (2022). 72-3: Deep Learning Based Visual Defect Detection in Noisy and Imbalanced Data. *SID Symposium Digest of Technical Papers*, *53*(1), 971–974.
[23] Cheng, Q., Zhang, C., & Shen, X. (2022). Estimation of Energy and Time Usage in 3D Printing With Multimodal Neural Network. *2022 4th International Conference on Frontiers Technology of Information and Computer (ICFTIC)*, 900–903.
[24] Xing, J. (2024). Enhancing Link Prediction with Fuzzy Graph Attention Networks and Dynamic Negative Sampling. *arXiv preprint arXiv:2411.07482*.
[25] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. *arXiv preprint arXiv:1710.10903*.
[26] Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in neural information processing systems 30*.
[27] Xing, J., Gao, C., & Zhou, J. (2022). Weighted fuzzy rough sets-based tri-training and its application to medical diagnosis. *Applied Soft Computing*, *124*, 109025.
[28] Gao, C., Zhou, J., Xing, J., & Yue, X. (2022). Parameterized maximum-entropy-based three-way approximate attribute reduction. *International Journal of Approximate Reasoning*, *151*, 85–100.
[29] Xing, J., Xing, R., & Sun, Y. (2024). FGATT: A Robust Framework for Wireless Data Imputation Using Fuzzy Graph Attention Networks and Transformer Encoders. *arXiv preprint arXiv:2412.01979*.
[30] Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PloS one*, *13*(5), e0196391.
[31] Xing, J., Luo, D., Cheng, Q., Xue, C., & Xing, R. (2024). Multi-view Fuzzy Graph Attention Networks for Enhanced Graph Learning. *arXiv preprint arXiv:2412.17271*.
[32] Heigold, G., Moreno, I. L., Bengio, S., & Shazeer, N. (2016). End-to-End Text-Dependent Speaker Verification. In *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 5115–5119).
[33] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, *9*(8), 1735–1780.
[34] Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural computation*, *12*(10), 2451–2471.
[35] Andén, J., & Mallat, S. (2014). Deep scattering spectrum. *IEEE Transactions on Signal Processing*, *62*(16), 4114–4128.
[36] Bruna, J., & Mallat, S. (2013). Invariant scattering convolution networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *35*(8), 1872–1886.
[37] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine learning*, *20*(3), 273–297.
[38] Bruna Estrach, J. (n.d.). Scattering representations for recognition.
[39] Kaiser, G. (1994). *A friendly guide to wavelets*.
[40] LeCun, Y., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., Jackel, L. D., & Boser, B. (1990). Handwritten digit recognition with a back-propagation network. In *Advances in neural information processing systems*.
[41] Mallat, S. (2012). Group invariant scattering. *Communications on Pure and Applied Mathematics*, *65*(10), 1331–1398.
[42] Mallat, S. (2016). Understanding deep convolutional networks. *arXiv preprint arXiv:1601.04920*.
[43] Balasubramanian, K., Fan, J., & Yang, Z. (2018). Tensor methods for additive index models under discordance and heterogeneity. *arXiv preprint arXiv:1807.06693*.
[44] Bauer, B., & Kohler, M. (2019). On deep learning as a remedy for the curse of dimensionality in nonparametric regression. *The Annals of Statistics*, *47*(6), 2261–2285.
[45] Bauer, F., Pereverzev, S., & Rosasco, L. (2007). On regularization algorithms in learning theory. *Journal of Complexity*, *23*(1), 52–72.
[46] Breymann, W., & Lüthi, D. (2013). ghyp: A package on generalized hyperbolic distributions. *Manual for R Package ghyp*.
[47] Candès, E. J., Li, X., Ma, Y., & Wright, J. (2009). Robust principal component analysis? *arXiv preprint arXiv:0912.3599*.
[48] Zou, C., & Zhang, W. (2022). Estimation of low rank high-dimensional multivariate linear models for multi-response data. *Journal of the American Statistical Association*, *117*(537), 693–703.
[49] Chen, K., Dong, H., & Chan, K.-S. (2012). Reduced rank regression via adaptive nuclear norm penalization. *arXiv preprint arXiv:1201.0381*.
[50] Chen, X., Zou, C., & Cook, R. D. (2010). Coordinate-independent sparse sufficient dimension reduction and variable selection. *The Annals of Statistics*, *38*(6), 3696–3723.
[51] Damian, A., Lee, J., & Soltanolkotabi, M. (2022). Neural networks can learn representations with gradient descent. In *Conference on Learning Theory*.
[52] Friedman, J. H., & Stuetzle, W. (1981). Projection pursuit regression. *Journal of the American Statistical Association*, *76*(376), 817–823.
[53] Zou, H., & Tibshirani, R. (2006). Sparse principal component analysis. *Journal of Computational and Graphical Statistics*, *15*(2), 265–286.
[54] Hyvärinen, A., & Dayan, P. (2005). Estimation of non-normalized statistical models by score matching. *Journal of Machine Learning Research*, *6*(1), 695–709.
[55] Janzamin, M., Sedghi, H., & Anandkumar, A. (2014). Score function features for discriminative learning: Matrix and tensor framework. *arXiv preprint arXiv:1412.2863*.
[56] Kobak, D., Bernaerts, Y., Weis, M. A., Scala, F., Tolias, A. S., & Berens, P. (2021). Sparse reduced-rank regression for exploratory visualisation of paired multivariate data. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, *70*(4), 980–1000.
[57] Lee, W., & Liu, Y. (2012). Simultaneous multiple response regression and inverse covariance matrix estimation via penalized gaussian maximum likelihood. *Journal of Multivariate Analysis*, *111*, 241–255.
[58] Li, K.-C. (1991). Sliced inverse regression for dimension reduction. *Journal of the American Statistical Association*, *86*(414), 316–327.
[59] Li, K.-C. (1992). On principal hessian directions for data visualization and dimension reduction: Another application of stein's lemma. *Journal of the American Statistical Association*, *87*(418), 1025–1039.
[60] Li, K.-C., & Duan, N. (1989). Regression analysis under link violation. *The Annals of Statistics*, *17*(3), 1009–1052.
[61] Li, Y., & Turner, R. E. (2017). Gradient estimators for implicit models. *arXiv preprint arXiv:1705.07107*.
[62] Lu, Z., Monteiro, R. D., & Yuan, M. (2012). Convex optimization methods for dimension reduction and coefficient estimation in multivariate linear regression. *Mathematical Programming*, *131*(1-3), 163–194.
[63] Mnih, A., Heess, N., Graves, A., et al. (2014). Recurrent models of visual attention. In *NIPS*.
[64] Mnih, V., Heess, N., Graves, A., et al. (2014). Recurrent models of visual attention. In *Advances in neural information processing systems*.
[65] Mnih, V., Heess, N., Graves, A., et al. (2014). Recurrent models of visual attention. In *Advances in neural information processing systems*.
[66] Makhzani, A., & Frey, B. (2013). K-sparse autoencoders. *arXiv preprint arXiv:1312.5663*.
[67] Meng, C., Song, Y., Li, W., & Ermon, S. (2021). Estimating high order gradients of the data distribution by denoising. *Advances in Neural Information Processing Systems*, *34*, 25359–25369.
[68] Mousavi-Hosseini, A., Park, S., Girotti, M., Mitliagkas, I., & Erdogdu, M. A. (2022). Neural networks efficiently learn low-dimensional representations with SGD. *arXiv preprint arXiv:2209.14863*.
[69] Mukherjee, A., & Zhu, J. (2011). Reduced rank ridge regression and its kernel extensions. *Statistical analysis and data mining: the ASA data science journal*, *4*(6), 612–622.
[70] O'Rourke, S., Vu, V., & Wang, K. (2018). Random perturbation of low rank matrices: Improving classical bounds. *Linear Algebra and its Applications*, *540*, 26–59.
[71] Pearson, K. (1901). Liii. on lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin philosophical magazine and journal of science*, *2*(11), 559–572.
[72] Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). Contractive auto-encoders: Explicit invariance during feature extraction. In *Proceedings of the 28th International Conference on Machine Learning*.
[73] Scala, F., Kobak, D., Bernabucci, M., Bernaerts, Y., Cadwell, C. R., Castro, J. R., Hartmanis, L., Jiang, X., Laturnus, S., Miranda, E., et al. (2021). Phenotypic variation of transcriptomic cell types in mouse motor cortex. *Nature*, *598*(7878), 144–150.
[74] Shi, J., Sun, S., & Zhu, J. (2018). A spectral approach to gradient estimation for implicit distributions. In *International Conference on Machine Learning*.
[75] Simon, N., Friedman, J., & Hastie, T. (2013). A blockwise descent algorithm for group-penalized multiresponse and multinomial regression. *arXiv preprint arXiv:1311.6529*.
[76] Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in neural information processing systems*, *32*.
[77] Song, Y., Garg, S., Shi, J., & Ermon, S. (2020). Sliced score matching: A scalable approach to density and score estimation. In *Uncertainty in Artificial Intelligence*.
[78] Strathmann, H., Sejdinovic, D., Livingstone, S., Szabo, Z., & Gretton, A. (2015). Gradient-free hamiltonian monte carlo with efficient kernel exponential families. *Advances in Neural Information Processing Systems*, *28*, 955–963.
[79] Tan, K. M., Wang, Z., Zhang, T., Liu, H., & Cook, R. D. (2018). A convex formulation for high-dimensional sparse sliced inverse regression. *Biometrika*, *105*(4), 769–782.
[80] Vershynin, R. (2018a). *High-dimensional probability: An introduction with applications in data science*, Vol. 47. Cambridge university press.
[81] Vershynin, R. (2018b). *High-dimensional probability: An introduction with applications in data science*, Vol. 47. Cambridge university press.
[82] Vincent, P. (2011). A connection between score matching and denoising autoencoders. *Neural computation*, *23*(6), 1661–1674.
[83] Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th International Conference on Machine Learning*.
[84] Wang, W., Liang, Y., & Xing, E. (2013). Block regularized lasso for multivariate multi-response linear regression. In *Artificial intelligence and statistics*.
[85] Xu, X. (2020). On the perturbation of the Moore–Penrose inverse of a matrix. *Applied Mathematics and Computation*, *374*, 124920.
[86] Yang, Z., Balasubramanian, K., & Liu, H. (2017a). High-dimensional non-Gaussian single index models via thresholded score function estimation. In *Proceedings of the 34th International Conference on Machine Learning*, *70*.
[87] Yang, Z., Balasubramanian, K., Wang, Z., & Liu, H. (2017b). Learning non-Gaussian multi-index model via second-order Stein's method. *Advances in Neural Information Processing Systems*, *30*, 6097–6106.
[88] Yu, Y., Wang, T., & Samworth, R. J. (2015). A useful variant of the Davis–Kahan theorem for statisticians. *Biometrika*, *102*(2), 315–323.
[89] Yuan, M., Ekici, A., Lu, Z., & Monteiro, R. (2007). Dimension reduction and coefficient estimation in multivariate linear regression. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, *69*(2), 329–346.
[90] Zhou, Y., Shi, J., & Zhu, J. (2020). Nonparametric score estimators. In *International Conference on Machine Learning*.
[91] Nicholson, J., & Healey, A. (2008). The present state of autonomous underwater vehicle (AUV) applications and technologies. *Marine Technology Society Journal*, *42*(1), 44–51.
[92] Griffiths, G. (2002). *Technology and applications of autonomous underwater vehicles*, Vol. 2. CRC Press.
[93] Miller, P. A., Farrell, J. A., Zhao, Y., & Djapic, V. (2010). Autonomous underwater vehicle navigation. *IEEE Journal of Oceanic Engineering*, *35*(3), 663–678.
[94] Rudolph, D., & Wilson, T. A. (2012). Doppler velocity log theory and preliminary considerations for design and construction. In *2012 Proceedings of IEEE Southeastcon* (pp. 1–7). IEEE.
[95] Cohen, N., & Klein, I. (2024). Inertial navigation meets deep learning: A survey of current trends and future directions. *Results in Engineering*, *103565*.
[96] Zhang, F., Zhao, S., Li, L., & Cao, C. (2025). Underwater DVL Optimization Network (UDON): A Learning-Based DVL Velocity Optimizing Method for Underwater Navigation. *Drones*, *9*(1), 56.
[97] Liu, P., Wang, B., Li, G., Hou, D., Zhu, Z., & Wang, Z. (2022). Sins/dvl integrated navigation method with current compensation using rbf neural network. *IEEE Sensors Journal*, *22*(14), 14366–14377.
[98] Topini, E., Fanelli, F., Topini, A., Pebody, M., Ridolfi, A., Phillips, A. B., & Allotta, B. (2023). An experimental comparison of Deep Learning strategies for AUV navigation in DVL-denied environments. *Ocean Engineering*, *274*, 114034.
[99] Makam, R., Pramuk, M., Thomas, S., & Sundaram, S. (2024). Spectrally Normalized Memory Neuron Network Based Navigation for Autonomous Underwater Vehicles in DVL-Denied Environment. In *OCEANS 2024-Singapore* (pp. 1–6). IEEE.
[100] Yampolsky, Z., & Klein, I. (2025). DCNet: A data-driven framework for DVL calibration. *Applied Ocean Research*, *158*, 104525.
[101] Yona, M., & Klein, I. (2024). MissBeamNet: Learning missing Doppler velocity log beam measurements. *Neural Computing and Applications*, *36*(9), 4947–4958.
[102] Cohen, N., & Klein, I. (2023). Set-transformer BeamsNet for AUV velocity forecasting in complete DVL outage scenarios. In *2023 IEEE Underwater Technology (UT)* (pp. 1–6). IEEE.
[103] Cohen, N., & Klein, I. (2022). BeamsNet: A data-driven approach enhancing Doppler velocity log measurements for autonomous underwater vehicle navigation. *Engineering Applications of Artificial Intelligence*, *114*, 105216.
[104] Cohen, N., & Klein, I. (2025). Adaptive Kalman-Informed Transformer. *Engineering Applications of Artificial Intelligence*, *146*, 110221.
[105] Levy, A., & Klein, I. (2025). Adaptive Neural Unscented Kalman Filter. *arXiv preprint arXiv:2503.05490*.
[106] Stolero, Y., & Klein, I. (2025). AUV Acceleration Prediction Using DVL and Deep Learning. *arXiv preprint arXiv:2503.16573*.
[107] Simon, D. (2006). *Optimal state estimation: Kalman, H infinity, and nonlinear approaches*. John Wiley & Sons.
[108] Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2004). *Estimation with applications to tracking and navigation: theory algorithms and software*. John Wiley & Sons.
[109] Groves, P. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition*. GNSS/GPS, Artech House.
[110] Farrell, J. (2008). *Aided navigation: GPS with high rate sensors*. McGraw-Hill, Inc.
[111] Alford, L. (2008). Estimating Extreme Responses Using a Non-Uniform Phase Distribution. *University of Michigan, Ann Arbor, MI, USA*.
[112] Bertram, V. (2011). *Practical Ship Hydrodynamics, 2nd edition*. Butterworth-Heinemann.
[113] Bishop, R. C., Belknap, W., Turner, C., Simon, B., & Kim, J. H. (2005). Parametric Investigation on the Influence of GM, Roll Damping, and Above-Water Form on the Roll Response of Model 5613. *Naval Surface Warfare Center Carderock Division, West Bethesda, MD, USA, Technical Report 50-TR-2005/027*.
[114] Guth, S., & Sapsis, T. (2023). Analytical and Computational Methods for Non-Gaussian Reliability Analysis of Nonlinear Systems Operating in Stochastic Environments. *Massachusetts Institute of Technology, Cambridge, MA, USA*.
[115] Kim, D-H. (2012). Design Loads Generator: Estimation of Extreme Environmental Loadings for Ship and Offshore Applications. *University of Michigan, Ann Arbor, MI, USA*.
[116] Levine, M. D., Edwards, S. J., Howard, D., Weems, K., Sapsis, T., & Pipiras, V. (2024). Multi-Fidelity Data-Adaptive Autonomous Seakeeping. *Ocean Engineering*, *292*.
[117] Levine, M. D., Belenky, V., & Weems, K. M. (2021). Method for Automated Safe Seakeeping Guidance. *Proceedings of the 1st International Conference on the Stability and Safety of Ships and Ocean Vehicles, 2021, Glasgow, UK*.
[118] Lin, W., & Yue, D. (1990). Numerical Solutions for Large Amplitude Ship Motions in the Time-Domain. *Proceedings of the 18th Symposium on Naval Hydrodynamics, 1990, Ann Arbor, Michigan, USA*.
[119] Longuet-Higgins, M. S. (1984). Statistical Properties of Wave Groups in a Random Sea State. *Philosophical Transactions of the Royal Society of London A: Mathematical, Physical and Engineering Sciences*, *312*(1521), 219–250.
[120] Reed, A. M. (2021). Predicting Extreme Loads and the Processes for Predicting Them Efficiently. *Proceedings of the 1st International Conference on the Stability and Safety of Ships and Ocean Vehicles, Glasgow, UK 2021*.
[121] Shin, Y. S., Belenky, V., Lin, W.-M., & Weems, K. M. (2003). Nonlinear Time Domain Simulation Technology for Seakeeping and Wave-Load Analysis for Modern Ship Design. *SNAME Transactions*, *111*, 557–589.
[122] Wan, Z. Y., Vlachas, P., Koumoutsakos, P., & Sapsis, T. (2018). Data-assisted Reduced-Order Modeling of Extreme Events in Complex Dynamical Systems. *PLOS ONE*, *13*(5).
[123] Weems, K., & Wundrow, D. (2013). Hybrid Models for the Fast Time-Domain Simulation of Stability Failures in Irregular Waves with Volume based Calculations for Froude-Krylov and Hydrostatic Forces. *13th International Ship Stability Workshop, Brest, France 2013*.


