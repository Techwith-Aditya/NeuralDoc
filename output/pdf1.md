# Enhancing Underwater Navigation through Cross-Correlation-Aware Deep INS/DVL Fusion

Nadav Cohen* and Itzik Klein  
The Hatter Department of Marine Technologies,  
Charney School of Marine Sciences,  
University of Haifa  
Haifa, Israel

## Abstract

The accurate navigation of autonomous underwater vehicles critically depends on the precision of Doppler velocity log (DVL) velocity measurements. Recent advancements in deep learning have demonstrated significant potential in improving DVL outputs by leveraging spatiotemporal dependencies across multiple sensor modalities. However, integrating these estimates into model-based filters, such as the extended Kalman filter, introduces statistical inconsistencies, most notably, cross-correlations between process and measurement noise. This paper addresses this challenge by proposing a cross-correlation-aware deep INS/DVL fusion framework. Building upon BeamsNet, a convolutional neural network designed to estimate AUV velocity using DVL and inertial data, we integrate its output into a navigation filter that explicitly accounts for the cross-correlation induced between the noise sources. This approach improves filter consistency and better reflects the underlying sensor error structure. Evaluated on two real-world underwater trajectories, the proposed method outperforms both least squares and cross-correlation-neglecting approaches in terms of state uncertainty. Notably, improvements exceed 10% in velocity and misalignment angle confidence metrics. Beyond demonstrating empirical performance, this framework provides a theoretically principled mechanism for embedding deep learning outputs within stochastic filters.

*Index Terms*—Autonomous Underwater Vehicle, Underwater Navigation, Deep Learning, Kalman Filtering, Sensor Fusion

## I. Introduction

Autonomous underwater vehicles (AUVs) are valuable tools for exploring and operating underwater. They are used in a wide range of applications, including seafloor mapping, underwater construction and inspection, environmental monitoring, and studying marine life [1], [2]. To accomplish their task accurately and robustly, precise navigation is required. Commonly, this is achieved by fusing inertial sensors with a Doppler velocity log (DVL) [3]. The DVL is an acoustic sensor that utilizes the Doppler effect. This device transmits four acoustic beams to the seafloor, which are then reflected back. Based on the frequency shift, the DVL calculates the velocity of each beam and then estimates the velocity vector of the AUV [4].

Data-driven methods have been employed in AUV-related tasks with promising outcomes [5]–[9]. DCNet, a data-driven framework that utilizes a two-dimensional convolution kernel in an innovative way, demonstrated its ability to improve the process of DVL calibration [10]. Deep-learning frameworks have been used to estimate real-world scenarios of missing DVL beams in partial and complete outage scenarios [11], [12]. Additionally, when all beams are available, the BeamsNet approach [13] offers a more accurate and robust velocity solution using a dedicated deep-learning framework. Deep learning methods were suggested for the fusion process to adaptively estimate the process noise covariance in the inertial DVL fusion process [14], [15]. Recently, an end-to-end deep learning approach was suggested to estimate the AUV acceleration vector, which is introduced to the navigation filter as an additional measurement [16]. In normal operating conditions of the DVL, the BeamsNet approach outperforms model-based approaches. This method employs both inertial measurements and past DVL measurements to estimate the current velocity vector. When updating the navigation filter with this measurement, a cross-correlation arises between the BeamsNet velocity vector measurement and the inertial-based process noise. This process-measurement cross-correlation matrix should be taken into account in the navigation filter to obtain a desired matched filter [17].

In this paper, we employ the inertial/DVL navigation filter based on the extended Kalman filter (EKF), taking into account the process-measurement cross-correlation matrix. The latter is calculated using the inertial and BeamsNet error sources. Using two real-world underwater AUV datasets, we show the necessity of the cross-correlation matrix to allow filter consistency and robustness.

The rest of the paper is organized as follows: Section II formulates the problem and presents the theoretical foundation for incorporating cross-correlations within the extended Kalman filter. Section III introduces the proposed cross-correlation-aware deep INS/DVL fusion framework, detailing the integration of BeamsNet with a modified filter formulation. Section IV presents the experimental results and performance analysis based on real-world AUV trajectories. Finally, Section V concludes the findings in the paper.

*Corresponding author: N. Cohen (email: ncohe140@campus.haifa.ac.il).

II. PROBLEM FORMULATION

A. EKF with Correlated Noise

In the classical derivation of the error-state EKF, it is typically assumed that the process noise and measurement noise are uncorrelated. This is due to the fact that the different sensors provide the information for the process and update. However, in some cases, like the one we introduce in this paper, non-negligible cross-correlations may exist between the process and measurement noise terms. In this section, we present an error-state EKF framework that accounts for such correlation, following the modified Kalman filter equations that explicitly incorporate this dependency.

Let the system dynamics and measurement model be described by the discrete-time equations [18], [19]:

$$x_k = F_{k-1}x_{k-1} + G_{k-1}w_{k-1},$$

where $x_k$ denotes the system state vector at time step $k$, $F_{k-1}$ is the state transition matrix that propagates the state from time $k - 1$ to $k$, and $G_{k-1}$ is the process noise input matrix.

The measurement model is:

$$y_k = H_kx_k + v_k,$$

where, $y_k$ is the measurement vector at time step $k$ and $H_k$ is the observation matrix that maps the state vector to the measurement space. Additionally, $w_k \sim N(0, Q_k)$ denotes the process noise with associated process noise covariance $Q_k$, and $v_k \sim N(0, R_k)$ denotes the measurement noise with associated measurement noise covariance $R_k$.

The cross-correlation matrix between the process and measurement noise covariances is defined as [17]:

$$E[w_kv_j^T] = M_k\delta_{k-j+1},$$

indicating that the process noise at time $k$ is correlated with the measurement noise at time $k+1$. This structure arises naturally in systems where the same external disturbance influences both the system dynamics and the measurement process, albeit with a one-step time lag.

To incorporate this cross-correlation into the Kalman gain computation, the innovation covariance must be adjusted such that the Kalman gain becomes:

$$K_k = (P_k^-H_k^T + M_k) \cdot (H_kP_k^-H_k^T + H_kM_k + M_k^TH_k^T + R_k)^{-1},$$

where $P_k^-$ is the prior error covariance matrix.

Consequently, the posterior error covariance is updated according to:

$$P_k = P_k^- - K_k(H_kP_k^- + M_k^T).$$

These modified expressions account for the non-zero cross-correlation between process and measurement noise, improving filter consistency in scenarios where this assumption is violated. The rest of the error state EKF equations and process remain the same with the exception of the Kalman gain (4) and can be seen in [20], for example.

B. Cross-Correlation within INS/DVL Fusion

Recent advances in deep learning have demonstrated significant potential in time series estimation and sensor fusion, especially in navigation systems where standard model-based filters often struggle with drift, nonlinearity, or degraded measurement conditions. By leveraging the expressive power of deep neural networks (DNNs), it is possible to model complex dependencies between sensor modalities and capture higher-order temporal patterns in the data. In particular, DNN-based approaches that jointly process inertial and acoustic measurements can yield more accurate velocity estimates than classical extended Kalman filtering alone.

Consider a system where the goal is to estimate the vehicle's velocity using both the inertial sensors, which include accelerometers that provide the specific force vector $f_k \in \mathbb{R}^3$ and gyroscopes that measure the angular velocity vector $\omega_k \in \mathbb{R}^3$, and a DVL, which provides beam velocity measurements $z_k^{DVL} \in \mathbb{R}^4$. A deep neural network can be trained to produce a fused velocity estimate via a nonlinear function:

$$\hat{v}_k = \mathcal{F}_\theta(f_{k-T:k}, \omega_{k-T:k}, z_{k-T:k}^{DVL}),$$

where $\mathcal{F}_\theta(\cdot)$ denotes the DNN with parameters $\theta$ and the inputs consist of a temporal window of size $T$. The output $\hat{v}_k$ is the estimated velocity vector at time $k$, typically expressed in the body frame.

The challenge arises from the stochastic properties of the inputs. The IMU measurements $f_k$ and $\omega_k$ are driven by process noise $w_k$, while the DVL beams $z_k^{DVL}$ are corrupted by measurement noise $v_k$. Let us denote:

$$f_k = f_k^{true} + w_k^f$$
$$\omega_k = \omega_k^{true} + w_k^\omega$$
$$z_k^{DVL} = z_k^{true} + v_k$$

where $w_k^f$ and $w_k^\omega$ represent the accelerometer and gyroscopes process noise, respectively, and $v_k$ denotes the DVL measurement noise. Due to the nonlinear mapping, $\mathcal{F}_\theta(\cdot)$, the output velocity estimate becomes a complex function of all the noise sources:

$$\hat{v}_k = \mathcal{F}_\theta(f_k^{true} + w_k^f, \omega_k^{true} + w_k^\omega, z_k^{true} + v_k).$$

Unlike linear estimators, where uncorrelated inputs lead to uncorrelated outputs, the nonlinear dependency structure in $\mathcal{F}_\theta$ causes interactions between $w_k$ and $v_k$. As a result, the effective measurement used for state correction, the output of the DNN, embeds cross-correlations between the process and measurement noise. That is,

$$E[w_kv_k^T] \neq 0,$$

and the distribution of the estimation error becomes analytically intractable due to the black-box nature of the network. Therefore, when such a fused estimate is used as an update measurement in an error-state EKF, the assumption of noise independence no longer holds. This violates the foundational


```mermaid
graph LR
    DVL[DVL] --> BeamsNet
    Inertial[Inertial Sensors] --> BeamsNet
    BeamsNet --> CrossCorrelation[Cross-Correlation]
    DVL --> EKF
    Inertial --> EKF
    CrossCorrelation --> EKF
    EKF --> Position
    EKF --> Velocity
    EKF --> Orientation
```

Fig. 1: Our proposed approach block diagram illustrates how inertial and DVL measurements are utilized as inputs to the BeamsNet framework. This model generates an enhanced velocity vector measurement update, which is then passed, together with the inertial uncertainty, to the cross-correlation block. This block computes the cross-correlation matrix, which is subsequently integrated into the EKF to derive the navigation solution.

assumptions of standard Kalman filtering theory and necessitates either reformulation of the filter to incorporate the cross-covariance or empirical techniques to mitigate its effect.

## III. PROPOSED APPROACH

In this work, we build upon our recent study presented in [13], where a deep learning-based method, named BeamsNet, was proposed as a replacement for the traditional least squares estimation performed by the DVL, with the goal of providing more accurate velocity measurements. BeamsNet employs a one-dimensional convolutional neural network with a multi-head input architecture that incorporates both past and current DVL beam measurements, as well as inertial data, including the specific force and angular velocity measurements. In this work, we leverage BeamsNet and, together with the inertial reading uncertainty, construct the cross-correlation matrix, which in turn is used in the EKF during the INS/DVL fusion. Fig.1 presents the information flow within our proposed approach. The BeamsNet architecture and corresponding hyperparameters are presented in Fig.2.

As discussed in Section II-B, fusing the two sensors that traditionally govern the process and measurement models of the error-state EKF within a deep learning framework introduces cross-correlation between the updated velocity measurements produced by BeamsNet and the process noise derived from the inertial sensors. This cross-correlation may degrade the filter's performance, reduce confidence in the state estimates, and ultimately impact the overall navigation solution. To address this issue, we examine whether accounting for the cross-correlation using the formulation presented in Section II-A can improve underwater navigation performance. This approach results in a cross-correlation-aware deep INS/DVL fusion framework. The matrix Mk, defined in (3), plays a key role in this formulation. While the inertial sensor process and measurement noise characteristics are provided in their respective datasheets, BeamsNet uncertainty characteristics are not provided by the network. Therefore, Mk was determined

```mermaid
graph TD
    A[Input Layer] --> B[1D Conv + Tanh]
    B --> C[Flatten]
    C --> D[FC Layer + ReLU]
    D --> E[Dropout Layer]
    E --> F[Output Layer]
```

Fig. 2: BeamsNet architecture where raw accelerometer and gyroscope measurements are first passed through parallel one-dimensional convolutional layers, each with six filters of size 2 × 1, to extract temporal features from the inertial data. The resulting features are flattened and concatenated, then passed through a dropout layer. A sequence of fully connected layers processes the combined features. Finally, the current DVL measurement is concatenated with the network output and fed into the last fully connected layer, which outputs a 3 × 1 estimated DVL velocity vector.

numerically. To construct a numerical approximation of the cross-covariance between the process noise wk ~ N(0, Qk) and the measurement noise vk ~ N(0, Rk), we assume that both covariance matrices are diagonal and define the cross-correlation matrix as:

$$E[w_kv_k^T] = \rho \cdot \sqrt{diag(Q_k)} \cdot (\sqrt{diag(R_k)})^T \quad (12)$$

where ρ ∈ [0, 1] is a scalar correlation coefficient.

Fig. 3: Two AUV trajectories presented in the North-East plane: the trajectory shown in (a) is referred to as "Trajectory #1," and the trajectory in (b) as "Trajectory #2."

| (a) | (b) |
|-----|-----|
| Trajectory #1 | Trajectory #2 |

IV. ANALYSIS AND RESULTS

To evaluate the performance of the cross-correlation-aware deep INS/DVL fusion, we used an AUV dataset introduced in the original BeamsNet paper, which is also publicly available through the BeamsNet GitHub repository. Approximately four hours of data were collected from an AUV performing various missions in the Mediterranean Sea and used to train and validate the network. This dataset was used to train the BeamsNet network. For testing, we utilized a different publicly available dataset introduced in [14], which can be accessed through the corresponding GitHub repository. We used two distinct 400-second-long missions, each exhibiting different characteristics and sea state conditions. Both were used to first evaluate the robustness of the BeamsNet approach and then assess the performance of the cross-correlation-aware method in comparison to the approach that neglects cross-correlations. The two trajectories used in the evaluation are shown in Fig.3.

To evaluate BeamsNet's performance on the unseen data, we employed the following metrics:

$$RMSE(x_i, \hat{x}_i) = \sqrt{\frac{\sum_{i=1}^N (x_i - \hat{x}_i)^2}{N}}$$ (13)

$$MAE(x_i, \hat{x}_i) = \frac{\sum_{i=1}^N |x_i - \hat{x}_i|}{N}$$ (14)

$$R^2(x_i, \hat{x}_i) = 1 - \frac{\sum_{i=1}^N (x_i - \hat{x}_i)^2}{\sum_{i=1}^N (x_i - \bar{x}_i)^2}$$ (15)

$$VAF(x_i, \hat{x}_i) = [1 - \frac{var(x_i - \hat{x}_i)}{var(x_i)}] \times 100$$ (16)

in this formulation, N denotes the total number of samples. The term x_i corresponds to the ground truth norm of the DVL-derived velocity vector, while x̂_i refers to the predicted velocity norm. The quantity x̄_i indicates the average value of the ground truth velocity norm. The function var denotes the variance. An ideal model would yield a VAF of 100, an R^2 value of 1, and both RMSE and MAE equal to zero, reflecting perfect predictive performance.

The results of the velocity vector estimation are summarized in Table I for trajectory #1 and Table II for trajectory #2.

TABLE I: Comparison of velocity estimation performance for Trajectory #1 using the least squares method and the Beam-sNet approach. The metrics include RMSE, MAE, coefficient of determination R^2, and variance accounted for (VAF).

| Method / Metric | RMSE [m/s] | MAE [m/s] | R^2 | VAF [%] |
|-----------------|------------|-----------|-----|---------|
| LS | 0.013494 | 0.012888 | 0.997273 | 99.976048 |
| BeamsNet (ours) | 0.004015 | 0.003171 | 0.999759 | 99.975878 |

TABLE II: Comparison of velocity estimation performance for Trajectory #2 using the least squares method and the Beam-sNet approach. The metrics include RMSE, MAE, coefficient of determination R^2, and variance accounted for (VAF).

| Method / Metric | RMSE [m/s] | MAE [m/s] | R^2 | VAF [%] |
|-----------------|------------|-----------|-----|---------|
| LS | 0.015122 | 0.014651 | 0.988125 | 99.927172 |
| BeamsNet (ours) | 0.003927 | 0.003047 | 0.999199 | 99.919904 |

The BeamsNet method maintained high accuracy, which was consistent with the results reported in the original study. The statistical fit of the model, as indicated by the R^2 and VAF metrics, remained high. In terms of RMSE, BeamsNet achieved an improvement of approximately 70% over the model-based approach for Trajectory #1, and around 74% for Trajectory #2. These results demonstrate the robustness of BeamsNet, as it performs reliably even on previously unseen data.

Once the superior performance of the data-driven approach over the least squares method was established, we proceeded to examine the accuracy of the overall navigation solution within the error-state EKF, considering both the case where cross-correlation is neglected and the case where a cross-correlation-

through trial and error and set to ρ = 0.42. Next, we examined the error in the state vector when comparing with and without the cross-correlation matrix. Those differences were negligible when examining the average error produced by the filters in the two scenarios. However, a notable distinction emerged in the estimated covariance: the cross-correlation-aware method exhibited higher confidence in the state estimation. This resulted in lower values for the error-state covariance. Fig. 4 and Fig. 5 present the standard deviations of the velocity, misalignment angles, accelerometer, and gyroscope biases error states for trajectory #1 and trajectory #2, respectively. It can be observed that when cross-correlation is properly accounted for, the standard deviation across all states is either comparable or, more frequently, lower in magnitude. This indicates increased confidence in the model's estimates. To quantify this, we examined each group of three-axis states: velocity, misalignment angles, accelerometer bias, and gyroscope bias. For each group, we first computed the average standard deviation of the state estimates over time, across the three axes of each

Fig. 4: Trajectory #1 standard deviation of the states when comparing the cross-correlation-aware approach (red) to the one that neglects it (black).

(a) Velocity standard deviation (b) Standard deviation of mis-
in the North, East, and Down alignment angles: roll, pitch,
axes. and yaw (from top to bottom).

(c) Standard deviation of ac- (d) Standard deviation of gy-
celerometer bias estimates in roscope bias estimates in the
the body frame. body frame.

Fig. 5: Trajectory #2 standard deviation of the states when comparing the cross-correlation-aware approach (red) to the one that neglects it (black).

(a) Velocity standard deviation (b) Standard deviation of mis-
in the North, East, and Down alignment angles: roll, pitch,
axes. and yaw (from top to bottom).

(c) Standard deviation of ac- (d) Standard deviation of gy-
celerometer bias estimates in roscope bias estimates in the
the body frame. body frame.

Fig. 6: Quantitative evaluation of filter uncertainty for (a) Trajectory #1 and (b) Trajectory #2. For each three-axis state group (velocity, misalignment angles, accelerometer bias, and gyroscope bias), we compute the average standard deviation over time, as well as the final summed standard deviation across axes. The bars indicate the percentage improvement achieved by the cross-correlation-aware method.

| State Group | Avg. Velocities [m/s] | Final Velocities Value [m/s] | Avg. Angles [rad] | Final Angles Value [rad] | Avg. Acc Biases [m/s²] | Final Acc Biases Value [m/s²] | Avg. Gyro Biases [rad/s] | Final Gyro Biases Value [rad/s] |
|-------------|----------------------|----------------------------|-------------------|--------------------------|------------------------|------------------------------|--------------------------|--------------------------------|
| Trajectory #1 | 17.5 | 15.0 | 12.5 | 10.0 | 7.5 | 2.5 | 1.5 | 1.0 |
| Trajectory #2 | 17.5 | 15.0 | 7.5 | 10.0 | 2.5 | 4.0 | 1.5 | 1.0 |

aware formulation is employed. To begin, we first found the scalar correlation coefficient, which was manually determined

state. Then, we summed the final standard deviation values derived from the estimated covariance matrix across the three axes of each state. These two measures indicate, respectively, whether there is an overall reduction in uncertainty over time and whether the filter converged to a lower value, as well as the percentage of improvement. All findings are summarized in Fig. 6. It can be observed that, for both trajectories, the cross-correlation-aware method outperforms the approach in which cross-correlation is neglected. Notable improvements are observed in the velocity and misalignment angle states, with a general improvement over time exceeding 10% in most scenarios.

## V. CONCLUSIONS

This work introduced a cross-correlation-aware deep INS/DVL fusion framework that integrates the strengths of both data-driven and model-based approaches. First, we built upon a previous work called BeamsNet and showed its robustness to unseen data. Then, by incorporating deep learning-based velocity estimates into an error-state EKF with an explicit cross-covariance model, we achieved a solution that is not only superior in terms of accuracy when compared to the model-based least squares approach but also more consistent and theoretically grounded. The proposed method addresses a critical limitation of traditional EKF formulations, namely the assumption of uncorrelated process and measurement noise, a condition often violated when using data-driven measurements. Our results demonstrate that accounting for these correlations yields improved confidence in state estimates and reduced uncertainty over time.

Beyond its empirical advantages, this approach offers a principled pathway for integrating modern deep learning techniques within the well-established Kalman filtering framework. This synergy is especially crucial in real-time underwater navigation applications, where reliability, robustness, and theoretical soundness are essential for operational success.

## ACKNOWLEDGMENTS

N.C. is supported by the Maurice Hatter Foundation and University of Haifa presidential scholarship for outstanding students on a direct Ph.D. track.

## REFERENCES

[1] J. Nicholson and A. Healey, "The present state of autonomous underwater vehicle (AUV) applications and technologies," Marine Technology Society Journal, vol. 42, no. 1, pp. 44–51, 2008.

[2] G. Griffiths, Technology and applications of autonomous underwater vehicles, vol. 2. CRC Press, 2002.

[3] P. A. Miller, J. A. Farrell, Y. Zhao, and V. Djapic, "Autonomous underwater vehicle navigation," IEEE Journal of Oceanic Engineering, vol. 35, no. 3, pp. 663–678, 2010.

[4] D. Rudolph and T. A. Wilson, "Doppler Velocity Log theory and preliminary considerations for design and construction," in 2012 Proceedings of IEEE Southeastcon, pp. 1–7, IEEE, 2012.

[5] N. Cohen and I. Klein, "Inertial navigation meets deep learning: A survey of current trends and future directions," Results in Engineering, p. 103565, 2024.

[6] F. Zhang, S. Zhao, L. Li, and C. Cao, "Underwater DVL Optimization Network (UDON): A Learning-Based DVL Velocity Optimizing Method for Underwater Navigation," Drones, vol. 9, no. 1, p. 56, 2025.

[7] Liu, Peijia and Wang, Bo and Li, Guanghua and Hou, Dongdong and Zhu, Zhengyu and Wang, Zhongyong, "Sins/dvl integrated navigation method with current compensation using rbf neural network," IEEE Sensors Journal, vol. 22, no. 14, pp. 14366–14377, 2022.

[8] E. Topini, F. Fanelli, A. Topini, M. Pebody, A. Ridolfi, A. B. Phillips, and B. Allotta, "An experimental comparison of Deep Learning strategies for AUV navigation in DVL-denied environments," Ocean Engineering, vol. 274, p. 114034, 2023.

[9] R. Makam, M. Pramuk, S. Thomas, and S. Sundaram, "Spectrally Normalized Memory Neuron Network Based Navigation for Autonomous Underwater Vehicles in DVL-Denied Environment," in OCEANS 2024-Singapore, pp. 1–6, IEEE, 2024.

[10] Z. Yampolsky and I. Klein, "DCNet: A data-driven framework for DVL calibration," Applied Ocean Research, vol. 158, p. 104525, 2025.

[11] M. Yona and I. Klein, "MissBeamNet: Learning missing Doppler velocity log beam measurements," Neural Computing and Applications, vol. 36, no. 9, pp. 4947–4958, 2024.

[12] N. Cohen, Z. Yampolsky, and I. Klein, "Set-transformer BeamsNet for AUV velocity forecasting in complete DVL outage scenarios," in 2023 IEEE Underwater Technology (UT), pp. 1–6, IEEE, 2023.

[13] N. Cohen and I. Klein, "BeamsNet: A data-driven approach enhancing Doppler velocity log measurements for autonomous underwater vehicle navigation," Engineering Applications of Artificial Intelligence, vol. 114, p. 105216, 2022.

[14] N. Cohen and I. Klein, "Adaptive Kalman-Informed Transformer," Engineering Applications of Artificial Intelligence, vol. 146, p. 110221, 2025.

[15] A. Levy and I. Klein, "Adaptive Neural Unscented Kalman Filter," arXiv preprint arXiv:2503.05490, 2025.

[16] Y. Stolero and I. Klein, "AUV Acceleration Prediction Using DVL and Deep Learning ," arXiv preprint arXiv: 2503.16573, 2025.

[17] D. Simon, Optimal state estimation: Kalman, H infinity, and nonlinear approaches. John Wiley & Sons, 2006.

[18] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with applications to tracking and navigation: theory algorithms and software. John Wiley & Sons, 2004.

[19] P. Groves, Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition. GNSS/GPS, Artech House, 2013.

[20] J. Farrell, Aided navigation: GPS with high rate sensors. McGraw-Hill, Inc., 2008.
![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image92-page3.png)

**Caption:**  Fig. 1: Our proposed approach block diagram illustrates how inertial and DVL measurements are utilized as inputs to the BeamsNet framework. This model generates an enhanced velocity vector measurement update, which is then passed, together with the inertial uncertainty, to the cross-correlation block. This block computes the cross-correlation matrix, which is subsequently integrated into the EKF to derive the navigation solution. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image125-page3.png)

**Caption:** Fig. 1: Our proposed approach block diagram illustrates how inertial and DVL measurements are utilized as inputs to the BeamsNet framework. This model generates an enhanced velocity vector measurement update, which is then passed, together with the inertial uncertainty, to the cross-correlation block. This block computes the cross-correlation matrix, which is subsequently integrated into the EKF to derive the navigation solution.  Fig. 2: BeamsNet architecture where raw accelerometer and gyroscope measurements are first passed through parallel one- dimensional convolutional layers, each with six filters of size 2 × 1, to extract temporal features from the inertial data. The resulting features are flattened and concatenated, then passed through a dropout layer. A sequence of fully connected layers processes the combined features. Finally, the current DVL measurement is concatenated with the network output and fed into the last fully connected layer, which outputs a 3 × 1 estimated DVL velocity vector. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image129-page4.png)

**Caption:**  (a) (b) 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image130-page4.png)

**Caption:**  (a) (b) 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image146-page5.png)

**Caption:**  (a) Velocity standard deviation in the North, East, and Down axes. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image147-page5.png)

**Caption:**  (a) Velocity standard deviation in the North, East, and Down axes. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image148-page5.png)

**Caption:** (a) Velocity standard deviation in the North, East, and Down axes.  (c) Standard deviation of ac- celerometer bias estimates in the body frame. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image149-page5.png)

**Caption:** (a) Velocity standard deviation in the North, East, and Down axes.  (c) Standard deviation of ac- celerometer bias estimates in the body frame. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image150-page5.png)

**Caption:** (c) Standard deviation of ac- celerometer bias estimates in the body frame.  (a) Velocity standard deviation in the North, East, and Down axes. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image151-page5.png)

**Caption:** (c) Standard deviation of ac- celerometer bias estimates in the body frame.  (a) Velocity standard deviation in the North, East, and Down axes. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image152-page5.png)

**Caption:** (a) Velocity standard deviation in the North, East, and Down axes.  (c) Standard deviation of ac- celerometer bias estimates in the body frame. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image153-page5.png)

**Caption:** (a) Velocity standard deviation in the North, East, and Down axes.  (c) Standard deviation of ac- celerometer bias estimates in the body frame. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image154-page5.png)

**Caption:** (c) Standard deviation of ac- celerometer bias estimates in the body frame.  (a) Velocity standard deviation in the North, East, and Down axes. 


![Image](C:\Users\Aditya\OneDrive\Desktop\NvidiaTraining\Training_Material_(2024-25)\5th_March_2025\output\pdf1\image155-page5.png)

**Caption:**  Fig. 5: Trajectory #2 standard deviation of the states when comparing the cross-correlation-aware approach (red) to the one that neglects it (black). 

