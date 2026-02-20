# ML_Projects_and_Resources
This repo contains my machine learning course projects and curated learning resources, covering supervised, unsupervised, semi-supervised, reinforcement learning, neural networks, and practical approaches. Organized for easy exploration and skill demonstration.

---

# Machine Learning Learning Resources

This folders contains a carefully structured collection of curated learning materials covering the theoretical foundations and practical implementation of Machine Learning.

The goal of this section is to build a strong conceptual base while simultaneously developing implementation fluency.

The structure follows a progressive learning path — starting from supervised learning fundamentals and advancing toward reinforcement learning and neural networks.

---


---

# 01 — Supervised Learning

This section covers the foundations of supervised learning, where models learn from labeled data.

### Topics Covered
- Classification vs Regression
- Training vs Testing Data
- Overfitting and Underfitting
- Bias-Variance Tradeoff
- Model Evaluation Metrics
- Cross-Validation
- Data Preprocessing Techniques

### Focus
Building intuition behind predictive modeling and generalization.

---

# 02 — Regression

This section focuses specifically on regression algorithms for continuous target prediction.

### Topics Covered
- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Regularization:
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
- Assumptions of Linear Models
- Residual Analysis
- Evaluation Metrics:
  - MSE
  - RMSE
  - MAE
  - R²

### Focus
Understanding linear modeling assumptions and controlling model complexity.

---

# 03 — Decision Trees and Ensemble Methods

This section covers tree-based models and advanced ensemble techniques.

### Topics Covered
- Decision Trees (Classification & Regression)
- Information Gain
- Gini Impurity
- Tree Pruning
- Random Forest
- Bagging
- Boosting
- AdaBoost
- Gradient Boosting
- Bias vs Variance in Ensembles

### Focus
Improving model performance through aggregation and variance reduction.

---

# 04 — Unsupervised Learning

This section explores pattern discovery without labeled data.

### Topics Covered
- Clustering:
  - K-Means
  - Hierarchical Clustering
  - DBSCAN
- Dimensionality Reduction:
  - PCA
- Distance Metrics
- Silhouette Score
- Data Scaling for Clustering

### Focus
Extracting structure and hidden patterns from unlabeled datasets.

---

# 05 — Semi-Supervised Learning

This section introduces learning techniques that combine labeled and unlabeled data.

### Topics Covered
- Self-training
- Label Propagation
- Graph-based Methods
- Practical Use Cases
- When to Use Semi-Supervised Learning

### Focus
Leveraging partially labeled datasets efficiently.

---

# 06 — Reinforcement Learning

This section introduces sequential decision-making and reward-based learning systems.

### Topics Covered
- Markov Decision Process (MDP)
- Policy vs Value Function
- Q-Learning
- Exploration vs Exploitation
- Temporal Difference Learning
- Reward Engineering Basics

### Focus
Understanding learning through interaction and reward optimization.

---

# 07 — Neural Networks

This section covers the foundations of neural computation and deep learning basics.

### Topics Covered
- Perceptron
- Activation Functions
- Backpropagation
- Gradient Descent
- Loss Functions
- Vanishing Gradient Problem
- Introduction to Deep Learning

### Focus
Building intuition for non-linear function approximation and representation learning.

---

# 08 — Practical Approach (Hands-On ML Book)

Primary reference:
"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" – Aurélien Géron

### Focus Areas
- Practical implementation with scikit-learn
- End-to-end ML projects
- Production mindset
- Pipeline construction
- Feature engineering
- Model deployment concepts

This section emphasizes real-world application over purely theoretical discussion.

---

# Learning Philosophy

This resource collection is structured to:

- Build theoretical clarity before implementation
- Reinforce mathematics behind algorithms
- Encourage experimentation
- Promote disciplined model evaluation
- Develop strong ML fundamentals before advancing to deep learning

---

# Intended Outcome

After completing these resources, the learner should:

- Understand core ML paradigms deeply
- Implement classical ML algorithms confidently
- Evaluate models properly
- Recognize bias-variance tradeoffs
- Apply ML concepts to real datasets
- Transition smoothly into advanced AI topics

---

**Maintained as part of a structured Machine Learning learning roadmap.**


# Projects

## 1️⃣ Marital_Status_Prediction_LDA  
**Category:** Supervised Learning – Binary Classification  

### Objective
Predict marital status using socio-academic features from the *bioChemists* dataset.

### Key Concepts
- Linear Discriminant Analysis (LDA)
- One-Hot Encoding
- Train/Test Split
- ROC Curve Analysis
- Feature Engineering

### Skills Demonstrated
- Building end-to-end classification pipelines  
- Handling categorical data  
- Model evaluation using classification metrics  
- Working with structured tabular datasets  

---

## 2️⃣ Iris_Species_Classification_SVM_KFold  
**Category:** Supervised Learning – Multi-Class Classification  

### Objective
Classify iris flower species using Support Vector Machines with systematic cross-validation.

### Key Concepts
- Label Encoding
- Support Vector Machines (SVM)
- Kernel comparison (Linear, RBF, Polynomial, Sigmoid)
- Hyperparameter exploration (C parameter)
- K-Fold Cross Validation

### Skills Demonstrated
- Multi-class classification  
- Model comparison and validation strategies  
- Hyperparameter experimentation  
- Understanding kernel-based learning  

---

## 3️⃣ Titanic_Survival_Prediction_DecisionTree  
**Category:** Supervised Learning – Binary Classification  

### Objective
Predict passenger survival on the Titanic dataset using Decision Trees.

### Key Concepts
- Data cleaning and preprocessing
- Handling missing values
- Encoding categorical features
- Tree-based learning
- Overfitting vs Underfitting analysis
- Hyperparameter control:
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`

### Skills Demonstrated
- Structured preprocessing pipeline  
- Model interpretability with tree-based methods  
- Understanding bias-variance tradeoff  
- Model complexity analysis  

---

## 4️⃣ Loan_Approval_Prediction_Ensemble_Methods  
**Category:** Ensemble Learning – Classification  

### Objective
Predict loan approval status using multiple ensemble techniques.

### Implemented Models
- Bagging Classifier
- Random Forest
- AdaBoost
- Gradient Boosting

### Key Concepts
- Weak learners
- Bias vs Variance reduction
- Ensemble aggregation
- Comparative model analysis

### Skills Demonstrated
- Implementation of advanced ensemble methods  
- Understanding boosting vs bagging  
- Practical application of bias-variance tradeoff  
- Structured model comparison  

---

## 5️⃣ Wholesale_Customer_Segmentation_Clustering  
**Category:** Unsupervised Learning – Clustering  

### Objective
Segment wholesale customers based on annual spending behavior to support marketing strategy optimization.

### Key Concepts
- K-Means Clustering
- Data scaling and normalization
- Customer segmentation
- Cluster interpretation
- Pattern discovery without labeled data

### Skills Demonstrated
- Unsupervised learning workflows  
- Behavioral data analysis  
- Business-oriented ML application  
- Interpreting clustering results for strategy insights  

---

# Technical Stack

- Python  
- pandas  
- scikit-learn  
- matplotlib  
- pydataset  

---

# Machine Learning Coverage

This repository covers the core foundations of machine learning:

- Supervised Learning (Classification)
- Tree-Based Models
- Ensemble Methods
- Cross Validation
- Hyperparameter Analysis
- Unsupervised Learning (Clustering)
- Feature Engineering
- Data Preprocessing Pipelines

---

# Purpose of This Repository

This repository serves as:

- A structured ML portfolio
- A practical demonstration of algorithm implementation
- A reference collection of organized learning materials
- Evidence of applied machine learning competency

---

## Author

Arman Esmaeili - AI enthusiast 
Focused on building strong foundations in classical ML before advancing into more complex domains.
